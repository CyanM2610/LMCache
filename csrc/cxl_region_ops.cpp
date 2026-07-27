// SPDX-License-Identifier: Apache-2.0

#include "cxl_region_ops.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t kRegionHeaderSize = 4096;
constexpr std::array<char, 8> kRegionMagic = {'B', 'L', 'G', 'C',
                                              'X', 'L', 'R', 'G'};
constexpr uint32_t kRegionVersion = 1;

#pragma pack(push, 1)
struct RegionHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint64_t capacity;
  uint64_t alignment;
};

struct CxlMemSimRegionHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t total_size;
  uint64_t data_offset;
  uint64_t metadata_offset;
  uint64_t num_cachelines;
  uint64_t base_addr;
};
#pragma pack(pop)

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

CudaRegionRegistration::CudaRegionRegistration(const std::string& shm_name,
                                               size_t expected_capacity,
                                               size_t payload_offset) {
  fd_ = shm_open(shm_name.c_str(), O_RDWR, 0);
  if (fd_ < 0) {
    throw std::runtime_error("failed to open POSIX shared region " + shm_name);
  }
  try {
    struct stat stat_buffer {};
    if (fstat(fd_, &stat_buffer) != 0 ||
        stat_buffer.st_size < static_cast<off_t>(sizeof(RegionHeader))) {
      throw std::runtime_error("shared region is smaller than its header");
    }
    mapping_size_ = static_cast<size_t>(stat_buffer.st_size);
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      throw std::runtime_error("failed to mmap POSIX shared region");
    }

    if (payload_offset > mapping_size_ ||
        expected_capacity > mapping_size_ - payload_offset) {
      throw std::runtime_error("shared region capacity does not match config");
    }
    const auto* header = static_cast<const RegionHeader*>(mapping_);
    const bool beluga_header =
        std::memcmp(header->magic, kRegionMagic.data(), kRegionMagic.size()) ==
            0 &&
        header->version == kRegionVersion &&
        header->header_size == payload_offset &&
        header->capacity == expected_capacity &&
        is_power_of_two(header->alignment);
    constexpr uint64_t kCxlMemSimMagic = 0x43584C4D454D5348ULL;
    const auto* cxlmemsim =
        static_cast<const CxlMemSimRegionHeader*>(mapping_);
    const bool cxlmemsim_header =
        cxlmemsim->magic == kCxlMemSimMagic && cxlmemsim->version == 1 &&
        cxlmemsim->total_size == mapping_size_ &&
        cxlmemsim->data_offset == payload_offset &&
        cxlmemsim->num_cachelines <=
            std::numeric_limits<uint64_t>::max() / 64 &&
        cxlmemsim->num_cachelines * 64 == expected_capacity;
    if (!beluga_header && !cxlmemsim_header) {
      throw std::runtime_error("shared region header is incompatible");
    }
    capacity_ = expected_capacity;
    payload_host_ = static_cast<char*>(mapping_) + payload_offset;
    check_cuda(
        cudaHostRegister(payload_host_, capacity_,
                         cudaHostRegisterMapped | cudaHostRegisterPortable),
        "cudaHostRegister");
    registered_ = true;
    check_cuda(cudaHostGetDevicePointer(&payload_device_, payload_host_, 0),
               "cudaHostGetDevicePointer");
  } catch (...) {
    close();
    throw;
  }
}

CudaRegionRegistration::~CudaRegionRegistration() { close(); }

size_t CudaRegionRegistration::capacity() const { return capacity_; }

uintptr_t CudaRegionRegistration::device_address(size_t offset,
                                                 size_t length) const {
  std::lock_guard<std::mutex> guard(mutex_);
  validate_range(offset, length);
  return reinterpret_cast<uintptr_t>(payload_device_) + offset;
}

void CudaRegionRegistration::copy_from_device(uintptr_t source, size_t offset,
                                              size_t length,
                                              uintptr_t stream_ptr) const {
  std::lock_guard<std::mutex> guard(mutex_);
  validate_range(offset, length);
  auto* destination = static_cast<char*>(payload_host_) + offset;
  check_cuda(cudaMemcpyAsync(destination, reinterpret_cast<const void*>(source),
                             length, cudaMemcpyDeviceToHost,
                             reinterpret_cast<cudaStream_t>(stream_ptr)),
             "cudaMemcpyAsync device-to-region");
}

void CudaRegionRegistration::copy_to_device(uintptr_t destination,
                                            size_t offset, size_t length,
                                            uintptr_t stream_ptr) const {
  std::lock_guard<std::mutex> guard(mutex_);
  validate_range(offset, length);
  const auto* source = static_cast<const char*>(payload_host_) + offset;
  check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(destination), source,
                             length, cudaMemcpyHostToDevice,
                             reinterpret_cast<cudaStream_t>(stream_ptr)),
             "cudaMemcpyAsync region-to-device");
}

void CudaRegionRegistration::close() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (registered_) {
    cudaHostUnregister(payload_host_);
    registered_ = false;
  }
  payload_device_ = nullptr;
  payload_host_ = nullptr;
  capacity_ = 0;
  if (mapping_ != nullptr) {
    munmap(mapping_, mapping_size_);
    mapping_ = nullptr;
    mapping_size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void CudaRegionRegistration::validate_range(size_t offset,
                                            size_t length) const {
  if (!registered_) {
    throw std::runtime_error("CUDA shared region is closed");
  }
  if (length == 0 || offset > capacity_ || length > capacity_ - offset) {
    throw std::invalid_argument("extent exceeds CUDA shared region bounds");
  }
}
