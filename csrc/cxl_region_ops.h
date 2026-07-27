// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

class CudaRegionRegistration {
 public:
  CudaRegionRegistration(const std::string& shm_name, size_t expected_capacity,
                         size_t payload_offset = 4096);
  ~CudaRegionRegistration();

  CudaRegionRegistration(const CudaRegionRegistration&) = delete;
  CudaRegionRegistration& operator=(const CudaRegionRegistration&) = delete;

  size_t capacity() const;
  uintptr_t device_address(size_t offset, size_t length) const;
  void copy_from_device(uintptr_t source, size_t offset, size_t length,
                        uintptr_t stream_ptr) const;
  void copy_to_device(uintptr_t destination, size_t offset, size_t length,
                      uintptr_t stream_ptr) const;
  void close();

 private:
  void validate_range(size_t offset, size_t length) const;

  int fd_ = -1;
  void* mapping_ = nullptr;
  void* payload_host_ = nullptr;
  void* payload_device_ = nullptr;
  size_t mapping_size_ = 0;
  size_t capacity_ = 0;
  bool registered_ = false;
  mutable std::mutex mutex_;
};
