# SPDX-License-Identifier: Apache-2.0
"""Provision stable POSIX shared-memory regions for the CXL proxy."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Protocol
import os
import struct
import uuid


REGION_HEADER_SIZE = 4096
_REGION_HEADER = struct.Struct("<8sIIQQ")
_REGION_MAGIC = b"BLGCXLRG"
_REGION_VERSION = 1
_CXLMEMSIM_HEADER = struct.Struct("<QQQQQQQ")
_CXLMEMSIM_MAGIC = 0x43584C4D454D5348
_CXLMEMSIM_VERSION = 1


def pack_region_header(capacity: int, alignment: int) -> bytes:
    """Encode the shared POSIX region ABI header.

    Args:
        capacity: Positive payload capacity in bytes.
        alignment: Positive power-of-two payload alignment.

    Returns:
        Binary header prefix to write at offset zero.

    Raises:
        ValueError: If capacity or alignment is invalid.
    """
    if capacity <= 0:
        raise ValueError("region capacity must be positive")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("region alignment must be a power of two")
    return _REGION_HEADER.pack(
        _REGION_MAGIC,
        _REGION_VERSION,
        REGION_HEADER_SIZE,
        capacity,
        alignment,
    )


@dataclass(frozen=True)
class RegionHandle:
    """Process-independent description of one shared backing region."""

    region_id: str
    shm_name: str
    capacity: int
    alignment: int
    capabilities: frozenset[str]
    data_offset: int = REGION_HEADER_SIZE

    def __post_init__(self) -> None:
        if not self.region_id or not self.shm_name:
            raise ValueError("region identity must not be empty")
        if self.capacity <= 0:
            raise ValueError("region capacity must be positive")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("region alignment must be a power of two")
        if self.data_offset < 0:
            raise ValueError("region data offset must be non-negative")


class RegionProvider(Protocol):
    """Provider boundary for replaceable CXL proxy backing storage."""

    def provision(self) -> RegionHandle:
        """Return the stable region handle or fail without a fallback.

        Returns:
            Immutable process-independent region metadata.

        Raises:
            OSError: If the configured region cannot be opened.
            RuntimeError: If its header or capacity is incompatible.
        """
        ...

    def close(self) -> None:
        """Release process-local mapping resources."""
        ...


class PosixShmRegionProvider:
    """Open and validate a pre-created POSIX shared-memory proxy region."""

    def __init__(
        self,
        *,
        region_id: str,
        shm_name: str,
        expected_capacity: int | None = None,
    ) -> None:
        self._region_id = region_id
        self._shm_name = shm_name
        self._expected_capacity = expected_capacity
        self._fd: int | None = None
        self._handle: RegionHandle | None = None

    def provision(self) -> RegionHandle:
        """Open and validate the named shared-memory region once.

        Returns:
            The same immutable handle on every successful call.

        Raises:
            FileNotFoundError: If the configured POSIX SHM object does not exist.
            ValueError: If ``shm_name`` is not a valid POSIX SHM name.
            RuntimeError: If the region header, size, or expected capacity is
                incompatible. No anonymous replacement is created.
        """
        if self._handle is not None:
            return self._handle
        if not self._shm_name.startswith("/") or "/" in self._shm_name[1:]:
            raise ValueError("shm_name must be a POSIX SHM name")
        fd = os.open(f"/dev/shm{self._shm_name}", os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            if size < REGION_HEADER_SIZE:
                raise RuntimeError("shared region is smaller than its header")
            header = os.pread(fd, _REGION_HEADER.size, 0)
            if len(header) != _REGION_HEADER.size:
                raise RuntimeError("shared region header is truncated")
            magic, version, header_size, capacity, alignment = _REGION_HEADER.unpack(
                header
            )
            if (
                magic != _REGION_MAGIC
                or version != _REGION_VERSION
                or header_size != REGION_HEADER_SIZE
                or capacity <= 0
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                raise RuntimeError("shared region header is incompatible")
            if size < REGION_HEADER_SIZE + capacity:
                raise RuntimeError("shared region size is smaller than header capacity")
            if (
                self._expected_capacity is not None
                and capacity != self._expected_capacity
            ):
                raise RuntimeError(
                    "shared region header capacity does not match config"
                )
            self._fd = fd
            self._handle = RegionHandle(
                region_id=self._region_id,
                shm_name=self._shm_name,
                capacity=capacity,
                alignment=alignment,
                capabilities=frozenset({"cuda_host_register_v1"}),
                data_offset=REGION_HEADER_SIZE,
            )
            return self._handle
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        """Close the process-local descriptor without unlinking shared storage."""
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._handle = None


class LocalDRAMRegionProvider:
    """Create one server-owned POSIX SHM region backed by local DRAM."""

    def __init__(self, *, capacity: int, alignment: int) -> None:
        """Configure a local CUDA-registerable DRAM region.

        Args:
            capacity: Positive payload capacity in bytes.
            alignment: Positive power-of-two extent alignment.

        Raises:
            ValueError: If capacity or alignment is invalid.
        """
        if capacity <= 0:
            raise ValueError("DRAM region capacity must be positive")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("DRAM region alignment must be a power of two")
        self._capacity = capacity
        self._alignment = alignment
        self._shm_name = f"/lmcache-dram-{os.getpid()}-{uuid.uuid4().hex}"
        self._fd: int | None = None
        self._handle: RegionHandle | None = None

    def provision(self) -> RegionHandle:
        """Create and initialize the local DRAM backing region.

        Returns:
            Stable process-independent metadata for CUDA registration.

        Raises:
            OSError: If exclusive POSIX SHM creation or initialization fails.
        """
        if self._handle is not None:
            return self._handle
        path = f"/dev/shm{self._shm_name}"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, REGION_HEADER_SIZE + self._capacity)
            header = pack_region_header(self._capacity, self._alignment)
            if os.pwrite(fd, header, 0) != len(header):
                raise OSError("local DRAM region header write was incomplete")
            self._fd = fd
            self._handle = RegionHandle(
                region_id=f"dram-local:{self._shm_name}",
                shm_name=self._shm_name,
                capacity=self._capacity,
                alignment=self._alignment,
                capabilities=frozenset({"cuda_host_register_v1", "local_dram_v1"}),
            )
            return self._handle
        except BaseException:
            os.close(fd)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            raise

    def close(self) -> None:
        """Close and unlink the server-owned DRAM region."""
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._handle = None
        try:
            os.unlink(f"/dev/shm{self._shm_name}")
        except FileNotFoundError:
            pass


class CXLMemSimShmRegionProvider:
    """Validate the data region provisioned by a CXLMemSim SHM server."""

    def __init__(
        self,
        *,
        region_id: str,
        shm_name: str,
        expected_capacity: int | None = None,
    ) -> None:
        self._region_id = region_id
        self._shm_name = shm_name
        self._expected_capacity = expected_capacity
        self._fd: int | None = None
        self._handle: RegionHandle | None = None

    def provision(self) -> RegionHandle:
        """Open a page-aligned CXLMemSim data mapping without fallback."""
        if self._handle is not None:
            return self._handle
        if not self._shm_name.startswith("/") or "/" in self._shm_name[1:]:
            raise ValueError("shm_name must be a POSIX SHM name")
        fd = os.open(f"/dev/shm{self._shm_name}", os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            if size < _CXLMEMSIM_HEADER.size:
                raise RuntimeError("CXLMemSim shared region header is truncated")
            encoded = os.pread(fd, _CXLMEMSIM_HEADER.size, 0)
            if len(encoded) != _CXLMEMSIM_HEADER.size:
                raise RuntimeError("CXLMemSim shared region header is truncated")
            (
                magic,
                version,
                total_size,
                data_offset,
                _metadata_offset,
                num_cachelines,
                _base_addr,
            ) = _CXLMEMSIM_HEADER.unpack(encoded)
            capacity = num_cachelines * 64
            page_size = os.sysconf("SC_PAGE_SIZE")
            if (
                magic != _CXLMEMSIM_MAGIC
                or version != _CXLMEMSIM_VERSION
                or total_size != size
                or data_offset < _CXLMEMSIM_HEADER.size
                or data_offset % page_size
                or capacity <= 0
                or data_offset > size
                or capacity > size - data_offset
            ):
                raise RuntimeError("CXLMemSim shared region header is incompatible")
            if (
                self._expected_capacity is not None
                and capacity != self._expected_capacity
            ):
                raise RuntimeError(
                    "CXLMemSim shared region capacity does not match config"
                )
            self._fd = fd
            self._handle = RegionHandle(
                region_id=self._region_id,
                shm_name=self._shm_name,
                capacity=capacity,
                alignment=page_size,
                capabilities=frozenset(
                    {"cuda_host_register_v1", "cxlmemsim_region_v1"}
                ),
                data_offset=data_offset,
            )
            return self._handle
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        """Close the descriptor without unlinking CXLMemSim storage."""
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._handle = None
