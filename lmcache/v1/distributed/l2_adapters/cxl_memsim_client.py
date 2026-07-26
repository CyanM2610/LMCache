# SPDX-License-Identifier: Apache-2.0
"""Typed wrapper for the CXLMemSim bulk shared-memory client ABI."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
import ctypes
import threading


class _CxlBulkResult(ctypes.Structure):
    _fields_ = [
        ("bytes", ctypes.c_uint64),
        ("host_copy_ns", ctypes.c_uint64),
        ("model_latency_ns", ctypes.c_uint64),
        ("serialization_ns", ctypes.c_uint64),
        ("cacheline_count", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class BulkTransferResult:
    """Timing and accounting returned for one native bulk transfer."""

    bytes: int
    host_copy_ns: int
    model_latency_ns: int
    serialization_ns: int
    cacheline_count: int


@dataclass(frozen=True)
class BulkClientStats:
    """Cumulative successful transfer counters for one client."""

    read_requests: int = 0
    write_requests: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_host_copy_ns: int = 0
    write_host_copy_ns: int = 0
    read_model_latency_ns: int = 0
    write_model_latency_ns: int = 0
    read_serialization_ns: int = 0
    write_serialization_ns: int = 0
    read_cachelines: int = 0
    write_cachelines: int = 0


class CxlMemSimError(RuntimeError):
    """Failure reported by the CXLMemSim bulk client or its Python wrapper."""

    def __init__(self, message: str, error_code: int | None = None) -> None:
        """Initialize a native client error.

        Args:
            message: Human-readable failure description.
            error_code: Optional numeric error returned by the C ABI.
        """
        super().__init__(message)
        self.error_code = error_code


class CxlMemSimClient:
    """Thread-safe owner of one CXLMemSim bulk client handle."""

    def __init__(
        self,
        library_path: str,
        control_name: str,
        timeout_ms: int,
        *,
        library_loader: Callable[[str], Any] = ctypes.CDLL,
    ) -> None:
        """Open the native client and attach to a bulk-shm server.

        Args:
            library_path: Path to ``libcxlmemsim_client.so``.
            control_name: POSIX shared-memory control object name.
            timeout_ms: Open and transfer timeout in milliseconds.
            library_loader: Dynamic-library loader, injectable for embedders.

        Raises:
            ValueError: If a string is empty or ``timeout_ms`` is not positive.
            CxlMemSimError: If the library or server cannot be opened.
        """
        if not library_path:
            raise ValueError("library_path must be a non-empty string")
        if not control_name:
            raise ValueError("control_name must be a non-empty string")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")

        try:
            library = library_loader(library_path)
        except OSError as exc:
            raise CxlMemSimError(
                f"failed to load CXLMemSim client library: {exc}"
            ) from exc

        self._library = library
        self._configure_abi()
        self._condition = threading.Condition()
        self._stats_lock = threading.Lock()
        self._active_transfers = 0
        self._closing = False
        self._closed = False
        self._stats = {field: 0 for field in BulkClientStats.__dataclass_fields__}

        handle = ctypes.c_void_p()
        error_code = int(
            self._library.cxl_bulk_client_open(
                control_name.encode(),
                ctypes.c_uint32(timeout_ms),
                ctypes.byref(handle),
            )
        )
        if error_code != 0:
            raise self._native_error("failed to open CXLMemSim bulk client", error_code)
        if handle.value is None:
            raise CxlMemSimError("CXLMemSim bulk client returned a null handle")

        self._handle: ctypes.c_void_p | None = handle
        self._capacity = int(self._library.cxl_bulk_client_capacity(handle))
        if self._capacity <= 0:
            self._library.cxl_bulk_client_close(handle)
            self._handle = None
            self._closed = True
            raise CxlMemSimError("CXLMemSim bulk client reported zero capacity")

    @property
    def capacity(self) -> int:
        """Return the server-advertised data capacity in bytes."""
        return self._capacity

    def write_from(
        self,
        offset: int,
        src_ptr: int,
        size: int,
    ) -> BulkTransferResult:
        """Write bytes from a host pointer into the simulated CXL arena.

        Args:
            offset: Byte offset in the server data arena.
            src_ptr: Readable host virtual address.
            size: Number of bytes to transfer.

        Returns:
            Native timing and cache-line accounting.

        Raises:
            ValueError: If the pointer or range is invalid.
            RuntimeError: If the client is closing or closed.
            CxlMemSimError: If the native request fails.
        """
        return self._transfer("write", offset, src_ptr, size)

    def read_into(
        self,
        offset: int,
        dst_ptr: int,
        size: int,
    ) -> BulkTransferResult:
        """Read bytes from the simulated CXL arena into a host pointer.

        Args:
            offset: Byte offset in the server data arena.
            dst_ptr: Writable host virtual address.
            size: Number of bytes to transfer.

        Returns:
            Native timing and cache-line accounting.

        Raises:
            ValueError: If the pointer or range is invalid.
            RuntimeError: If the client is closing or closed.
            CxlMemSimError: If the native request fails.
        """
        return self._transfer("read", offset, dst_ptr, size)

    def snapshot_stats(self) -> BulkClientStats:
        """Return an immutable snapshot of successful transfer counters."""
        with self._stats_lock:
            return BulkClientStats(**self._stats)

    def close(self) -> None:
        """Wait for active transfers and release the native client handle."""
        with self._condition:
            if self._closed:
                return
            self._closing = True
            while self._active_transfers > 0:
                self._condition.wait()
            handle = self._handle
            self._handle = None
            self._closed = True

        if handle is not None:
            self._library.cxl_bulk_client_close(handle)

    def __enter__(self) -> CxlMemSimClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _configure_abi(self) -> None:
        self._library.cxl_bulk_client_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.cxl_bulk_client_open.restype = ctypes.c_int
        self._library.cxl_bulk_client_close.argtypes = [ctypes.c_void_p]
        self._library.cxl_bulk_client_close.restype = None
        self._library.cxl_bulk_client_capacity.argtypes = [ctypes.c_void_p]
        self._library.cxl_bulk_client_capacity.restype = ctypes.c_uint64
        transfer_args = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_CxlBulkResult),
        ]
        self._library.cxl_bulk_read.argtypes = transfer_args
        self._library.cxl_bulk_read.restype = ctypes.c_int
        self._library.cxl_bulk_write.argtypes = transfer_args
        self._library.cxl_bulk_write.restype = ctypes.c_int
        self._library.cxl_bulk_error_string.argtypes = [ctypes.c_int]
        self._library.cxl_bulk_error_string.restype = ctypes.c_char_p

    def _transfer(
        self,
        direction: Literal["read", "write"],
        offset: int,
        pointer: int,
        size: int,
    ) -> BulkTransferResult:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if pointer <= 0:
            raise ValueError("pointer must be a positive host address")
        if size <= 0:
            raise ValueError("size must be positive")
        if offset > self._capacity or size > self._capacity - offset:
            raise ValueError("transfer range exceeds the CXLMemSim capacity")

        with self._condition:
            if self._closing or self._closed or self._handle is None:
                raise RuntimeError("CXLMemSim bulk client is closed")
            handle = self._handle
            self._active_transfers += 1

        native_result = _CxlBulkResult()
        try:
            function = (
                self._library.cxl_bulk_read
                if direction == "read"
                else self._library.cxl_bulk_write
            )
            error_code = int(
                function(
                    handle,
                    ctypes.c_uint64(offset),
                    ctypes.c_void_p(pointer),
                    ctypes.c_uint64(size),
                    ctypes.byref(native_result),
                )
            )
            if error_code != 0:
                raise self._native_error(
                    f"CXLMemSim bulk {direction} failed",
                    error_code,
                )
            if int(native_result.bytes) != size:
                raise CxlMemSimError(
                    f"CXLMemSim bulk {direction} returned "
                    f"{int(native_result.bytes)} bytes for {size}"
                )
            result = BulkTransferResult(
                bytes=int(native_result.bytes),
                host_copy_ns=int(native_result.host_copy_ns),
                model_latency_ns=int(native_result.model_latency_ns),
                serialization_ns=int(native_result.serialization_ns),
                cacheline_count=int(native_result.cacheline_count),
            )
            self._record_stats(direction, result)
            return result
        finally:
            with self._condition:
                self._active_transfers -= 1
                if self._active_transfers == 0:
                    self._condition.notify_all()

    def _record_stats(
        self,
        direction: Literal["read", "write"],
        result: BulkTransferResult,
    ) -> None:
        with self._stats_lock:
            self._stats[f"{direction}_requests"] += 1
            self._stats[f"{direction}_bytes"] += result.bytes
            self._stats[f"{direction}_host_copy_ns"] += result.host_copy_ns
            self._stats[f"{direction}_model_latency_ns"] += result.model_latency_ns
            self._stats[f"{direction}_serialization_ns"] += result.serialization_ns
            self._stats[f"{direction}_cachelines"] += result.cacheline_count

    def _native_error(self, prefix: str, error_code: int) -> CxlMemSimError:
        raw_message = self._library.cxl_bulk_error_string(ctypes.c_int(error_code))
        message = (
            raw_message.decode(errors="replace")
            if isinstance(raw_message, bytes)
            else "unknown native error"
        )
        return CxlMemSimError(f"{prefix}: {message}", error_code)
