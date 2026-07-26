# SPDX-License-Identifier: Apache-2.0
"""Tests for the CXLMemSim bulk client wrapper."""

# Standard
from collections.abc import Callable
import ctypes
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.cxl_memsim_client import (
    BulkClientStats,
    BulkTransferResult,
    CxlMemSimClient,
    CxlMemSimError,
)


class _FakeFunction:
    def __init__(self, function: Callable[..., object]) -> None:
        self.function = function
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *args: object) -> object:
        return self.function(*args)


class _FakeBulkLibrary:
    def __init__(self) -> None:
        self.closed = False
        self.write_error = 0
        self.result_bytes: int | None = None
        self.write_entered = threading.Event()
        self.write_release = threading.Event()
        self.block_writes = False

        self.cxl_bulk_client_open = _FakeFunction(self._open)
        self.cxl_bulk_client_close = _FakeFunction(self._close)
        self.cxl_bulk_client_capacity = _FakeFunction(self._capacity)
        self.cxl_bulk_read = _FakeFunction(self._read)
        self.cxl_bulk_write = _FakeFunction(self._write)
        self.cxl_bulk_error_string = _FakeFunction(self._error_string)

    @staticmethod
    def _int_value(value: object) -> int:
        return int(getattr(value, "value", value))

    def _open(self, control_name: object, timeout_ms: object, out: object) -> int:
        assert control_name == b"/test_bulk"
        assert self._int_value(timeout_ms) == 250
        handle = ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))
        handle[0] = ctypes.c_void_p(0x1234)
        return 0

    def _close(self, handle: object) -> None:
        assert self._int_value(handle) == 0x1234
        self.closed = True

    def _capacity(self, handle: object) -> int:
        assert self._int_value(handle) == 0x1234
        return 4096

    def _read(
        self,
        handle: object,
        offset: object,
        pointer: object,
        size: object,
        result: object,
    ) -> int:
        assert self._int_value(handle) == 0x1234
        assert self._int_value(offset) == 128
        assert self._int_value(pointer) == 0x2000
        self._set_result(result, self._int_value(size))
        return 0

    def _write(
        self,
        handle: object,
        offset: object,
        pointer: object,
        size: object,
        result: object,
    ) -> int:
        assert self._int_value(handle) == 0x1234
        assert self._int_value(offset) == 64
        assert self._int_value(pointer) == 0x1000
        self.write_entered.set()
        if self.block_writes:
            assert self.write_release.wait(timeout=5)
        if self.write_error:
            return self.write_error
        result_size = self.result_bytes
        self._set_result(
            result,
            self._int_value(size) if result_size is None else result_size,
        )
        return 0

    @staticmethod
    def _set_result(result_pointer: object, size: int) -> None:
        result = ctypes.cast(result_pointer, ctypes.POINTER(ctypes.c_uint64))
        result[0] = size
        result[1] = 11
        result[2] = 22
        result[3] = 33
        result[4] = 2

    @staticmethod
    def _error_string(error_code: object) -> bytes:
        code = int(getattr(error_code, "value", error_code))
        return {6: b"bulk request timed out"}.get(code, b"unknown error")


def _make_client(library: _FakeBulkLibrary) -> CxlMemSimClient:
    return CxlMemSimClient(
        library_path="/fake/libcxlmemsim_client.so",
        control_name="/test_bulk",
        timeout_ms=250,
        library_loader=lambda _: library,
    )


def test_bulk_client_translates_native_results_and_accumulates_stats() -> None:
    library = _FakeBulkLibrary()
    client = _make_client(library)
    try:
        assert client.capacity == 4096
        assert client.write_from(offset=64, src_ptr=0x1000, size=128) == (
            BulkTransferResult(
                bytes=128,
                host_copy_ns=11,
                model_latency_ns=22,
                serialization_ns=33,
                cacheline_count=2,
            )
        )
        assert client.read_into(offset=128, dst_ptr=0x2000, size=128) == (
            BulkTransferResult(
                bytes=128,
                host_copy_ns=11,
                model_latency_ns=22,
                serialization_ns=33,
                cacheline_count=2,
            )
        )
        assert client.snapshot_stats() == BulkClientStats(
            read_requests=1,
            write_requests=1,
            read_bytes=128,
            write_bytes=128,
            read_host_copy_ns=11,
            write_host_copy_ns=11,
            read_model_latency_ns=22,
            write_model_latency_ns=22,
            read_serialization_ns=33,
            write_serialization_ns=33,
            read_cachelines=2,
            write_cachelines=2,
        )
    finally:
        client.close()


def test_bulk_client_raises_native_error_with_code_and_message() -> None:
    library = _FakeBulkLibrary()
    library.write_error = 6
    client = _make_client(library)
    try:
        with pytest.raises(CxlMemSimError, match="bulk request timed out") as exc_info:
            client.write_from(offset=64, src_ptr=0x1000, size=128)
        assert exc_info.value.error_code == 6
        assert client.snapshot_stats() == BulkClientStats()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("offset", "pointer", "size"),
    [(-1, 0x1000, 64), (0, 0, 64), (0, 0x1000, 0)],
)
def test_bulk_client_rejects_invalid_transfer_arguments(
    offset: int,
    pointer: int,
    size: int,
) -> None:
    client = _make_client(_FakeBulkLibrary())
    try:
        with pytest.raises(ValueError):
            client.write_from(offset=offset, src_ptr=pointer, size=size)
    finally:
        client.close()


def test_bulk_client_rejects_short_success_as_protocol_error() -> None:
    library = _FakeBulkLibrary()
    library.result_bytes = 64
    client = _make_client(library)
    try:
        with pytest.raises(CxlMemSimError, match="returned 64 bytes for 128"):
            client.write_from(offset=64, src_ptr=0x1000, size=128)
        assert client.snapshot_stats() == BulkClientStats()
    finally:
        client.close()


def test_bulk_client_close_waits_for_active_transfer_and_is_idempotent() -> None:
    library = _FakeBulkLibrary()
    library.block_writes = True
    client = _make_client(library)

    transfer = threading.Thread(
        target=client.write_from,
        kwargs={"offset": 64, "src_ptr": 0x1000, "size": 128},
    )
    transfer.start()
    assert library.write_entered.wait(timeout=5)

    close = threading.Thread(target=client.close)
    close.start()
    time.sleep(0.05)
    assert close.is_alive()
    assert library.closed is False

    library.write_release.set()
    transfer.join(timeout=5)
    close.join(timeout=5)
    assert transfer.is_alive() is False
    assert close.is_alive() is False
    assert library.closed is True

    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.read_into(offset=128, dst_ptr=0x2000, size=128)
