# SPDX-License-Identifier: Apache-2.0
"""Tests for the CXLMemSim MP L2 adapter."""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import cast
import ctypes
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l2_adapters.cxl_memsim_client import (
    BulkClientStats,
    BulkTransferResult,
)
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.cxl_memsim_l2_adapter import (
    CxlMemSimL2Adapter,
    CxlMemSimL2AdapterConfig,
)
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


class _FakeClient:
    def __init__(self, *, capacity: int = 16 * 1024) -> None:
        self.capacity = capacity
        self.closed = False
        self.buffer = bytearray(capacity)
        self.write_calls: list[tuple[int, int]] = []
        self.read_calls: list[tuple[int, int]] = []
        self.fail_next_write = False
        self.fail_next_read = False
        self.write_started = threading.Event()
        self.write_release = threading.Event()
        self.write_release.set()
        self.read_started = threading.Event()
        self.read_release = threading.Event()
        self.read_release.set()
        self._stats_lock = threading.Lock()
        self._stats = {field: 0 for field in BulkClientStats.__dataclass_fields__}

    def block_next_write(self) -> None:
        self.write_started.clear()
        self.write_release.clear()

    def release_write(self) -> None:
        self.write_release.set()

    def block_next_read(self) -> None:
        self.read_started.clear()
        self.read_release.clear()

    def release_read(self) -> None:
        self.read_release.set()

    def write_from(
        self,
        offset: int,
        src_ptr: int,
        size: int,
    ) -> BulkTransferResult:
        self.write_started.set()
        assert self.write_release.wait(timeout=5)
        self.write_calls.append((offset, size))
        if self.fail_next_write:
            self.fail_next_write = False
            raise RuntimeError("injected write failure")
        self.buffer[offset : offset + size] = ctypes.string_at(src_ptr, size)
        result = BulkTransferResult(
            bytes=size,
            host_copy_ns=10,
            model_latency_ns=20,
            serialization_ns=30,
            cacheline_count=(size + 63) // 64,
        )
        with self._stats_lock:
            self._stats["write_requests"] += 1
            self._stats["write_bytes"] += size
            self._stats["write_host_copy_ns"] += result.host_copy_ns
            self._stats["write_model_latency_ns"] += result.model_latency_ns
            self._stats["write_serialization_ns"] += result.serialization_ns
            self._stats["write_cachelines"] += result.cacheline_count
        return result

    def read_into(
        self,
        offset: int,
        dst_ptr: int,
        size: int,
    ) -> BulkTransferResult:
        self.read_started.set()
        assert self.read_release.wait(timeout=5)
        self.read_calls.append((offset, size))
        if self.fail_next_read:
            self.fail_next_read = False
            raise RuntimeError("injected read failure")
        source = (ctypes.c_ubyte * size).from_buffer(self.buffer, offset)
        ctypes.memmove(dst_ptr, ctypes.addressof(source), size)
        result = BulkTransferResult(
            bytes=size,
            host_copy_ns=11,
            model_latency_ns=21,
            serialization_ns=31,
            cacheline_count=(size + 63) // 64,
        )
        with self._stats_lock:
            self._stats["read_requests"] += 1
            self._stats["read_bytes"] += size
            self._stats["read_host_copy_ns"] += result.host_copy_ns
            self._stats["read_model_latency_ns"] += result.model_latency_ns
            self._stats["read_serialization_ns"] += result.serialization_ns
            self._stats["read_cachelines"] += result.cacheline_count
        return result

    def snapshot_stats(self) -> BulkClientStats:
        with self._stats_lock:
            return BulkClientStats(**self._stats)

    def close(self) -> None:
        self.closed = True


class _ClientFactory:
    def __init__(self, *, capacity: int = 16 * 1024) -> None:
        self.capacity = capacity
        self.calls: list[dict[str, object]] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, **kwargs: object) -> _FakeClient:
        self.calls.append(kwargs)
        client = _FakeClient(capacity=self.capacity)
        self.clients.append(client)
        return client


class _MemoryObjWithRawTensor:
    def __init__(self, delegate: MemoryObj, raw_tensor: torch.Tensor) -> None:
        self._delegate = delegate
        self.raw_tensor = raw_tensor

    @property
    def data_ptr(self) -> int:
        return self._delegate.data_ptr

    @property
    def metadata(self):
        return self._delegate.metadata

    def get_size(self) -> int:
        return self._delegate.get_size()

    def get_physical_size(self) -> int:
        return self._delegate.get_physical_size()

    def release(self) -> None:
        self._delegate.ref_count_down()


class _ExplodingMemoryObj:
    def __init__(self, delegate: MemoryObj) -> None:
        self._delegate = delegate

    @property
    def raw_tensor(self) -> torch.Tensor:
        raise RuntimeError("injected buffer inspection failure")

    def release(self) -> None:
        self._delegate.ref_count_down()


class _RecordingListener(L2AdapterListener):
    def __init__(self) -> None:
        self.stored: list[tuple[list[ObjectKey], list[int]]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(
        self,
        keys: list[ObjectKey],
        sizes: list[int],
    ) -> None:
        self.stored.append((list(keys), list(sizes)))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.append(list(keys))


def _config(**overrides: object) -> CxlMemSimL2AdapterConfig:
    values: dict[str, object] = {
        "client_library": "/tmp/libcxlmemsim_client.so",
        "slot_bytes": 4096,
        "num_store_workers": 1,
        "num_lookup_workers": 1,
        "num_load_workers": 1,
    }
    values.update(overrides)
    return CxlMemSimL2AdapterConfig.from_dict(values)


def _make_adapter(
    *,
    factory: _ClientFactory | None = None,
    **config_overrides: object,
) -> tuple[CxlMemSimL2Adapter, _ClientFactory]:
    client_factory = factory or _ClientFactory()
    adapter = CxlMemSimL2Adapter(
        _config(**config_overrides),
        client_factory=client_factory,
    )
    return adapter, client_factory


def _key(index: int, *, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(index),
        model_name="cxl-memsim-test",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _memory_obj(
    fill_value: float,
    *,
    elements: int = 2048,
) -> MemoryObj:
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate(
        [torch.Size([elements])],
        [torch.bfloat16],
        fmt=MemoryFormat.KV_2LTD,
    )
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(fill_value)
    return obj


def _wait_for_event(fd: int, timeout: float = 5.0) -> None:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(int(timeout * 1000))
    consume_fd(fd)


def _wait_store(
    adapter: CxlMemSimL2Adapter,
    task_id: int,
    *,
    timeout: float = 5.0,
) -> L2StoreResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _wait_for_event(adapter.get_store_event_fd(), deadline - time.monotonic())
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
    pytest.fail(f"store task {task_id} did not complete")


def _lookup_once(adapter: CxlMemSimL2Adapter, key: ObjectKey) -> bool:
    task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
    _wait_for_event(adapter.get_lookup_and_lock_event_fd())
    result = adapter.query_lookup_and_lock_result(task_id)
    assert result is not None
    return result.test(0)


def _lookup_many(
    adapter: CxlMemSimL2Adapter,
    keys: list[ObjectKey],
) -> list[bool]:
    task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    _wait_for_event(adapter.get_lookup_and_lock_event_fd())
    result = adapter.query_lookup_and_lock_result(task_id)
    assert result is not None
    return [result.test(index) for index in range(len(keys))]


def _wait_load(
    adapter: CxlMemSimL2Adapter,
    task_id: int,
    *,
    size: int = 1,
    timeout: float = 5.0,
) -> list[bool]:
    _wait_for_event(adapter.get_load_event_fd(), timeout)
    result = adapter.query_load_result(task_id)
    assert result is not None
    return [result.test(index) for index in range(size)]


def test_cxl_memsim_config_parses_and_registers_adapter_type() -> None:
    config = CxlMemSimL2AdapterConfig.from_dict(
        {
            "type": "cxl_memsim",
            "client_library": " /opt/cxl/libcxlmemsim_client.so ",
            "slot_bytes": 8192,
            "control_name": " /lmcache_test ",
            "offset_bytes": 4096,
            "capacity_bytes": 32768,
            "timeout_ms": 7000,
            "num_store_workers": 2,
            "num_lookup_workers": 3,
            "num_load_workers": 4,
        }
    )

    assert config.client_library == "/opt/cxl/libcxlmemsim_client.so"
    assert config.control_name == "/lmcache_test"
    assert config.slot_bytes == 8192
    assert config.offset_bytes == 4096
    assert config.capacity_bytes == 32768
    assert config.timeout_ms == 7000
    assert config.num_store_workers == 2
    assert config.num_lookup_workers == 3
    assert config.num_load_workers == 4
    assert "cxl_memsim" in get_registered_l2_adapter_types()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("client_library", "", "client_library"),
        ("control_name", "", "control_name"),
        ("slot_bytes", 0, "slot_bytes"),
        ("slot_bytes", True, "slot_bytes"),
        ("offset_bytes", -1, "offset_bytes"),
        ("offset_bytes", True, "offset_bytes"),
        ("capacity_bytes", 0, "capacity_bytes"),
        ("capacity_bytes", True, "capacity_bytes"),
        ("timeout_ms", 0, "timeout_ms"),
        ("num_store_workers", 0, "num_store_workers"),
        ("num_lookup_workers", False, "num_lookup_workers"),
        ("num_load_workers", -1, "num_load_workers"),
    ],
)
def test_cxl_memsim_config_rejects_invalid_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    values: dict[str, object] = {
        "client_library": "/tmp/libcxlmemsim_client.so",
        "slot_bytes": 4096,
    }
    values[field] = value

    with pytest.raises(ValueError, match=match):
        CxlMemSimL2AdapterConfig.from_dict(values)


def test_cxl_memsim_config_requires_fields() -> None:
    with pytest.raises(ValueError, match="client_library"):
        CxlMemSimL2AdapterConfig.from_dict({"slot_bytes": 4096})
    with pytest.raises(ValueError, match="slot_bytes"):
        CxlMemSimL2AdapterConfig.from_dict(
            {"client_library": "/tmp/libcxlmemsim_client.so"}
        )


def test_cxl_memsim_adapter_validates_arena_and_opens_expected_client() -> None:
    adapter, factory = _make_adapter(offset_bytes=2048, capacity_bytes=12288)
    try:
        assert factory.calls == [
            {
                "library_path": "/tmp/libcxlmemsim_client.so",
                "control_name": "/cxlmemsim_bulk",
                "timeout_ms": 5000,
            }
        ]
        status = adapter.report_status()
        assert status["capacity_bytes"] == 12288
        assert status["max_slots"] == 3
        assert status["slot_bytes"] == 4096
        assert status["offset_bytes"] == 2048
    finally:
        adapter.close()
    assert factory.clients[0].closed


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"offset_bytes": 16 * 1024 + 1},
        {"offset_bytes": 8192, "capacity_bytes": 8193},
        {"capacity_bytes": 2048},
    ],
)
def test_cxl_memsim_adapter_rejects_out_of_bounds_or_empty_arena(
    config_overrides: dict[str, object],
) -> None:
    factory = _ClientFactory()

    with pytest.raises(ValueError, match="arena|capacity|offset|slot"):
        CxlMemSimL2Adapter(
            _config(**config_overrides),
            client_factory=factory,
        )

    assert factory.clients[0].closed


def test_cxl_memsim_adapter_has_distinct_eventfds_and_idempotent_close() -> None:
    adapter, factory = _make_adapter()
    event_fds = {
        adapter.get_store_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
        adapter.get_load_event_fd(),
    }
    assert len(event_fds) == 3

    adapter.close()
    adapter.close()

    assert factory.clients[0].closed


def test_cxl_memsim_adapter_rejects_tasks_after_close() -> None:
    adapter, _ = _make_adapter()
    adapter.close()

    with pytest.raises(RuntimeError, match="closing|closed"):
        adapter.submit_lookup_and_lock_task([], _EMPTY_LAYOUT)


def test_client_library_path_need_not_exist_until_adapter_open(tmp_path: Path) -> None:
    path = tmp_path / "missing-client.so"
    config = _config(client_library=str(path))
    assert config.client_library == str(path)


def test_client_factory_protocol_is_runtime_injectable() -> None:
    factory: Callable[..., _FakeClient] = _ClientFactory()
    adapter = CxlMemSimL2Adapter(_config(), client_factory=factory)
    adapter.close()


def test_store_transfers_payload_signals_and_drains_completion_once() -> None:
    adapter, factory = _make_adapter(offset_bytes=1024)
    source = _memory_obj(7, elements=1024)
    key = _key(1, cache_salt="tenant-a")
    try:
        task_id = adapter.submit_store_task([key], [source])
        result = _wait_store(adapter, task_id)

        assert result.is_successful()
        assert result.bytes_transferred() == 2048
        assert adapter.pop_completed_store_tasks() == {}
        assert factory.clients[0].write_calls == [(1024, 2048)]
        assert factory.clients[0].buffer[1024:3072] == ctypes.string_at(
            source.data_ptr, 2048
        )
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 4096
        assert usage.total_capacity_bytes == 12288
        assert dict(usage.bytes_by_cache_salt) == {"tenant-a": 4096}
    finally:
        source.ref_count_down()
        adapter.close()


def test_store_capacity_exhaustion_does_not_call_native_client() -> None:
    adapter, factory = _make_adapter(capacity_bytes=4096)
    first = _memory_obj(1)
    second = _memory_obj(2)
    try:
        first_result = _wait_store(
            adapter,
            adapter.submit_store_task([_key(1)], [first]),
        )
        second_result = _wait_store(
            adapter,
            adapter.submit_store_task([_key(2)], [second]),
        )

        assert first_result.is_successful()
        assert not second_result.is_successful()
        assert factory.clients[0].write_calls == [(0, 4096)]
        assert adapter.get_usage().total_bytes_used == 4096
    finally:
        first.ref_count_down()
        second.ref_count_down()
        adapter.close()


def test_failed_store_rolls_back_slot_for_next_key() -> None:
    adapter, factory = _make_adapter(capacity_bytes=4096)
    first = _memory_obj(1)
    second = _memory_obj(2)
    factory.clients[0].fail_next_write = True
    try:
        failed = _wait_store(
            adapter,
            adapter.submit_store_task([_key(1)], [first]),
        )
        succeeded = _wait_store(
            adapter,
            adapter.submit_store_task([_key(2)], [second]),
        )

        assert not failed.is_successful()
        assert succeeded.is_successful()
        assert factory.clients[0].write_calls == [(0, 4096), (0, 4096)]
        assert adapter.get_usage().total_bytes_used == 4096
    finally:
        first.ref_count_down()
        second.ref_count_down()
        adapter.close()


def test_buffer_inspection_exception_rolls_back_slot_for_next_key() -> None:
    adapter, _ = _make_adapter(capacity_bytes=4096)
    exploding = cast(MemoryObj, _ExplodingMemoryObj(_memory_obj(1)))
    replacement = _memory_obj(2)
    try:
        failed = _wait_store(
            adapter,
            adapter.submit_store_task([_key(1)], [exploding]),
        )
        succeeded = _wait_store(
            adapter,
            adapter.submit_store_task([_key(2)], [replacement]),
        )

        assert not failed.is_successful()
        assert succeeded.is_successful()
    finally:
        cast(_ExplodingMemoryObj, exploding).release()
        replacement.ref_count_down()
        adapter.close()


def test_store_rejects_noncontiguous_host_buffer_before_native_call() -> None:
    adapter, factory = _make_adapter()
    invalid = _memory_obj(3)
    assert invalid.tensor is not None
    invalid_tensor = torch.empty((16, 16), dtype=torch.bfloat16).t()
    invalid = cast(MemoryObj, _MemoryObjWithRawTensor(invalid, invalid_tensor))
    try:
        result = _wait_store(
            adapter,
            adapter.submit_store_task([_key(1)], [invalid]),
        )

        assert not result.is_successful()
        assert factory.clients[0].write_calls == []
        assert adapter.get_usage().total_bytes_used == 0
    finally:
        cast(_MemoryObjWithRawTensor, invalid).release()
        adapter.close()


def test_store_key_is_hidden_until_full_write_commits() -> None:
    adapter, factory = _make_adapter(num_store_workers=2)
    client = factory.clients[0]
    client.block_next_write()
    source = _memory_obj(5)
    key = _key(1)
    try:
        store_task = adapter.submit_store_task([key], [source])
        assert client.write_started.wait(timeout=5)

        assert not _lookup_once(adapter, key)

        client.release_write()
        assert _wait_store(adapter, store_task).is_successful()
        assert _lookup_once(adapter, key)
        adapter.submit_unlock([key])
    finally:
        client.release_write()
        source.ref_count_down()
        adapter.close()


def test_concurrent_duplicate_store_waits_for_first_commit() -> None:
    adapter, factory = _make_adapter(num_store_workers=2)
    client = factory.clients[0]
    client.block_next_write()
    source = _memory_obj(6)
    duplicate = _memory_obj(9)
    key = _key(1)
    try:
        first_task = adapter.submit_store_task([key], [source])
        assert client.write_started.wait(timeout=5)
        duplicate_task = adapter.submit_store_task([key], [duplicate])
        time.sleep(0.05)
        assert client.write_calls == []

        client.release_write()
        completed: dict[int, L2StoreResult] = {}
        deadline = time.monotonic() + 5
        while len(completed) < 2 and time.monotonic() < deadline:
            _wait_for_event(
                adapter.get_store_event_fd(),
                deadline - time.monotonic(),
            )
            completed.update(adapter.pop_completed_store_tasks())

        assert completed[first_task].is_successful()
        assert completed[first_task].bytes_transferred() == 4096
        assert completed[duplicate_task].is_successful()
        assert completed[duplicate_task].bytes_transferred() == 0
        assert client.write_calls == [(0, 4096)]
        assert adapter.get_usage().total_bytes_used == 4096
    finally:
        client.release_write()
        source.ref_count_down()
        duplicate.ref_count_down()
        adapter.close()


def test_lookup_load_and_delete_round_trip_with_metadata_and_listeners() -> None:
    adapter, factory = _make_adapter(offset_bytes=512)
    listener = _RecordingListener()
    adapter.register_listener(listener)
    source = _memory_obj(7)
    target = _memory_obj(0)
    key = _key(1, cache_salt="tenant-a")
    source.metadata.cached_positions = torch.tensor([3, 5, 8])
    try:
        assert _wait_store(
            adapter,
            adapter.submit_store_task([key], [source]),
        ).is_successful()
        source.metadata.cached_positions[0] = 99

        assert _lookup_many(adapter, [key, _key(2)]) == [True, False]
        load_task = adapter.submit_load_task([key, _key(2)], [target, target])
        assert _wait_load(adapter, load_task, size=2) == [True, False]
        assert adapter.query_load_result(load_task) is None
        assert target.tensor is not None
        assert source.tensor is not None
        assert torch.equal(target.tensor, source.tensor)
        assert target.metadata.cached_positions is not None
        assert torch.equal(
            target.metadata.cached_positions,
            torch.tensor([3, 5, 8]),
        )
        assert listener.stored == [([key], [4096])]
        assert listener.accessed == [[key]]

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _lookup_once(adapter, key)
        assert listener.deleted == [[key]]
        assert adapter.get_usage().total_bytes_used == 0
        assert dict(adapter.get_usage().bytes_by_cache_salt) == {}

        status = adapter.report_status()
        assert status["transport"]["write_requests"] == 1
        assert status["transport"]["read_requests"] == 1
        assert status["transport"]["write_bytes"] == 4096
        assert status["transport"]["read_bytes"] == 4096
        assert factory.clients[0].write_calls == [(512, 4096)]
        assert factory.clients[0].read_calls == [(512, 4096)]
    finally:
        source.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_lookup_lock_reference_count_prevents_delete_until_fully_unlocked() -> None:
    adapter, _ = _make_adapter(capacity_bytes=4096)
    source = _memory_obj(1)
    key = _key(1)
    try:
        assert _wait_store(
            adapter,
            adapter.submit_store_task([key], [source]),
        ).is_successful()
        assert _lookup_once(adapter, key)
        assert _lookup_once(adapter, key)

        adapter.delete([key])
        adapter.submit_unlock([key])
        adapter.delete([key])
        assert _lookup_once(adapter, key)
        adapter.submit_unlock([key])

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _lookup_once(adapter, key)
    finally:
        source.ref_count_down()
        adapter.close()


def test_load_rejects_small_destination_without_native_read() -> None:
    adapter, factory = _make_adapter()
    source = _memory_obj(4)
    target = _memory_obj(0, elements=1024)
    key = _key(1)
    try:
        assert _wait_store(
            adapter,
            adapter.submit_store_task([key], [source]),
        ).is_successful()

        load_task = adapter.submit_load_task([key], [target])
        assert _wait_load(adapter, load_task) == [False]
        assert factory.clients[0].read_calls == []
    finally:
        source.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_failed_load_returns_miss_and_releases_read_borrow() -> None:
    adapter, factory = _make_adapter(capacity_bytes=4096)
    source = _memory_obj(2)
    target = _memory_obj(0)
    key = _key(1)
    try:
        assert _wait_store(
            adapter,
            adapter.submit_store_task([key], [source]),
        ).is_successful()
        factory.clients[0].fail_next_read = True

        assert _wait_load(adapter, adapter.submit_load_task([key], [target])) == [False]
        adapter.delete([key])
        assert adapter.get_usage().total_bytes_used == 0
    finally:
        source.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_delete_during_load_defers_slot_reuse_until_native_read_finishes() -> None:
    adapter, factory = _make_adapter(capacity_bytes=4096)
    client = factory.clients[0]
    source = _memory_obj(3)
    target = _memory_obj(0)
    replacement = _memory_obj(8)
    first_key = _key(1)
    replacement_key = _key(2)
    try:
        assert _wait_store(
            adapter,
            adapter.submit_store_task([first_key], [source]),
        ).is_successful()
        client.block_next_read()
        load_task = adapter.submit_load_task([first_key], [target])
        assert client.read_started.wait(timeout=5)

        adapter.delete([first_key])
        during_read = _wait_store(
            adapter,
            adapter.submit_store_task([replacement_key], [replacement]),
        )
        assert not during_read.is_successful()
        assert adapter.report_status()["pending_free_slot_count"] == 1

        client.release_read()
        assert _wait_load(adapter, load_task) == [True]
        assert target.tensor is not None
        assert source.tensor is not None
        assert torch.equal(target.tensor, source.tensor)

        after_read = _wait_store(
            adapter,
            adapter.submit_store_task([replacement_key], [replacement]),
        )
        assert after_read.is_successful()
        assert client.write_calls == [(0, 4096), (0, 4096)]
    finally:
        client.release_read()
        source.ref_count_down()
        target.ref_count_down()
        replacement.ref_count_down()
        adapter.close()


def test_submit_racing_close_has_a_defined_public_outcome() -> None:
    adapter, _factory = _make_adapter()
    source = _memory_obj(1)
    start = threading.Barrier(2)
    submitted: list[object] = []
    submit_errors: list[BaseException] = []

    def submit() -> None:
        start.wait()
        try:
            submitted.append(adapter.submit_store_task([_key(1)], [source]))
        except BaseException as exc:
            submit_errors.append(exc)

    def close() -> None:
        start.wait()
        adapter.close()

    submit_thread = threading.Thread(target=submit)
    close_thread = threading.Thread(target=close)
    try:
        submit_thread.start()
        close_thread.start()
        submit_thread.join(timeout=5)
        close_thread.join(timeout=5)

        assert not submit_thread.is_alive()
        assert not close_thread.is_alive()
        assert bool(submitted) != bool(submit_errors)
        assert all(isinstance(error, RuntimeError) for error in submit_errors)
    finally:
        submit_thread.join(timeout=5)
        close_thread.join(timeout=5)
        source.ref_count_down()
        adapter.close()
