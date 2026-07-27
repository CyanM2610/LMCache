# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Callable
import hashlib
import threading
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.data_plane_adapter import VLLMTransferRequest
from lmcache.v1.multiprocess.cxl.region_provider import (
    REGION_HEADER_SIZE,
    pack_region_header,
)
from lmcache.v1.multiprocess.modules.cxl_shared_tier import CXLSharedTierModule
from lmcache.v1.multiprocess.modules.cxl_shared_tier import (
    InMemoryCXLOperationSink,
)


pytestmark = pytest.mark.no_shared_allocator


class _Registration:
    def __init__(self, shm_name: str, capacity: int) -> None:
        self.shm_name = shm_name
        self.capacity = capacity

    def device_address(self, offset: int, length: int) -> int:
        del length
        return 0x100000 + offset

    def close(self) -> None:
        pass


class _NativeOps:
    class TransferDirection:
        H2D = "h2d"
        D2H = "d2h"

    def __init__(self) -> None:
        self.transfers: list[dict[str, object]] = []

    def CudaRegionRegistration(
        self, shm_name: str, capacity: int, data_offset: int = REGION_HEADER_SIZE
    ) -> _Registration:
        del data_offset
        return _Registration(shm_name, capacity)

    def cxl_region_block_kv_transfer(
        self,
        paged_pointers: str,
        region_pointers: list[int],
        block_ids: list[int],
        device: torch.device,
        shape: SimpleNamespace,
        direction: str,
        slots_per_chunk: int,
        engine_format: str,
        skip_blocks: int,
    ) -> None:
        del region_pointers, device, shape, slots_per_chunk, engine_format, skip_blocks
        self.transfers.append(
            {
                "paged_pointers": paged_pointers,
                "block_ids": list(block_ids),
                "direction": direction,
            }
        )


class _Runtime:
    def execute(self, cache_context: object, operation: Callable[[], None]) -> int:
        del cache_context
        operation()
        return 100


class _ConcurrentRuntime(_Runtime):
    def __init__(self) -> None:
        self.concurrent = False
        self.entered = threading.Event()
        self.both_entered = threading.Event()
        self.release = threading.Event()
        self._entered = 0
        self._lock = threading.Lock()

    def execute(self, cache_context: object, operation: Callable[[], None]) -> int:
        del cache_context
        operation()
        if self.concurrent:
            self.entered.set()
            with self._lock:
                self._entered += 1
                if self._entered == 2:
                    self.both_entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("concurrent transfer was not released")
        return 100


class _ExpiringRuntime(_Runtime):
    def __init__(self, now_ns: list[int]) -> None:
        self.expire = False
        self.now_ns = now_ns

    def execute(self, cache_context: object, operation: Callable[[], None]) -> int:
        del cache_context
        operation()
        if self.expire:
            self.now_ns[0] += 31_000_000_000
        return 100


@dataclass(frozen=True)
class _ObjectGroup:
    kernel_group_indices: tuple[int, ...]


class _GroupsManager:
    object_groups = [_ObjectGroup((0,))]
    num_kernel_groups = 1
    num_object_groups = 1

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 16

    def get_attn_desc(self) -> SimpleNamespace:
        return SimpleNamespace(num_chunks_in_sw=[-1])


class _NativeGroupsManager(_GroupsManager):
    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 32


class _CacheContext:
    device = torch.device("cuda:0")
    stream = object()
    cupy_stream = object()
    lmcache_tokens_per_chunk = 16
    num_layers = 1
    max_batch_size = 1
    kv_layer_groups_manager = _GroupsManager()

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        assert num_tokens == 16
        assert kernel_group_idx == 0
        return (2, 1, 16, 4), torch.float16

    def get_engine_kv_format(self, kernel_group_idx: int) -> str:
        assert kernel_group_idx == 0
        return "NL_X_NB_BS_HS"

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return (num_tokens + 15) // 16

    def stage_block_ids(self, groups: list[list[int]]) -> list[list[int]]:
        return groups

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> str:
        assert kernel_group_idx == 0
        return f"paged:{self.identity}"

    def get_shape_desc(self, kernel_group_idx: int) -> SimpleNamespace:
        assert kernel_group_idx == 0
        return SimpleNamespace(bs=16)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 16


class _NativeCacheContext(_CacheContext):
    lmcache_tokens_per_chunk = 32

    def __init__(self, identity: str, native_ops: object, fill: float) -> None:
        super().__init__(identity)
        self.stream = torch.cuda.Stream(device=self.device)
        self._native_ops = native_ops
        self.kv_layer_groups_manager = _NativeGroupsManager()
        self._num_blocks = 8
        self._head_size = 8
        values = torch.arange(
            2 * self._num_blocks * 16 * self._head_size,
            dtype=torch.float32,
            device=self.device,
        ).to(torch.float16)
        self.paged = (values + fill).view(2, self._num_blocks, 16, 1, self._head_size)
        self._pointers = torch.tensor(
            [self.paged.data_ptr()], dtype=torch.int64, device=self.device
        )

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        assert num_tokens == 32
        assert kernel_group_idx == 0
        return (2, 1, num_tokens, self._head_size), torch.float16

    def get_engine_kv_format(self, kernel_group_idx: int) -> object:
        assert kernel_group_idx == 0
        return self._native_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return (num_tokens + 15) // 16

    def stage_block_ids(self, groups: list[list[int]]) -> list[torch.Tensor]:
        return [
            torch.tensor(group, dtype=torch.int64, device=self.device)
            for group in groups
        ]

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        assert kernel_group_idx == 0
        return self._pointers

    def get_shape_desc(self, kernel_group_idx: int) -> object:
        assert kernel_group_idx == 0
        shape = self._native_ops.PageBufferShapeDesc()
        shape.kv_size = 2
        shape.nl = 1
        shape.nb = self._num_blocks
        shape.bs = 16
        shape.nh = 1
        shape.hs = self._head_size
        shape.element_size = 2
        shape.block_stride_elems = 0
        return shape

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 32


def _key(value: bytes) -> ObjectKey:
    return ObjectKey(value, "Qwen2.5-7B-Instruct", 0)


def _request(
    op_id: str,
    instance_id: int,
    key: ObjectKey,
    block_id: int,
) -> VLLMTransferRequest:
    return VLLMTransferRequest(
        op_id=op_id,
        instance_id=instance_id,
        model_name="Qwen2.5-7B-Instruct",
        token_count=16,
        object_keys=(key,),
        block_ids_by_group=((block_id,),),
        payload_checksum_expected="a" * 64,
    )


def _open_module(
    runtime: _Runtime | None = None,
    operation_sink: InMemoryCXLOperationSink | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> tuple[CXLSharedTierModule, _NativeOps, shared_memory.SharedMemory]:
    capacity = 4096
    name = f"beluga-gate-c-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name,
        create=True,
        size=REGION_HEADER_SIZE + capacity,
    )
    header = pack_region_header(capacity, 64)
    shm.buf[: len(header)] = header
    native = _NativeOps()
    module = CXLSharedTierModule.open(
        CXLSharedTierConfig(
            enabled=True,
            provider="posix_shm",
            shm_name=f"/{name}",
            capacity_bytes=capacity,
            alignment_bytes=64,
            layout_id="packed_kv_v1",
        ),
        native_ops=native,
        runtime=runtime or _Runtime(),
        operation_sink=operation_sink,
        **({} if clock_ns is None else {"clock_ns": clock_ns}),
    )
    return module, native, shm


def test_instance_b_retrieves_a_ready_object_with_only_its_own_hbm_blocks() -> None:
    module, native, shm = _open_module()
    key = _key(b"cross-instance")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        module.register_engine(22, _CacheContext("B"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store-a", 11, key, 3)).status == "ok"

        module.unregister_engine(11)
        first = module.retrieve(_request("retrieve-b", 22, key, 9))

        assert first.status == "ok"
        h2d = [
            transfer for transfer in native.transfers if transfer["direction"] == "h2d"
        ]
        assert [transfer["block_ids"] for transfer in h2d] == [[9]]
        assert [transfer["paged_pointers"] for transfer in h2d] == ["paged:B"]

        module.unregister_engine(22)
        module.register_engine(22, _CacheContext("B-restarted"), "Qwen2.5-7B-Instruct")
        restarted = module.retrieve(_request("retrieve-b-restarted", 22, key, 12))

        assert restarted.status == "ok"
        assert native.transfers[-1]["block_ids"] == [12]
        assert native.transfers[-1]["paged_pointers"] == "paged:B-restarted"
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_ready_prefix_stops_at_first_missing_chunk() -> None:
    module, _, shm = _open_module()
    first = _key(b"first")
    missing = _key(b"missing")
    later = _key(b"later")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store-first", 11, first, 1)).status == "ok"
        assert module.store(_request("store-later", 11, later, 2)).status == "ok"

        assert module.count_ready_prefix((first, missing, later), 1) == 1
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_duplicate_store_and_retrieve_miss_emit_terminal_events() -> None:
    sink = InMemoryCXLOperationSink()
    module, native, shm = _open_module(operation_sink=sink)
    ready_key = _key(b"already-ready")
    missing_key = _key(b"not-present")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("initial", 11, ready_key, 1)).status == "ok"
        transfer_count = len(native.transfers)

        duplicate = module.store(_request("duplicate", 11, ready_key, 2))
        missing = module.retrieve(_request("missing", 11, missing_key, 3))

        assert duplicate.status == "ok"
        assert missing.status == "miss"
        assert len(native.transfers) == transfer_count
        terminal = {
            event.op_id: event
            for event in sink.snapshot()
            if event.terminal and event.op_id.startswith(("duplicate", "missing"))
        }
        assert terminal["duplicate:chunk:0"].state == "ready"
        assert terminal["duplicate:chunk:0"].cuda_elapsed_ns == 0
        assert terminal["missing:chunk:0"].state == "miss"
        assert terminal["missing:chunk:0"].extent is None
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_two_readers_hold_leases_concurrently_before_eviction() -> None:
    runtime = _ConcurrentRuntime()
    module, _, shm = _open_module(runtime)
    key = _key(b"concurrent-readers")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        module.register_engine(22, _CacheContext("B"), "Qwen2.5-7B-Instruct")
        module.register_engine(33, _CacheContext("C"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store", 11, key, 3)).status == "ok"
        runtime.concurrent = True

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(module.retrieve, _request("retrieve-b", 22, key, 9))
            second = pool.submit(module.retrieve, _request("retrieve-c", 33, key, 12))
            readers_were_concurrent = runtime.both_entered.wait(timeout=1)
            module.evict(key)
            blocked_store = module.store(_request("store-while-read", 11, key, 4))
            runtime.release.set()
            assert first.result(timeout=5).status == "ok"
            assert second.result(timeout=5).status == "ok"

        assert readers_were_concurrent is True
        assert blocked_store.status == "error"
        replacement = module.store(_request("store-after-read", 11, key, 5))
        assert replacement.status == "ok"
        assert module.lookup_ready(key).generation == 2
    finally:
        runtime.release.set()
        module.close()
        shm.close()
        shm.unlink()


def test_cancelled_retrieve_drains_before_unregister_and_restart() -> None:
    runtime = _ConcurrentRuntime()
    sink = InMemoryCXLOperationSink(capacity=16)
    module, _, shm = _open_module(runtime, operation_sink=sink)
    key = _key(b"orphan-cleanup")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        module.register_engine(22, _CacheContext("B"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store", 11, key, 3)).status == "ok"
        runtime.concurrent = True

        with ThreadPoolExecutor(max_workers=2) as pool:
            retrieve = pool.submit(
                module.retrieve, _request("orphaned-retrieve", 22, key, 9)
            )
            assert runtime.entered.wait(timeout=1)
            assert module.cancel("orphaned-retrieve") is True
            unregister = pool.submit(module.unregister_engine, 22)
            assert unregister.done() is False
            runtime.release.set()

            assert retrieve.result(timeout=5).status == "cancelled"
            unregister.result(timeout=5)

        terminal = [
            event
            for event in sink.snapshot()
            if event.op_id.startswith("orphaned-retrieve") and event.terminal
        ]
        assert [event.state for event in terminal] == ["cancelled"]
        assert terminal[0].lease_id is not None
        runtime.concurrent = False
        module.register_engine(22, _CacheContext("B-restarted"), "Qwen2.5-7B-Instruct")
        restarted = module.retrieve(_request("restart-retrieve", 22, key, 12))
        assert restarted.status == "ok"
    finally:
        runtime.release.set()
        module.close()
        shm.close()
        shm.unlink()


def test_cancelled_store_never_publishes_partial_data() -> None:
    runtime = _ConcurrentRuntime()
    sink = InMemoryCXLOperationSink(capacity=16)
    module, _, shm = _open_module(runtime, operation_sink=sink)
    key = _key(b"cancelled-store")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        runtime.concurrent = True

        with ThreadPoolExecutor(max_workers=2) as pool:
            store = pool.submit(
                module.store, _request("cancelled-store-op", 11, key, 3)
            )
            assert runtime.entered.wait(timeout=1)
            cancellation = pool.submit(module.cancel, "cancelled-store-op")
            try:
                assert cancellation.result(timeout=1) is True
            finally:
                runtime.release.set()
            assert store.result(timeout=5).status == "cancelled"

        assert module.lookup_ready(key) is None
        cancelled = [event for event in sink.snapshot() if event.state == "cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0].terminal is True
        assert cancelled[0].extent is not None
        runtime.concurrent = False
        retry = module.store(_request("retry-store", 11, key, 4))
        assert retry.status == "ok"
    finally:
        runtime.release.set()
        module.close()
        shm.close()
        shm.unlink()


def test_cross_instance_telemetry_proves_publish_before_lease() -> None:
    sink = InMemoryCXLOperationSink(capacity=16)
    module, _, shm = _open_module(operation_sink=sink)
    key = _key(b"telemetry")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        module.register_engine(22, _CacheContext("B"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store-a", 11, key, 3)).status == "ok"
        assert module.retrieve(_request("retrieve-b", 22, key, 9)).status == "ok"

        events = sink.snapshot()
        ready = next(event for event in events if event.state == "ready")
        leased = next(event for event in events if event.state == "lease_acquired")
        retrieved = next(event for event in events if event.state == "ok")

        assert ready.timestamp_ns < leased.timestamp_ns <= retrieved.timestamp_ns
        assert ready.instance_id == 11
        assert leased.instance_id == 22
        assert ready.object_key == key.to_encoded_object_key()
        assert ready.extent is not None
        assert ready.generation == 1
        assert ready.layout_fingerprint == ready.extent.layout_fingerprint
        assert ready.payload_checksum == "a" * 64
        assert ready.cuda_elapsed_ns == 100
        assert leased.lease_id is not None
        assert retrieved.cuda_elapsed_ns == 100
        assert retrieved.path == "cxl_direct"
        serialized = str([event.to_primitive() for event in events])
        assert "block_ids" not in serialized
        assert "pointer" not in serialized
        assert "token" not in serialized
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_retrieve_rejects_a_lease_that_expires_during_transfer() -> None:
    now_ns = [100]
    runtime = _ExpiringRuntime(now_ns)
    sink = InMemoryCXLOperationSink(capacity=16)
    module, _, shm = _open_module(
        runtime,
        operation_sink=sink,
        clock_ns=lambda: now_ns[0],
    )
    key = _key(b"expired-lease")
    try:
        module.register_engine(11, _CacheContext("A"), "Qwen2.5-7B-Instruct")
        module.register_engine(22, _CacheContext("B"), "Qwen2.5-7B-Instruct")
        assert module.store(_request("store", 11, key, 3)).status == "ok"
        runtime.expire = True

        retrieve = module.retrieve(_request("expired-retrieve", 22, key, 9))

        assert retrieve.status == "error"
        terminal = [
            event
            for event in sink.snapshot()
            if event.op_id.startswith("expired-retrieve") and event.terminal
        ]
        assert [event.state for event in terminal] == ["error"]
        module.evict(key)
        assert module.lookup_ready(key) is None
    finally:
        module.close()
        shm.close()
        shm.unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_native_a_to_b_round_trip_survives_source_exit_and_b_restart() -> None:
    import lmcache.c_ops as native_ops

    if not hasattr(native_ops, "CudaRegionRegistration"):
        pytest.skip("CXL region native extension is not built")
    capacity = 8192
    name = f"beluga-gate-c-native-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name,
        create=True,
        size=REGION_HEADER_SIZE + capacity,
    )
    header = pack_region_header(capacity, 64)
    shm.buf[: len(header)] = header
    sink = InMemoryCXLOperationSink(capacity=32)
    module = CXLSharedTierModule.open(
        CXLSharedTierConfig(
            enabled=True,
            provider="posix_shm",
            shm_name=f"/{name}",
            capacity_bytes=capacity,
            alignment_bytes=64,
            layout_id="packed_kv_v1",
        ),
        operation_sink=sink,
    )
    source = _NativeCacheContext("A", native_ops, fill=0)
    destination = _NativeCacheContext("B", native_ops, fill=-20_000)
    key = _key(b"native-cross-instance")
    source_blocks = (3, 1)
    first_destination_blocks = (6, 0)
    expected = source.paged[:, list(source_blocks)].clone()
    expected_checksum = hashlib.sha256(
        expected.cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    store = VLLMTransferRequest(
        op_id="native-store-a",
        instance_id=11,
        model_name="Qwen2.5-7B-Instruct",
        token_count=32,
        object_keys=(key,),
        block_ids_by_group=(source_blocks,),
        payload_checksum_expected=expected_checksum,
    )
    torch.cuda.synchronize()
    try:
        module.register_engine(11, source, "Qwen2.5-7B-Instruct")
        module.register_engine(22, destination, "Qwen2.5-7B-Instruct")
        untouched = destination.paged[:, 4].clone()
        assert module.store(store).status == "ok"
        module.unregister_engine(11)

        first = module.retrieve(
            VLLMTransferRequest(
                op_id="native-retrieve-b",
                instance_id=22,
                model_name=store.model_name,
                token_count=store.token_count,
                object_keys=store.object_keys,
                block_ids_by_group=(first_destination_blocks,),
                payload_checksum_expected=expected_checksum,
            )
        )
        torch.cuda.synchronize()

        assert first.status == "ok"
        assert torch.equal(
            destination.paged[:, list(first_destination_blocks)], expected
        )
        assert torch.equal(destination.paged[:, 4], untouched)

        module.unregister_engine(22)
        restarted = _NativeCacheContext("B-restarted", native_ops, fill=-10_000)
        module.register_engine(22, restarted, "Qwen2.5-7B-Instruct")
        second_destination_blocks = (2, 7)
        second = module.retrieve(
            VLLMTransferRequest(
                op_id="native-retrieve-b-restarted",
                instance_id=22,
                model_name=store.model_name,
                token_count=store.token_count,
                object_keys=store.object_keys,
                block_ids_by_group=(second_destination_blocks,),
                payload_checksum_expected=expected_checksum,
            )
        )
        torch.cuda.synchronize()

        assert second.status == "ok"
        actual = restarted.paged[:, list(second_destination_blocks)]
        assert torch.equal(actual, expected)
        assert (
            hashlib.sha256(actual.cpu().contiguous().numpy().tobytes()).hexdigest()
            == expected_checksum
        )
        ready = next(event for event in sink.snapshot() if event.state == "ready")
        leases = [event for event in sink.snapshot() if event.state == "lease_acquired"]
        assert ready.timestamp_ns < leases[0].timestamp_ns < leases[1].timestamp_ns
    finally:
        module.close()
        shm.close()
        shm.unlink()
