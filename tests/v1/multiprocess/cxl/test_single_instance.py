# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Callable, Iterator
import struct
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.contracts import CompositeCompletion
from lmcache.v1.multiprocess.cxl.data_plane_adapter import VLLMTransferRequest
from lmcache.v1.multiprocess.cxl.region_provider import (
    REGION_HEADER_SIZE,
    RegionHandle,
    pack_region_header,
)
from lmcache.v1.multiprocess.cxl.model_client import (
    ModelCompletion,
    ModeledAccessRequest,
    RegisteredModelRegion,
)
from lmcache.v1.multiprocess.cxl.policy_protocol import (
    GATE_E_PROTOCOL_VERSION,
    GateERequestEnvelope,
)
from lmcache.v1.multiprocess.cxl.residency import ResidencyState
from lmcache.v1.multiprocess.modules.cxl_shared_tier import (
    CXLSharedTierModule,
    CXLTransferResult,
    InMemoryCXLOperationSink,
)
from lmcache.v1.mp_observability.event import Event


class _FakeRegistration:
    def __init__(self, shm_name: str, capacity: int) -> None:
        self.shm_name = shm_name
        self.capacity = capacity
        self.closed = False

    def device_address(self, offset: int, length: int) -> int:
        assert offset >= 0
        assert length > 0
        return 0x100000 + offset

    def close(self) -> None:
        self.closed = True


class _FakeNativeOps:
    class TransferDirection:
        H2D = "h2d"
        D2H = "d2h"

    def __init__(self) -> None:
        self.registrations: list[_FakeRegistration] = []
        self.transfers: list[tuple[object, ...]] = []
        self.fail_transfer = False

    def CudaRegionRegistration(
        self, shm_name: str, capacity: int, data_offset: int = REGION_HEADER_SIZE
    ) -> _FakeRegistration:
        del data_offset
        registration = _FakeRegistration(shm_name, capacity)
        self.registrations.append(registration)
        return registration

    def cxl_region_block_kv_transfer(self, *args: object) -> None:
        if self.fail_transfer:
            raise RuntimeError("injected CUDA failure")
        self.transfers.append(args)


class _FakeRuntime:
    def execute(self, cache_context: object, operation: Callable[[], None]) -> int:
        del cache_context
        operation()
        return 50


class _FakeModelClient:
    def __init__(self) -> None:
        self.requests: dict[str, ModeledAccessRequest] = {}
        self.data_terminals: list[tuple[str, str, int]] = []
        self.closed = False

    def register_region(self, handle: RegionHandle) -> RegisteredModelRegion:
        return RegisteredModelRegion(
            region_id=handle.region_id,
            server_region_token=17,
            capacity=handle.capacity,
            alignment=handle.alignment,
        )

    def begin_access(self, request: ModeledAccessRequest) -> ModelCompletion:
        self.requests[request.op_id] = request
        return ModelCompletion(
            request.op_id,
            "pending",
            len(self.requests),
            20,
            100,
            request.start_ns + 100,
            None,
        )

    def data_complete(self, op_id: str, status: str, complete_ns: int) -> None:
        self.data_terminals.append((op_id, status, complete_ns))

    def await_completion(self, op_id: str) -> ModelCompletion:
        request = self.requests[op_id]
        return ModelCompletion(
            op_id,
            "ok",
            len(self.requests),
            20,
            100,
            request.start_ns + 100,
            None,
        )

    def cancel(self, op_id: str, reason: str) -> None:
        del op_id, reason

    def close(self) -> None:
        self.closed = True


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


class _TwoGroupsManager(_GroupsManager):
    object_groups = [_ObjectGroup((0,)), _ObjectGroup((1,))]
    num_kernel_groups = 2
    num_object_groups = 2

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx in (0, 1)
        return 16

    def get_attn_desc(self) -> SimpleNamespace:
        return SimpleNamespace(num_chunks_in_sw=[-1, -1])


class _SlidingGroupsManager(_GroupsManager):
    def get_attn_desc(self) -> SimpleNamespace:
        return SimpleNamespace(num_chunks_in_sw=[2])


class _CacheContext:
    device = torch.device("cuda:0")
    stream = object()
    cupy_stream = object()
    lmcache_tokens_per_chunk = 16
    num_layers = 1
    max_batch_size = 1
    kv_layer_groups_manager = _GroupsManager()

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
        return "paged-pointers"

    def get_shape_desc(self, kernel_group_idx: int) -> SimpleNamespace:
        assert kernel_group_idx == 0
        return SimpleNamespace(bs=16)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 16

    def close(self) -> None:
        pass


class _TwoGroupCacheContext(_CacheContext):
    kv_layer_groups_manager = _TwoGroupsManager()

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        assert num_tokens == 16
        assert kernel_group_idx in (0, 1)
        return (2, 1, 16, 4), torch.float16

    def get_engine_kv_format(self, kernel_group_idx: int) -> str:
        assert kernel_group_idx in (0, 1)
        return "NL_X_NB_BS_HS"

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        assert kernel_group_idx in (0, 1)
        return (num_tokens + 15) // 16

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> str:
        assert kernel_group_idx in (0, 1)
        return "paged-pointers"

    def get_shape_desc(self, kernel_group_idx: int) -> SimpleNamespace:
        assert kernel_group_idx in (0, 1)
        return SimpleNamespace(bs=16)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx in (0, 1)
        return 16


class _SlidingCacheContext(_CacheContext):
    kv_layer_groups_manager = _SlidingGroupsManager()


class _NativeCacheContext(_CacheContext):
    lmcache_tokens_per_chunk = 32

    def __init__(self, native_ops: object) -> None:
        self.device = torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        self.cupy_stream = object()
        self._native_ops = native_ops
        self.kv_layer_groups_manager = _NativeGroupsManager()
        self._num_blocks = 8
        self._head_size = 8
        self.paged = torch.arange(
            2 * self._num_blocks * 16 * self._head_size,
            dtype=torch.float32,
            device=self.device,
        ).to(torch.float16)
        self.paged = self.paged.view(2, self._num_blocks, 16, 1, self._head_size)
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


class _EventBackend:
    def check_event_support(self, device: torch.device) -> None:
        del device

    def create_event(self, device: torch.device) -> object:
        del device
        return object()

    def import_event(self, handle: bytes, device: torch.device) -> object:
        del handle, device
        return object()

    def wait_event(self, event: object, stream: object) -> None:
        del event, stream

    def record_event(self, event: object, stream: object) -> None:
        del event, stream

    def export_event(self, event: object, device: torch.device) -> bytes:
        del event, device
        return b"cxl-complete"


class _EventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def publish_on_stream(self, stream: object, event: Event) -> None:
        del stream
        self.events.append(event)


class _NoDRAMStorageManager:
    def finish_write(self, keys: list[ObjectKey]) -> None:
        del keys

    def finish_read_prefetched(self, keys: list[ObjectKey]) -> None:
        del keys

    def reserve_write(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("CXL direct STORE touched DRAM storage")

    def read_prefetched_results(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("CXL direct RETRIEVE touched DRAM storage")


class _DRAMMemoryObj:
    def get_size(self) -> int:
        return 256


class _RecordingDRAMStorageManager:
    def __init__(self) -> None:
        self.reads: list[list[ObjectKey]] = []

    def finish_write(self, keys: list[ObjectKey]) -> None:
        del keys

    def finish_read_prefetched(self, keys: list[ObjectKey]) -> None:
        del keys

    @contextmanager
    def read_prefetched_results(
        self, keys: list[ObjectKey]
    ) -> Iterator[list[_DRAMMemoryObj]]:
        self.reads.append(keys)
        yield [_DRAMMemoryObj() for _ in keys]


class _NoopDispatcher:
    def register(self, kind: str, handler: object, payload_type: object) -> None:
        del kind, handler, payload_type

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _RecordingSharedTier:
    def __init__(
        self,
        result: CXLTransferResult,
        *,
        contains_hit: bool = True,
    ) -> None:
        self.result = result
        self.contains_hit = contains_hit
        self.store_requests: list[VLLMTransferRequest] = []
        self.retrieve_requests: list[VLLMTransferRequest] = []

    def register_engine(
        self, instance_id: int, cache_context: object, model_name: str
    ) -> None:
        del instance_id, cache_context, model_name

    def unregister_engine(self, instance_id: int) -> None:
        del instance_id

    def store(self, request: VLLMTransferRequest) -> CXLTransferResult:
        self.store_requests.append(request)
        return self.result

    def contains(self, request: VLLMTransferRequest) -> bool:
        del request
        return self.contains_hit

    def retrieve(self, request: VLLMTransferRequest) -> CXLTransferResult:
        self.retrieve_requests.append(request)
        return self.result

    def close(self) -> None:
        pass


def _direct_result(status: str = "ok") -> CXLTransferResult:
    return CXLTransferResult(
        status=status,  # type: ignore[arg-type]
        path="cxl_direct",
        completions=(
            CompositeCompletion(
                op_id="op",
                cuda_status="ok" if status == "ok" else "error",
                modeled_status="not_required",
                cuda_elapsed_ns=50 if status == "ok" else None,
                modeled_queue_ns=None,
                modeled_service_ns=None,
                error=None if status == "ok" else "injected failure",
            ),
        ),
        payload_bytes=256,
        dram_allocated_bytes_delta=0,
        error=None if status != "error" else "injected failure",
    )


def _key(value: bytes) -> ObjectKey:
    return ObjectKey(value, "Qwen2.5-7B-Instruct", 0)


def _group_key(value: bytes, object_group_id: int) -> ObjectKey:
    return ObjectKey(
        value,
        "Qwen2.5-7B-Instruct",
        0,
        object_group_id=object_group_id,
    )


def _request(op_id: str, keys: tuple[ObjectKey, ...]) -> VLLMTransferRequest:
    return VLLMTransferRequest(
        op_id=op_id,
        instance_id=7,
        model_name="Qwen2.5-7B-Instruct",
        token_count=16,
        object_keys=keys,
        block_ids_by_group=(tuple(range(10, 10 + len(keys))),),
    )


def _open_module(
    native: _FakeNativeOps,
    cache_context: _CacheContext | None = None,
    *,
    register_engine: bool = True,
) -> tuple[CXLSharedTierModule, shared_memory.SharedMemory]:
    capacity = 4096
    name = f"beluga-gate-b-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    header = pack_region_header(capacity, 64)
    shm.buf[: len(header)] = header
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
        runtime=_FakeRuntime(),
        clock_ns=lambda: 100,
    )
    if register_engine:
        module.register_engine(
            7, cache_context or _CacheContext(), "Qwen2.5-7B-Instruct"
        )
    return module, shm


def test_cxlmemsim_modeled_completion_is_used_by_store_and_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 4096
    data_offset = 4096
    name = f"cxlmemsim-gate-d-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=data_offset + capacity
    )
    header = struct.pack(
        "<QQQQQQQ",
        0x43584C4D454D5348,
        1,
        len(shm.buf),
        data_offset,
        0,
        capacity // 64,
        0,
    )
    shm.buf[: len(header)] = header
    model = _FakeModelClient()
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.cxl_shared_tier.CXLMemSimModelClient.open",
        lambda **_: model,
    )
    native = _FakeNativeOps()
    sink = InMemoryCXLOperationSink()
    module = CXLSharedTierModule.open(
        CXLSharedTierConfig(
            enabled=True,
            provider="cxlmemsim_shm",
            shm_name=f"/{name}",
            capacity_bytes=capacity,
            alignment_bytes=64,
            model_mode="cxlmemsim",
            model_client_library="/unused/libmodeled.so",
        ),
        native_ops=native,
        runtime=_FakeRuntime(),
        clock_ns=lambda: 100,
        operation_sink=sink,
    )
    try:
        module.register_engine(7, _CacheContext(), "Qwen2.5-7B-Instruct")
        keys = (_key(b"modeled"),)

        stored = module.store(_request("modeled-store", keys))
        retrieved = module.retrieve(_request("modeled-retrieve", keys))

        assert stored.completions[0].modeled_status == "ok"
        assert stored.completions[0].effective_elapsed_ns == 100
        assert retrieved.completions[0].modeled_status == "ok"
        assert [item.direction for item in model.requests.values()] == [
            "store",
            "retrieve",
        ]
        terminal = [event for event in sink.snapshot() if event.terminal]
        assert terminal[-1].modeled_queue_ns == 20
        assert terminal[-1].modeled_service_ns == 100
        assert terminal[-1].effective_elapsed_ns == 100
    finally:
        module.close()
        shm.close()
        shm.unlink()
    assert model.closed


def test_store_publishes_immutable_chunks_only_after_cuda_success() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native)
    keys = (_key(b"a"), _key(b"b"))
    try:
        result = module.store(_request("store-1", keys))

        assert result.status == "ok"
        assert result.path == "cxl_direct"
        assert result.dram_allocated_bytes_delta == 0
        assert len(result.completions) == 2
        assert [transfer[5] for transfer in native.transfers] == ["d2h", "d2h"]
        assert all(
            module.lookup_ready(key).state == ResidencyState.READY for key in keys
        )

        duplicate = module.store(_request("store-duplicate", keys))
        assert duplicate.status == "ok"
        assert len(native.transfers) == 2
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_gate_e_policy_binds_before_hit_and_recompute_returns_no_ticket() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native)
    keys = (_key(b"policy-a"), _key(b"policy-b"))
    try:
        assert module.store(_request("store-policy", keys)).status == "ok"
        fingerprint = module.lookup_ready(keys[0]).descriptor.layout_fingerprint
        fetch = GateERequestEnvelope(
            GATE_E_PROTOCOL_VERSION,
            "request",
            None,
            1_000_000_000,
            fingerprint,
        )

        assert module.policy_bind_ready_prefix("request", keys, 1, fetch) == 2
        assert len(module.get_bound_ticket_ids("request")) == 2
        module.release_lookup_tickets("request", "APC overlap", object_keys=(keys[0],))
        assert len(module.get_bound_ticket_ids("request")) == 1
        module.release_lookup_tickets("request", "lookup replaced")

        recompute = GateERequestEnvelope(
            GATE_E_PROTOCOL_VERSION,
            "request",
            None,
            1,
            fingerprint,
        )
        assert module.policy_bind_ready_prefix("request", keys, 1, recompute) == 0
        assert module.get_bound_ticket_ids("request") == ()
        assert '"winner": "recompute"' in module.get_lookup_decision_reason("request")
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_registration_rejects_sliding_window_layouts_fail_closed() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native, register_engine=False)
    try:
        with pytest.raises(ValueError, match="full-attention"):
            module.register_engine(7, _SlidingCacheContext(), "Qwen2.5-7B-Instruct")
        assert native.registrations == []
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_store_cuda_failure_aborts_without_directory_publication() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native)
    key = _key(b"failed")
    native.fail_transfer = True
    try:
        result = module.store(_request("store-fail", (key,)))

        assert result.status == "error"
        assert "injected CUDA failure" in (result.error or "")
        assert module.lookup_ready(key) is None

        native.fail_transfer = False
        retry = module.store(_request("store-retry", (key,)))
        assert retry.status == "ok"
        assert module.lookup_ready(key).state == ResidencyState.READY
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_store_rejects_alias_collision_before_copy_or_publication() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native, _TwoGroupCacheContext())
    shared_alias = _group_key(b"shared-alias", 1)
    first_primary = _group_key(b"first-primary", 0)
    conflicting_primary = _group_key(b"conflicting-primary", 0)

    def request(op_id: str, primary: ObjectKey) -> VLLMTransferRequest:
        return VLLMTransferRequest(
            op_id=op_id,
            instance_id=7,
            model_name="Qwen2.5-7B-Instruct",
            token_count=16,
            object_keys=(primary, shared_alias),
            block_ids_by_group=((1,), (2,)),
        )

    try:
        assert module.store(request("first", first_primary)).status == "ok"
        transfers_after_first = len(native.transfers)

        result = module.store(request("conflict", conflicting_primary))

        assert result.status == "error"
        assert "alias" in (result.error or "")
        assert len(native.transfers) == transfers_after_first
        assert module.lookup_ready(conflicting_primary) is None
        assert module.lookup_ready(shared_alias) == module.lookup_ready(first_primary)
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_retrieve_uses_ready_extent_and_releases_its_read_lease() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native)
    key = _key(b"ready")
    try:
        assert module.store(_request("store", (key,))).status == "ok"

        result = module.retrieve(_request("retrieve", (key,)))

        assert result.status == "ok"
        assert result.dram_allocated_bytes_delta == 0
        assert [transfer[5] for transfer in native.transfers] == ["d2h", "h2d"]
        module.evict(key)
        assert module.lookup_ready(key) is None
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_retrieve_reports_cxl_miss_without_launch_or_dram_staging() -> None:
    native = _FakeNativeOps()
    module, shm = _open_module(native)
    try:
        result = module.retrieve(_request("retrieve-miss", (_key(b"missing"),)))

        assert result.status == "miss"
        assert result.path == "cxl_direct"
        assert result.dram_allocated_bytes_delta == 0
        assert native.transfers == []
    finally:
        module.close()
        shm.close()
        shm.unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_native_single_instance_round_trip_restores_paged_kv_bytes() -> None:
    import lmcache.c_ops as native_ops

    if not hasattr(native_ops, "CudaRegionRegistration"):
        pytest.skip("CXL region native extension is not built")
    capacity = 8192
    name = f"beluga-gate-b-native-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    header = pack_region_header(capacity, 64)
    shm.buf[: len(header)] = header
    module = CXLSharedTierModule.open(
        CXLSharedTierConfig(
            enabled=True,
            provider="posix_shm",
            shm_name=f"/{name}",
            capacity_bytes=capacity,
            alignment_bytes=64,
            layout_id="packed_kv_v1",
        )
    )
    context = _NativeCacheContext(native_ops)
    module.register_engine(7, context, "Qwen2.5-7B-Instruct")
    keys = (_key(b"native-a"), _key(b"native-b"))
    block_ids = (3, 1, 7, 0)
    request = VLLMTransferRequest(
        op_id="native-store",
        instance_id=7,
        model_name="Qwen2.5-7B-Instruct",
        token_count=32,
        object_keys=keys,
        block_ids_by_group=(block_ids,),
    )
    expected = context.paged[:, list(block_ids)].clone()
    torch.cuda.synchronize()
    try:
        store = module.store(request)
        assert store.status == "ok"
        assert store.payload_bytes == expected.numel() * expected.element_size()
        assert store.dram_allocated_bytes_delta == 0
        context.paged[:, list(block_ids)] = 0
        torch.cuda.synchronize()

        retrieve = module.retrieve(
            VLLMTransferRequest(
                op_id="native-retrieve",
                instance_id=request.instance_id,
                model_name=request.model_name,
                token_count=request.token_count,
                object_keys=request.object_keys,
                block_ids_by_group=request.block_ids_by_group,
            )
        )

        assert retrieve.status == "ok"
        assert retrieve.payload_bytes == expected.numel() * expected.element_size()
        assert retrieve.dram_allocated_bytes_delta == 0
        assert torch.equal(context.paged[:, list(block_ids)], expected)
    finally:
        module.close()
        shm.close()
        shm.unlink()


def test_lmcache_store_delegates_to_cxl_without_touching_dram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as transfer_module,
    )

    object_key = _key(b"delegated")
    event_bus = _EventBus()
    ctx = SimpleNamespace(
        chunk_size=16,
        separate_object_groups=True,
        full_sw_kv=False,
        storage_manager=_NoDRAMStorageManager(),
        layout_desc_registry=LayoutDescRegistry(),
        event_bus=event_bus,
        resolve_obj_keys=lambda key, group_ids: [[object_key]],
    )
    cache_context = _CacheContext()
    backend = _EventBackend()
    shared_tier = _RecordingSharedTier(_direct_result())
    monkeypatch.setattr(transfer_module, "DeviceHostFuncDispatcher", _NoopDispatcher)
    monkeypatch.setattr(
        transfer_module, "create_cache_context", lambda *args, **kwargs: cache_context
    )
    monkeypatch.setattr(
        transfer_module,
        "get_layout_desc",
        lambda *args, **kwargs: MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, 16, 4])], dtypes=[torch.float16]
        ),
    )
    monkeypatch.setattr(
        transfer_module, "get_event_ipc_backend", lambda device: backend
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "device", lambda device: nullcontext()
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "stream", lambda stream: nullcontext()
    )

    module = transfer_module.LMCacheDrivenTransferModule(
        ctx, cxl_shared_tier=shared_tier
    )
    module.register_kv_cache(7, [], "Qwen2.5-7B-Instruct", 1, EngineType.VLLM, {}, [])
    key = IPCCacheServerKey.from_token_ids(
        model_name="Qwen2.5-7B-Instruct",
        world_size=1,
        worker_id=0,
        token_ids=list(range(16)),
        start=0,
        end=16,
        request_id="request-1",
    )

    event_handle, succeeded = module.store(key, 7, [[4]], b"producer")

    assert event_handle == b"cxl-complete"
    assert succeeded is True
    assert len(shared_tier.store_requests) == 1
    request = shared_tier.store_requests[0]
    assert request.instance_id == 7
    assert request.object_keys == (object_key,)
    assert request.block_ids_by_group == ((4,),)
    assert any(event.metadata.get("path") == "cxl_direct" for event in event_bus.events)
    module.close()


def test_lmcache_retrieve_cxl_error_does_not_fallback_to_dram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as transfer_module,
    )

    object_key = _key(b"retrieve-error")
    event_bus = _EventBus()
    ctx = SimpleNamespace(
        chunk_size=16,
        separate_object_groups=True,
        full_sw_kv=False,
        storage_manager=_NoDRAMStorageManager(),
        layout_desc_registry=LayoutDescRegistry(),
        event_bus=event_bus,
        resolve_obj_keys=lambda key, group_ids: [[object_key]],
    )
    cache_context = _CacheContext()
    backend = _EventBackend()
    shared_tier = _RecordingSharedTier(_direct_result("error"))
    monkeypatch.setattr(transfer_module, "DeviceHostFuncDispatcher", _NoopDispatcher)
    monkeypatch.setattr(
        transfer_module, "create_cache_context", lambda *args, **kwargs: cache_context
    )
    monkeypatch.setattr(
        transfer_module,
        "get_layout_desc",
        lambda *args, **kwargs: MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, 16, 4])], dtypes=[torch.float16]
        ),
    )
    monkeypatch.setattr(
        transfer_module, "get_event_ipc_backend", lambda device: backend
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "device", lambda device: nullcontext()
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "stream", lambda stream: nullcontext()
    )

    module = transfer_module.LMCacheDrivenTransferModule(
        ctx, cxl_shared_tier=shared_tier
    )
    module.register_kv_cache(7, [], "Qwen2.5-7B-Instruct", 1, EngineType.VLLM, {}, [])
    key = IPCCacheServerKey.from_token_ids(
        model_name="Qwen2.5-7B-Instruct",
        world_size=1,
        worker_id=0,
        token_ids=list(range(16)),
        start=0,
        end=16,
        request_id="request-2",
    )

    event_handle, succeeded = module.retrieve(key, 7, [[9]], b"producer")

    assert event_handle == b"cxl-complete"
    assert succeeded is False
    assert len(shared_tier.retrieve_requests) == 1
    assert any(event.metadata.get("path") == "cxl_direct" for event in event_bus.events)
    module.close()


def test_lmcache_retrieve_cxl_miss_preserves_dram_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as transfer_module,
    )

    object_key = _key(b"dram-fallback")
    event_bus = _EventBus()
    storage_manager = _RecordingDRAMStorageManager()
    ctx = SimpleNamespace(
        chunk_size=16,
        separate_object_groups=True,
        full_sw_kv=False,
        storage_manager=storage_manager,
        layout_desc_registry=LayoutDescRegistry(),
        event_bus=event_bus,
        resolve_obj_keys=lambda key, group_ids: [[object_key]],
    )
    cache_context = _CacheContext()
    backend = _EventBackend()
    shared_tier = _RecordingSharedTier(_direct_result(), contains_hit=False)
    monkeypatch.setattr(transfer_module, "DeviceHostFuncDispatcher", _NoopDispatcher)
    monkeypatch.setattr(
        transfer_module, "create_cache_context", lambda *args, **kwargs: cache_context
    )
    monkeypatch.setattr(
        transfer_module,
        "get_layout_desc",
        lambda *args, **kwargs: MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, 16, 4])], dtypes=[torch.float16]
        ),
    )
    monkeypatch.setattr(
        transfer_module, "get_event_ipc_backend", lambda device: backend
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "device", lambda device: nullcontext()
    )
    monkeypatch.setattr(
        transfer_module.torch_dev, "stream", lambda stream: nullcontext()
    )
    monkeypatch.setattr(
        transfer_module,
        "transfer_kv_per_object_group",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        transfer_module,
        "submit_callback_to_stream",
        lambda *args, **kwargs: None,
    )

    module = transfer_module.LMCacheDrivenTransferModule(
        ctx, cxl_shared_tier=shared_tier
    )
    module.register_kv_cache(7, [], "Qwen2.5-7B-Instruct", 1, EngineType.VLLM, {}, [])
    key = IPCCacheServerKey.from_token_ids(
        model_name="Qwen2.5-7B-Instruct",
        world_size=1,
        worker_id=0,
        token_ids=list(range(16)),
        start=0,
        end=16,
        request_id="request-dram",
    )

    event_handle, succeeded = module.retrieve(key, 7, [[3]], b"producer")

    assert event_handle == b"cxl-complete"
    assert succeeded is True
    assert shared_tier.retrieve_requests == []
    assert storage_manager.reads == [[object_key]]
    assert any(event.metadata.get("path") == "dram" for event in event_bus.events)
    module.close()
