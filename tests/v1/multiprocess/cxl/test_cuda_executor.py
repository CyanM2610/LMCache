# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Callable
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.contracts import ExtentDescriptor, TransferPlan
from lmcache.v1.multiprocess.cxl.cuda_executor import (
    LMCacheIPCTransferExecutor,
    RegisteredRegionView,
)
from lmcache.v1.multiprocess.cxl.region_provider import (
    REGION_HEADER_SIZE,
    RegionHandle,
    pack_region_header,
)
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager


pytestmark = pytest.mark.no_shared_allocator
FINGERPRINT = "a" * 64


def _accept_descriptor(descriptor: ExtentDescriptor) -> None:
    del descriptor


def _handle(capacity: int = 4096) -> RegionHandle:
    return RegionHandle(
        region_id="proxy0",
        shm_name="/proxy0",
        capacity=capacity,
        alignment=64,
        capabilities=frozenset({"cuda_host_register_v1"}),
    )


def _extent(**overrides: object) -> ExtentDescriptor:
    values = {
        "region_id": "proxy0",
        "offset": 0,
        "length": 512,
        "generation": 1,
        "tier": "cxl",
        "layout_id": "packed_kv_v1",
        "layout_fingerprint": FINGERPRINT,
    }
    values.update(overrides)
    return ExtentDescriptor(**values)  # type: ignore[arg-type]


def _plan(**overrides: object) -> TransferPlan:
    keys = (
        ObjectKey(b"a", "Qwen2.5-7B-Instruct", 0),
        ObjectKey(b"b", "Qwen2.5-7B-Instruct", 0),
    )
    values = {
        "plan_version": 1,
        "op_id": "op-1",
        "direction": "store",
        "instance_id": 3,
        "object_keys": keys,
        "block_ids_by_group": ((9, 4),),
        "extent": _extent(),
        "payload_checksum_expected": None,
    }
    values.update(overrides)
    return TransferPlan(**values)  # type: ignore[arg-type]


class _FakeRegistration:
    def __init__(self, shm_name: str, expected_capacity: int) -> None:
        self.shm_name = shm_name
        self.capacity = expected_capacity
        self.closed = False
        self.resolutions: list[tuple[int, int]] = []

    def device_address(self, offset: int, length: int) -> int:
        self.resolutions.append((offset, length))
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
        self, shm_name: str, expected_capacity: int
    ) -> _FakeRegistration:
        registration = _FakeRegistration(shm_name, expected_capacity)
        self.registrations.append(registration)
        return registration

    def cxl_region_block_kv_transfer(self, *args: object) -> None:
        if self.fail_transfer:
            raise RuntimeError("kernel failure")
        self.transfers.append(args)


class _FakeRuntime:
    def __init__(self) -> None:
        self.executions = 0

    def execute(self, cache_context: object, operation: Callable[[], None]) -> int:
        del cache_context
        self.executions += 1
        operation()
        return 50


@dataclass
class _ObjectGroup:
    kernel_group_indices: tuple[int, ...]


class _CacheContext:
    device = torch.device("cuda:0")
    stream = object()
    lmcache_tokens_per_chunk = 16
    kv_layer_groups_manager = SimpleNamespace(
        object_groups=[_ObjectGroup((0,))], num_kernel_groups=1
    )

    def __init__(self) -> None:
        self.staged_groups: list[list[list[int]]] = []

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        assert num_tokens == 16
        assert kernel_group_idx == 0
        return (2, 1, 16, 4), torch.float16

    def stage_block_ids(self, groups: list[list[int]]) -> list[list[int]]:
        self.staged_groups.append(groups)
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

    def get_engine_kv_format(self, kernel_group_idx: int) -> str:
        assert kernel_group_idx == 0
        return "format"


class _MultiKernelCacheContext(_CacheContext):
    kv_layer_groups_manager = SimpleNamespace(
        object_groups=[_ObjectGroup((0, 1))], num_kernel_groups=2
    )

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        assert num_tokens == 16
        width = 2 if kernel_group_idx == 0 else 6
        return (1, 1, 16, width), torch.float16

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> str:
        return f"paged-pointers-{kernel_group_idx}"

    def get_shape_desc(self, kernel_group_idx: int) -> SimpleNamespace:
        assert kernel_group_idx in (0, 1)
        return SimpleNamespace(bs=16)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx in (0, 1)
        return 16

    def get_engine_kv_format(self, kernel_group_idx: int) -> str:
        return f"format-{kernel_group_idx}"


def test_registered_region_opens_once_and_resolves_region_relative_bounds() -> None:
    native = _FakeNativeOps()
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=_accept_descriptor
    )

    first = view.resolve(_extent(offset=64, length=128))
    second = view.resolve(_extent(offset=256, length=64))

    assert first == 0x100040
    assert second == 0x100100
    assert len(native.registrations) == 1
    assert native.registrations[0].resolutions == [(64, 128), (256, 64)]


@pytest.mark.parametrize(
    "extent",
    [
        _extent(region_id="other"),
        _extent(offset=4090, length=16),
        _extent(layout_fingerprint="f" * 64),
    ],
)
def test_registered_region_rejects_identity_bounds_and_layout_before_cuda(
    extent: ExtentDescriptor,
) -> None:
    native = _FakeNativeOps()
    view = RegisteredRegionView.open(
        _handle(),
        native_ops=native,
        expected_layout_fingerprint=FINGERPRINT,
        descriptor_validator=_accept_descriptor,
    )

    with pytest.raises(ValueError):
        view.resolve(extent)
    assert native.registrations[0].resolutions == []


def test_executor_submits_packed_objects_directly_from_registered_region() -> None:
    native = _FakeNativeOps()
    runtime = _FakeRuntime()
    view = RegisteredRegionView.open(
        _handle(),
        native_ops=native,
        expected_layout_fingerprint=FINGERPRINT,
        descriptor_validator=_accept_descriptor,
    )
    executor = LMCacheIPCTransferExecutor(
        view, native_ops=native, runtime=runtime, clock_ns=lambda: 100
    )
    executor.register_engine(3, _CacheContext())

    completion = executor.submit(_plan())

    assert completion.status == "ok"
    assert completion.elapsed_ns == 50
    assert runtime.executions == 1
    assert len(native.transfers) == 1
    assert native.transfers[0][1] == [0x100000, 0x100100]
    assert native.transfers[0][5] == "d2h"


def test_executor_offsets_each_kernel_group_within_packed_objects() -> None:
    native = _FakeNativeOps()
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=_accept_descriptor
    )
    executor = LMCacheIPCTransferExecutor(
        view, native_ops=native, runtime=_FakeRuntime()
    )
    executor.register_engine(3, _MultiKernelCacheContext())
    plan = _plan(block_ids_by_group=((9, 4), (9, 4)))

    completion = executor.submit(plan)

    assert completion.status == "ok"
    assert len(native.transfers) == 2
    assert native.transfers[0][1] == [0x100000, 0x100100]
    assert native.transfers[1][1] == [0x100040, 0x100140]


def test_executor_does_not_launch_after_stale_generation_validation() -> None:
    native = _FakeNativeOps()
    runtime = _FakeRuntime()
    view = RegisteredRegionView.open(
        _handle(),
        native_ops=native,
        descriptor_validator=lambda descriptor: (_ for _ in ()).throw(
            ValueError(f"stale generation {descriptor.generation}")
        ),
    )
    executor = LMCacheIPCTransferExecutor(view, native_ops=native, runtime=runtime)
    executor.register_engine(3, _CacheContext())

    with pytest.raises(ValueError, match="stale generation"):
        executor.submit(_plan())
    assert runtime.executions == 0
    assert native.transfers == []


def test_executor_reports_cuda_error_without_fallback_copy() -> None:
    native = _FakeNativeOps()
    native.fail_transfer = True
    runtime = _FakeRuntime()
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=_accept_descriptor
    )
    executor = LMCacheIPCTransferExecutor(
        view, native_ops=native, runtime=runtime, clock_ns=lambda: 100
    )
    executor.register_engine(3, _CacheContext())

    completion = executor.submit(_plan())

    assert completion.status == "error"
    assert completion.error == "kernel failure"
    assert runtime.executions == 1


def test_registration_failure_propagates_without_launching_a_copy() -> None:
    class _FailingNativeOps(_FakeNativeOps):
        def CudaRegionRegistration(
            self, shm_name: str, expected_capacity: int
        ) -> _FakeRegistration:
            del shm_name, expected_capacity
            raise RuntimeError("cudaHostRegister failed")

    native = _FailingNativeOps()

    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        RegisteredRegionView.open(
            _handle(), native_ops=native, descriptor_validator=_accept_descriptor
        )
    assert native.transfers == []


def test_executor_reconstructs_prefix_skip_for_paged_kernel_alignment() -> None:
    native = _FakeNativeOps()
    runtime = _FakeRuntime()
    context = _CacheContext()
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=_accept_descriptor
    )
    executor = LMCacheIPCTransferExecutor(view, native_ops=native, runtime=runtime)
    executor.register_engine(3, context)
    plan = _plan(direction="retrieve", block_ids_by_group=((4,),))

    completion = executor.submit(plan)

    assert completion.status == "ok"
    assert context.staged_groups == [[[0, 4]]]
    assert native.transfers[0][8] == 1


def test_unregister_and_close_release_context_and_region_registration() -> None:
    native = _FakeNativeOps()
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=_accept_descriptor
    )
    executor = LMCacheIPCTransferExecutor(view, native_ops=native)
    executor.register_engine(3, _CacheContext())

    executor.unregister_engine(3)
    with pytest.raises(KeyError, match="instance"):
        executor.submit(_plan())
    view.close()
    assert native.registrations[0].closed


def test_registered_region_rejects_reclaimed_generation_before_native_resolve() -> None:
    native = _FakeNativeOps()
    manager = CXLRegionManager(
        _handle(),
        layout_id="packed_kv_v1",
        layout_fingerprint=FINGERPRINT,
    )
    reservation = manager.reserve(512, 64)
    stale = manager.begin_write(reservation.reservation_id)
    manager.publish(reservation.reservation_id)
    manager.begin_evict(stale)
    manager.reclaim(stale)
    replacement = manager.reserve(512, 64)
    assert replacement.offset == stale.offset
    assert replacement.generation > stale.generation
    view = RegisteredRegionView.open(
        _handle(), native_ops=native, descriptor_validator=manager.validate_descriptor
    )

    with pytest.raises(ValueError, match="stale"):
        view.resolve(stale)
    assert native.registrations[0].resolutions == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_native_registered_region_contiguous_round_trip() -> None:
    import lmcache.c_ops as native_ops

    if not hasattr(native_ops, "CudaRegionRegistration"):
        pytest.skip("CXL region native extension is not built")
    name = f"beluga-cuda-{uuid.uuid4().hex}"
    capacity = 4096
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    try:
        header = pack_region_header(capacity, 4096)
        shm.buf[: len(header)] = header
        registration = native_ops.CudaRegionRegistration(f"/{name}", capacity)
        source = torch.arange(1024, device="cuda", dtype=torch.int32)
        destination = torch.zeros_like(source)
        stream = torch.cuda.current_stream()

        registration.copy_from_device(
            source.data_ptr(), 0, source.nbytes, stream.cuda_stream
        )
        registration.copy_to_device(
            destination.data_ptr(), 0, destination.nbytes, stream.cuda_stream
        )
        stream.synchronize()

        assert torch.equal(source, destination)
        registration.close()
    finally:
        shm.close()
        shm.unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("block_ids", [(0, 1, 2, 3), (3, 1, 7, 0)])
def test_native_registered_region_paged_kv_round_trip(
    block_ids: tuple[int, ...],
) -> None:
    import lmcache.c_ops as native_ops

    if not hasattr(native_ops, "cxl_region_block_kv_transfer"):
        pytest.skip("CXL region native extension is not built")
    name = f"beluga-paged-{uuid.uuid4().hex}"
    capacity = 4096
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    registration = None
    try:
        header = pack_region_header(capacity, 4096)
        shm.buf[: len(header)] = header
        registration = native_ops.CudaRegionRegistration(f"/{name}", capacity)
        num_blocks = 8
        block_size = 16
        chunk_size = 32
        head_size = 8
        paged = torch.arange(
            2 * num_blocks * block_size * head_size,
            dtype=torch.float32,
            device="cuda",
        ).to(torch.float16)
        paged = paged.view(2, num_blocks, block_size, 1, head_size)
        expected = paged[:, list(block_ids)].clone()
        paged_pointers = torch.tensor(
            [paged.data_ptr()], dtype=torch.int64, device="cuda"
        )
        staged_block_ids = torch.tensor(block_ids, dtype=torch.int64, device="cuda")
        shape = native_ops.PageBufferShapeDesc()
        shape.kv_size = 2
        shape.nl = 1
        shape.nb = num_blocks
        shape.bs = block_size
        shape.nh = 1
        shape.hs = head_size
        shape.element_size = 2
        shape.block_stride_elems = 0
        bytes_per_object = 2 * chunk_size * head_size * 2
        object_pointers = [
            registration.device_address(0, bytes_per_object),
            registration.device_address(bytes_per_object, bytes_per_object),
        ]
        device = torch.device("cuda:0")
        kv_format = native_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS

        native_ops.cxl_region_block_kv_transfer(
            paged_pointers,
            object_pointers,
            staged_block_ids,
            device,
            shape,
            native_ops.TransferDirection.D2H,
            chunk_size,
            kv_format,
            0,
        )
        paged.zero_()
        native_ops.cxl_region_block_kv_transfer(
            paged_pointers,
            object_pointers,
            staged_block_ids,
            device,
            shape,
            native_ops.TransferDirection.H2D,
            chunk_size,
            kv_format,
            0,
        )
        torch.cuda.synchronize()

        assert torch.equal(paged[:, list(block_ids)], expected)
    finally:
        if registration is not None:
            registration.close()
        shm.close()
        shm.unlink()
