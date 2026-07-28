# SPDX-License-Identifier: Apache-2.0
"""Server-side CUDA execution over a registered CXL proxy region."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Protocol, cast
import importlib
import threading
import time

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.cache_context import BaseCacheContext

# Local
from .bounded import BoundedSet
from .contracts import DataCompletion, ExtentDescriptor, TransferPlan
from .region_provider import RegionHandle


class _RegionRegistration(Protocol):
    capacity: int

    def device_address(self, offset: int, length: int) -> int: ...

    def close(self) -> None: ...


class _NativeOps(Protocol):
    CudaRegionRegistration: Callable[..., _RegionRegistration]
    TransferDirection: Any

    def cxl_region_block_kv_transfer(self, *args: object) -> None: ...


@dataclass(frozen=True)
class _KernelPlacement:
    kernel_group_id: int
    offset: int
    length: int


class _ExecutionRuntime(Protocol):
    def execute(
        self, cache_context: BaseCacheContext, operation: Callable[[], None]
    ) -> int:
        """Run one operation on the context stream and return elapsed ns."""
        ...


class _TorchCudaRuntime:
    def execute(
        self, cache_context: BaseCacheContext, operation: Callable[[], None]
    ) -> int:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with (
            torch.cuda.device(cache_context.device),
            torch.cuda.stream(cache_context.stream),
        ):
            start.record(cache_context.stream)
            operation()
            end.record(cache_context.stream)
        end.synchronize()
        return int(start.elapsed_time(end) * 1_000_000)


class RegisteredRegionView:
    """Process-local CUDA registration resolved through safe extent metadata."""

    def __init__(
        self,
        handle: RegionHandle,
        registration: _RegionRegistration,
        *,
        descriptor_validator: Callable[[ExtentDescriptor], None],
        expected_layout_fingerprint: str | None = None,
    ) -> None:
        if registration.capacity != handle.capacity:
            raise ValueError("native registration capacity does not match handle")
        self._handle = handle
        self._registration = registration
        self._expected_layout_fingerprint = expected_layout_fingerprint
        self._descriptor_validator = descriptor_validator
        self._closed = False

    @classmethod
    def open(
        cls,
        handle: RegionHandle,
        *,
        descriptor_validator: Callable[[ExtentDescriptor], None],
        native_ops: _NativeOps | None = None,
        expected_layout_fingerprint: str | None = None,
    ) -> RegisteredRegionView:
        """Register a provisioned region once for CUDA mapped access.

        Args:
            handle: Validated process-independent region description.
            descriptor_validator: Current-generation lifecycle validator.
            native_ops: Optional native module replacement at the CUDA boundary.
            expected_layout_fingerprint: Optional exact layout identity.

        Returns:
            The process-local registered region view.

        Raises:
            RuntimeError: If capability, mapping, or CUDA registration fails.
            ValueError: If native capacity differs from the region handle.
        """
        if "cuda_host_register_v1" not in handle.capabilities:
            raise RuntimeError("region is not CUDA-registerable")
        ops = cast(
            _NativeOps,
            native_ops or importlib.import_module("lmcache.c_ops"),
        )
        try:
            registration = ops.CudaRegionRegistration(
                handle.shm_name, handle.capacity, handle.data_offset
            )
        except TypeError:
            if handle.data_offset != 4096:
                raise
            registration = ops.CudaRegionRegistration(handle.shm_name, handle.capacity)
        return cls(
            handle,
            registration,
            descriptor_validator=descriptor_validator,
            expected_layout_fingerprint=expected_layout_fingerprint,
        )

    def resolve(self, descriptor: ExtentDescriptor) -> int:
        """Resolve a validated region-relative extent to a device address.

        Args:
            descriptor: Current-generation extent metadata.

        Returns:
            Process-local device address used only for immediate CUDA launch.

        Raises:
            RuntimeError: If the view has been closed.
            ValueError: If identity, bounds, layout, or generation is invalid.
        """
        if self._closed:
            raise RuntimeError("registered region is closed")
        if descriptor.region_id != self._handle.region_id:
            raise ValueError("extent belongs to another region")
        if descriptor.offset + descriptor.length > self._handle.capacity:
            raise ValueError("extent exceeds registered region bounds")
        if (
            self._expected_layout_fingerprint is not None
            and descriptor.layout_fingerprint != self._expected_layout_fingerprint
        ):
            raise ValueError("extent layout does not match registered view")
        self._descriptor_validator(descriptor)
        return int(
            self._registration.device_address(descriptor.offset, descriptor.length)
        )

    def close(self) -> None:
        """Unregister and unmap the region exactly once."""
        if not self._closed:
            self._registration.close()
            self._closed = True


class LMCacheIPCTransferExecutor:
    """Only copy executor for direct vLLM HBM/CXL proxy transfers."""

    def __init__(
        self,
        region_view: RegisteredRegionView,
        *,
        native_ops: _NativeOps | None = None,
        runtime: _ExecutionRuntime | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._region_view = region_view
        self._native_ops = native_ops
        self._runtime = runtime or _TorchCudaRuntime()
        self._clock_ns = clock_ns
        self._contexts: dict[int, BaseCacheContext] = {}
        self._cancelled: BoundedSet[str] = BoundedSet()
        self._lock = threading.RLock()

    def register_engine(
        self, instance_id: int, cache_context: BaseCacheContext
    ) -> None:
        """Register one imported vLLM CUDA cache context.

        Args:
            instance_id: Non-negative stable engine instance identifier.
            cache_context: Existing server-side LMCache CUDA context.

        Raises:
            ValueError: If identity or device type is invalid.
            RuntimeError: If the instance is already registered.
        """
        if instance_id < 0:
            raise ValueError("instance_id must be non-negative")
        if getattr(cache_context.device, "type", None) != "cuda":
            raise ValueError("CXL direct transfer requires a CUDA cache context")
        with self._lock:
            if instance_id in self._contexts:
                raise RuntimeError("engine instance is already registered")
            self._contexts[instance_id] = cache_context

    def unregister_engine(self, instance_id: int) -> None:
        """Drop the imported context used by one engine instance.

        Args:
            instance_id: Previously registered engine identifier.
        """
        with self._lock:
            self._contexts.pop(instance_id, None)

    def submit(self, plan: TransferPlan) -> DataCompletion:
        """Execute a transfer directly between paged HBM and the extent.

        Args:
            plan: Validated request-local block and extent metadata.

        Returns:
            A terminal CUDA data completion. Descriptor errors are raised before
            launch; launch errors are returned without a staging fallback.

        Raises:
            KeyError: If the engine instance is not registered.
            ValueError: If extent bounds, generation, layout, or size is invalid.
            RuntimeError: If the operation was cancelled before submission.
        """
        with self._lock:
            try:
                cache_context = self._contexts[plan.instance_id]
            except KeyError as error:
                raise KeyError("engine instance is not registered") from error
            if plan.op_id in self._cancelled:
                self._cancelled.discard(plan.op_id)
                raise RuntimeError("operation was cancelled before submission")
        base_address = self._region_view.resolve(plan.extent)
        kernel_pointers = self._resolve_kernel_pointers(
            cache_context, plan, base_address
        )
        staged_block_ids, skip_blocks = self._stage_block_ids(
            cache_context, plan, kernel_pointers
        )
        start_ns = self._clock_ns()
        try:
            elapsed_ns = self._runtime.execute(
                cache_context,
                lambda: self._launch(
                    cache_context,
                    plan,
                    kernel_pointers,
                    staged_block_ids,
                    skip_blocks,
                ),
            )
        except Exception as error:
            return DataCompletion(
                op_id=plan.op_id,
                status="error",
                complete_ns=self._clock_ns(),
                elapsed_ns=None,
                error=str(error),
            )
        return DataCompletion(
            op_id=plan.op_id,
            status="ok",
            complete_ns=start_ns + elapsed_ns,
            elapsed_ns=elapsed_ns,
            error=None,
        )

    def cancel(self, op_id: str) -> None:
        """Prevent a not-yet-submitted operation from launching.

        Args:
            op_id: Logical operation identifier.

        Raises:
            ValueError: If the operation identifier is empty.
        """
        if not op_id:
            raise ValueError("op_id must not be empty")
        with self._lock:
            self._cancelled.add(op_id)

    def _resolve_kernel_pointers(
        self, cache_context: BaseCacheContext, plan: TransferPlan, base_address: int
    ) -> dict[int, list[int]]:
        layouts = self._object_group_layouts(cache_context)
        pointers: dict[int, list[int]] = {
            placement.kernel_group_id: []
            for _, placements in layouts.values()
            for placement in placements
        }
        offset = 0
        for object_key in plan.object_keys:
            group_id = object_key.object_group_id
            if group_id not in layouts:
                raise ValueError("ObjectKey references an unknown object group")
            group_size, placements = layouts[group_id]
            for placement in placements:
                pointers[placement.kernel_group_id].append(
                    base_address + offset + placement.offset
                )
            offset += group_size
        if offset != plan.extent.length:
            raise ValueError(
                "packed object bytes do not exactly fill the extent: "
                f"objects={offset}, extent={plan.extent.length}"
            )
        return pointers

    @staticmethod
    def _object_group_layouts(
        cache_context: BaseCacheContext,
    ) -> dict[int, tuple[int, tuple[_KernelPlacement, ...]]]:
        layouts: dict[int, tuple[int, tuple[_KernelPlacement, ...]]] = {}
        manager = cache_context.kv_layer_groups_manager
        for group_id, object_group in enumerate(manager.object_groups):
            offset = 0
            placements: list[_KernelPlacement] = []
            for kernel_group_id in object_group.kernel_group_indices:
                shape, dtype = cache_context.get_kernel_group_shape_dtype(
                    cache_context.lmcache_tokens_per_chunk, kernel_group_id
                )
                elements = reduce(mul, (int(dimension) for dimension in shape), 1)
                length = elements * torch.empty((), dtype=dtype).element_size()
                placements.append(_KernelPlacement(kernel_group_id, offset, length))
                offset += length
            layouts[group_id] = (offset, tuple(placements))
        return layouts

    @staticmethod
    def _stage_block_ids(
        cache_context: BaseCacheContext,
        plan: TransferPlan,
        kernel_pointers: dict[int, list[int]],
    ) -> tuple[list[Any], dict[int, int]]:
        staged_groups: list[list[int]] = []
        skip_blocks: dict[int, int] = {}
        for kernel_group_id, planned_ids in enumerate(plan.block_ids_by_group):
            shape_desc = cache_context.get_shape_desc(kernel_group_id)
            slots_per_object = cache_context.get_slots_per_chunk_in_sw(kernel_group_id)
            if slots_per_object % shape_desc.bs:
                raise ValueError("kernel group slots are not block aligned")
            blocks_per_object = slots_per_object // shape_desc.bs
            expected_blocks = (
                len(kernel_pointers.get(kernel_group_id, ())) * blocks_per_object
            )
            missing_prefix = expected_blocks - len(planned_ids)
            if missing_prefix < 0:
                raise ValueError("transfer plan has too many block IDs")
            if expected_blocks <= 0 or missing_prefix >= expected_blocks:
                raise ValueError("transfer plan has no executable blocks")
            staged_groups.append([0] * missing_prefix + list(planned_ids))
            skip_blocks[kernel_group_id] = missing_prefix
        return cache_context.stage_block_ids(staged_groups), skip_blocks

    def _launch(
        self,
        cache_context: BaseCacheContext,
        plan: TransferPlan,
        kernel_pointers: dict[int, list[int]],
        staged_block_ids: list[Any],
        skip_blocks: dict[int, int],
    ) -> None:
        native_ops = cast(
            _NativeOps,
            self._native_ops or importlib.import_module("lmcache.c_ops"),
        )
        direction = (
            native_ops.TransferDirection.D2H
            if plan.direction == "store"
            else native_ops.TransferDirection.H2D
        )
        manager = cache_context.kv_layer_groups_manager
        for object_group in manager.object_groups:
            for kernel_group_id in object_group.kernel_group_indices:
                pointers = kernel_pointers[kernel_group_id]
                native_ops.cxl_region_block_kv_transfer(
                    cache_context.get_kernel_group_kv_pointers(kernel_group_id),
                    pointers,
                    staged_block_ids[kernel_group_id],
                    cache_context.device,
                    cache_context.get_shape_desc(kernel_group_id),
                    direction,
                    cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                    cache_context.get_engine_kv_format(kernel_group_id),
                    skip_blocks[kernel_group_id],
                )
