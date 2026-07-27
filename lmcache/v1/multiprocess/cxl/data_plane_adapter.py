# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral transfer planning and the vLLM cache-context adapter."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Any, Protocol

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    downsample_block_ids,
    recalculate_blocks_to_skip,
)

# Local
from .contracts import (
    ExtentDescriptor,
    PackedLayoutSpec,
    TransferDirection,
    TransferPlan,
)
from .layout import layout_fingerprint


@dataclass(frozen=True)
class VLLMTransferRequest:
    """One-time vLLM metadata needed to construct a transfer plan."""

    op_id: str
    instance_id: int
    model_name: str
    token_count: int
    object_keys: tuple[ObjectKey, ...]
    block_ids_by_group: tuple[tuple[int, ...], ...]
    skip_first_n_tokens: int = 0
    payload_checksum_expected: str | None = None

    def __post_init__(self) -> None:
        if not self.op_id or not self.model_name:
            raise ValueError("operation and model identity must not be empty")
        if self.instance_id < 0 or self.token_count <= 0:
            raise ValueError("instance must be non-negative and tokens positive")
        if not self.object_keys or not self.block_ids_by_group:
            raise ValueError("object keys and block ID groups must not be empty")
        if any(not group for group in self.block_ids_by_group):
            raise ValueError("block ID groups must not be empty")
        if self.skip_first_n_tokens < 0:
            raise ValueError("skip_first_n_tokens must be non-negative")


class DataPlaneAdapter(Protocol):
    """Planning seam between engine metadata and shared-tier execution."""

    adapter_id: str
    plan_version: int

    def build_store_plan(
        self, request: VLLMTransferRequest, extent: ExtentDescriptor
    ) -> TransferPlan:
        """Build a validated HBM-to-extent plan.

        Args:
            request: Engine-neutral vLLM transfer metadata.
            extent: Current-generation destination extent.

        Returns:
            A serializable store plan.

        Raises:
            ValueError: If request or layout metadata is incompatible.
        """
        ...

    def build_retrieve_plan(
        self, request: VLLMTransferRequest, extent: ExtentDescriptor
    ) -> TransferPlan:
        """Build a validated extent-to-HBM plan.

        Args:
            request: Engine-neutral vLLM transfer metadata.
            extent: Current-generation source extent.

        Returns:
            A serializable retrieve plan.

        Raises:
            ValueError: If request or layout metadata is incompatible.
        """
        ...

    def validate_context(self, cache_context: Any, layout: PackedLayoutSpec) -> None:
        """Reject an engine context incompatible with packed bytes.

        Args:
            cache_context: Imported engine cache context.
            layout: Producer layout to validate.

        Raises:
            ValueError: If the cache context cannot interpret the layout.
        """
        ...


class VLLMDataPlaneAdapter:
    """Translate an LMCache vLLM cache context into versioned plans."""

    adapter_id = "vllm_paged_kv_v1"
    plan_version = 1
    layout_id = "packed_kv_v1"
    layout_version = 1

    def __init__(self, cache_context: Any) -> None:
        self._cache_context = cache_context

    def describe_layout(self, model_name: str, token_count: int) -> PackedLayoutSpec:
        """Describe the exact packed order exposed by the cache context.

        Args:
            model_name: Producer/consumer model identity.
            token_count: Logical tokens represented by each packed object.

        Returns:
            Canonical metadata in declared object/kernel-group order.
        """
        object_group_order: list[int] = []
        shapes: list[tuple[int, ...]] = []
        dtypes: list[str] = []
        engine_kv_formats: list[str] = []
        manager = self._cache_context.kv_layer_groups_manager
        for object_group_id, object_group in enumerate(manager.object_groups):
            for kernel_group_id in object_group.kernel_group_indices:
                shape, dtype = self._cache_context.get_kernel_group_shape_dtype(
                    token_count, kernel_group_id
                )
                engine_format = self._cache_context.get_engine_kv_format(
                    kernel_group_id
                )
                object_group_order.append(object_group_id)
                shapes.append(tuple(int(dimension) for dimension in shape))
                dtypes.append(str(dtype))
                engine_kv_formats.append(
                    getattr(engine_format, "name", str(engine_format))
                )
        return PackedLayoutSpec(
            layout_id=self.layout_id,
            layout_version=self.layout_version,
            model_name=model_name,
            token_count=token_count,
            object_group_order=tuple(object_group_order),
            shapes=tuple(shapes),
            dtypes=tuple(dtypes),
            engine_kv_formats=tuple(engine_kv_formats),
        )

    @staticmethod
    def fingerprint(layout: PackedLayoutSpec) -> str:
        """Return the canonical identity of a described layout.

        Args:
            layout: Complete packed-layout metadata.

        Returns:
            SHA-256 canonical layout fingerprint.
        """
        return layout_fingerprint(layout)

    def validate_context(self, cache_context: Any, layout: PackedLayoutSpec) -> None:
        """Reject a cache context with a different packed layout.

        Args:
            cache_context: Existing LMCache platform cache context.
            layout: Layout recorded by the extent producer.

        Raises:
            ValueError: If any layout dimension differs.
        """
        consumer = VLLMDataPlaneAdapter(cache_context).describe_layout(
            layout.model_name, layout.token_count
        )
        if layout_fingerprint(consumer) != layout_fingerprint(layout):
            raise ValueError("cache context is incompatible with packed layout")

    def split_chunks(
        self, request: VLLMTransferRequest
    ) -> tuple[VLLMTransferRequest, ...]:
        """Split a request into independently publishable packed chunks.

        Args:
            request: Complete chunk-aligned transfer request.

        Returns:
            Requests ordered by chunk, each carrying one object from every
            object group and the matching block IDs from every kernel group.

        Raises:
            ValueError: If object groups or block IDs do not form complete,
                equally sized chunks.
        """
        manager = self._cache_context.kv_layer_groups_manager
        keys_by_group: list[list[ObjectKey]] = [[] for _ in manager.object_groups]
        for object_key in request.object_keys:
            group_id = object_key.object_group_id
            if group_id < 0 or group_id >= len(keys_by_group):
                raise ValueError("ObjectKey references an unknown object group")
            keys_by_group[group_id].append(object_key)
        chunk_counts = {len(keys) for keys in keys_by_group}
        if len(chunk_counts) != 1 or not chunk_counts or 0 in chunk_counts:
            raise ValueError("every object group must contain the same chunks")
        num_chunks = next(iter(chunk_counts))

        block_slices: list[list[tuple[int, ...]]] = []
        if len(request.block_ids_by_group) != manager.num_kernel_groups:
            raise ValueError("block ID group count does not match cache context")
        for kernel_group_id, block_ids in enumerate(request.block_ids_by_group):
            blocks_per_chunk = self._cache_context.calculate_num_blocks(
                request.token_count, kernel_group_id
            )
            if len(block_ids) != num_chunks * blocks_per_chunk:
                raise ValueError("block IDs do not exactly cover packed chunks")
            block_slices.append(
                [
                    tuple(
                        block_ids[
                            chunk_id * blocks_per_chunk : (chunk_id + 1)
                            * blocks_per_chunk
                        ]
                    )
                    for chunk_id in range(num_chunks)
                ]
            )

        return tuple(
            replace(
                request,
                op_id=f"{request.op_id}:chunk:{chunk_id}",
                object_keys=tuple(group_keys[chunk_id] for group_keys in keys_by_group),
                block_ids_by_group=tuple(
                    group_slices[chunk_id] for group_slices in block_slices
                ),
                skip_first_n_tokens=0,
            )
            for chunk_id in range(num_chunks)
        )

    def packed_size_bytes(self, request: VLLMTransferRequest) -> int:
        """Return the exact packed extent bytes for a transfer request.

        Args:
            request: Request whose ObjectKeys declare packed object groups.

        Returns:
            Sum of all kernel-group payload bytes in declared object order.

        Raises:
            ValueError: If an ObjectKey references an unknown object group.
        """
        manager = self._cache_context.kv_layer_groups_manager
        total = 0
        for object_key in request.object_keys:
            group_id = object_key.object_group_id
            if group_id < 0 or group_id >= len(manager.object_groups):
                raise ValueError("ObjectKey references an unknown object group")
            object_group = manager.object_groups[group_id]
            for kernel_group_id in object_group.kernel_group_indices:
                shape, dtype = self._cache_context.get_kernel_group_shape_dtype(
                    request.token_count, kernel_group_id
                )
                elements = reduce(mul, (int(dimension) for dimension in shape), 1)
                total += elements * torch.empty((), dtype=dtype).element_size()
        if total <= 0:
            raise ValueError("packed request has no payload bytes")
        return total

    def build_store_plan(
        self, request: VLLMTransferRequest, extent: ExtentDescriptor
    ) -> TransferPlan:
        """Build a store plan preserving vLLM block ordering.

        Args:
            request: One-time vLLM request and block metadata.
            extent: Current-generation destination extent.

        Returns:
            A validated HBM-to-extent transfer plan.

        Raises:
            ValueError: If skip metadata or layout is incompatible.
        """
        if request.skip_first_n_tokens:
            raise ValueError("store plans do not support prefix skip")
        self._validate_request_layout(request, extent)
        block_ids = self._prepare_block_ids(request, retrieve=False)
        return self._build_plan(request, extent, "store", block_ids)

    def build_retrieve_plan(
        self, request: VLLMTransferRequest, extent: ExtentDescriptor
    ) -> TransferPlan:
        """Build a retrieve plan with request-local prefix skip applied.

        Args:
            request: One-time vLLM request and block metadata.
            extent: Current-generation source extent.

        Returns:
            A validated extent-to-HBM transfer plan.

        Raises:
            ValueError: If layout is incompatible or skip consumes a group.
        """
        self._validate_request_layout(request, extent)
        block_ids = self._prepare_block_ids(request, retrieve=True)
        return self._build_plan(request, extent, "retrieve", block_ids)

    def _prepare_block_ids(
        self, request: VLLMTransferRequest, *, retrieve: bool
    ) -> tuple[tuple[int, ...], ...]:
        if self._cache_context.lmcache_tokens_per_chunk != request.token_count:
            raise ValueError("request token count does not match cache context")
        raw_groups = [list(group) for group in request.block_ids_by_group]
        groups = downsample_block_ids(self._cache_context, raw_groups)
        if not retrieve:
            return tuple(tuple(group) for group in groups)

        manager = self._cache_context.kv_layer_groups_manager
        attn_desc = manager.get_attn_desc()
        object_group_by_kernel = {
            kernel_group_id: object_group_id
            for object_group_id, object_group in enumerate(manager.object_groups)
            for kernel_group_id in object_group.kernel_group_indices
        }
        planned: list[tuple[int, ...]] = []
        for kernel_group_id, (raw, downsampled) in enumerate(
            zip(request.block_ids_by_group, groups, strict=True)
        ):
            blocks_per_chunk = self._cache_context.calculate_num_blocks(
                request.token_count, kernel_group_id
            )
            tokens_per_window = min(
                request.token_count,
                manager.get_subchunk_sw_size_tokens(kernel_group_id),
            )
            blocks_per_window = self._cache_context.calculate_num_blocks(
                tokens_per_window, kernel_group_id
            )
            num_objects = len(raw) // blocks_per_chunk
            object_group_id = object_group_by_kernel[kernel_group_id]
            window_chunks = attn_desc.num_chunks_in_sw[object_group_id]
            object_skip = (
                0
                if window_chunks < 0
                else max(0, num_objects - window_chunks) * blocks_per_window
            )
            prefix_skip = recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                self._cache_context.calculate_num_blocks(
                    request.skip_first_n_tokens, kernel_group_id
                ),
            )
            skip_blocks = max(object_skip, prefix_skip)
            if skip_blocks >= len(downsampled):
                raise ValueError("prefix skip consumes every block in a group")
            planned.append(tuple(downsampled[skip_blocks:]))
        return tuple(planned)

    def _validate_request_layout(
        self, request: VLLMTransferRequest, extent: ExtentDescriptor
    ) -> None:
        manager = self._cache_context.kv_layer_groups_manager
        if len(request.block_ids_by_group) != manager.num_kernel_groups:
            raise ValueError("block ID group count does not match cache context")
        layout = self.describe_layout(request.model_name, request.token_count)
        if extent.layout_id != layout.layout_id or (
            extent.layout_fingerprint != layout_fingerprint(layout)
        ):
            raise ValueError("extent layout is incompatible with request")

    def _build_plan(
        self,
        request: VLLMTransferRequest,
        extent: ExtentDescriptor,
        direction: TransferDirection,
        block_ids_by_group: tuple[tuple[int, ...], ...],
    ) -> TransferPlan:
        return TransferPlan(
            plan_version=self.plan_version,
            op_id=request.op_id,
            direction=direction,
            instance_id=request.instance_id,
            object_keys=request.object_keys,
            block_ids_by_group=block_ids_by_group,
            extent=extent,
            payload_checksum_expected=request.payload_checksum_expected,
        )
