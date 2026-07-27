# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.contracts import ExtentDescriptor
from lmcache.v1.multiprocess.cxl.data_plane_adapter import (
    VLLMDataPlaneAdapter,
    VLLMTransferRequest,
)


pytestmark = pytest.mark.no_shared_allocator


class _Format(Enum):
    FIRST = 1
    SECOND = 2


@dataclass
class _ObjectGroup:
    kernel_group_indices: tuple[int, ...]


class _GroupsManager:
    object_groups = [_ObjectGroup((0,)), _ObjectGroup((1,))]
    num_kernel_groups = 2

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        del kernel_group_idx
        return 64

    def get_attn_desc(self) -> SimpleNamespace:
        return SimpleNamespace(num_chunks_in_sw=[-1, -1])


class _CacheContext:
    kv_layer_groups_manager = _GroupsManager()
    lmcache_tokens_per_chunk = 64

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        layers = 28 if kernel_group_idx == 0 else 4
        return (2, layers, num_tokens, 128), torch.bfloat16

    def get_engine_kv_format(self, kernel_group_idx: int) -> _Format:
        return (_Format.FIRST, _Format.SECOND)[kernel_group_idx]

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        del kernel_group_idx
        return (num_tokens + 15) // 16


def _keys() -> tuple[ObjectKey, ...]:
    return (
        ObjectKey(b"first", "Qwen2.5-7B-Instruct", 0, object_group_id=0),
        ObjectKey(b"second", "Qwen2.5-7B-Instruct", 0, object_group_id=1),
    )


def _request(**overrides: object) -> VLLMTransferRequest:
    values = {
        "op_id": "op-1",
        "instance_id": 7,
        "model_name": "Qwen2.5-7B-Instruct",
        "token_count": 64,
        "object_keys": _keys(),
        "block_ids_by_group": ((9, 3, 7, 1), (22, 19, 5, 8)),
        "skip_first_n_tokens": 0,
        "payload_checksum_expected": None,
    }
    values.update(overrides)
    return VLLMTransferRequest(**values)  # type: ignore[arg-type]


def _extent(adapter: VLLMDataPlaneAdapter) -> ExtentDescriptor:
    layout = adapter.describe_layout("Qwen2.5-7B-Instruct", 64)
    return ExtentDescriptor(
        region_id="proxy0",
        offset=0,
        length=4096,
        generation=1,
        tier="cxl",
        layout_id=layout.layout_id,
        layout_fingerprint=adapter.fingerprint(layout),
    )


def test_store_plan_preserves_object_and_block_order() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())
    plan = adapter.build_store_plan(_request(), _extent(adapter))

    assert plan.direction == "store"
    assert plan.instance_id == 7
    assert plan.object_keys == _keys()
    assert plan.block_ids_by_group == ((9, 3, 7, 1), (22, 19, 5, 8))


def test_split_chunks_preserves_object_groups_and_exact_packed_size() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())
    first_chunk = _keys()
    second_chunk = (
        ObjectKey(b"first-2", "Qwen2.5-7B-Instruct", 0, object_group_id=0),
        ObjectKey(b"second-2", "Qwen2.5-7B-Instruct", 0, object_group_id=1),
    )
    request = _request(
        object_keys=(
            first_chunk[0],
            second_chunk[0],
            first_chunk[1],
            second_chunk[1],
        ),
        block_ids_by_group=(
            (9, 3, 7, 1, 6, 4, 2, 0),
            (22, 19, 5, 8, 17, 13, 11, 10),
        ),
    )

    chunks = adapter.split_chunks(request)

    assert [chunk.object_keys for chunk in chunks] == [first_chunk, second_chunk]
    assert chunks[0].block_ids_by_group == ((9, 3, 7, 1), (22, 19, 5, 8))
    assert chunks[1].block_ids_by_group == ((6, 4, 2, 0), (17, 13, 11, 10))
    assert [adapter.packed_size_bytes(chunk) for chunk in chunks] == [
        1_048_576,
        1_048_576,
    ]


def test_retrieve_plan_applies_prefix_skip_per_kernel_group() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())
    request = _request(skip_first_n_tokens=16)

    plan = adapter.build_retrieve_plan(request, _extent(adapter))

    assert plan.direction == "retrieve"
    assert plan.block_ids_by_group == ((3, 7, 1), (19, 5, 8))


def test_retrieve_plan_reuses_sliding_window_block_downsampling() -> None:
    class _SlidingWindowGroupsManager(_GroupsManager):
        def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
            return 64 if kernel_group_idx == 0 else 32

    class _SlidingWindowContext(_CacheContext):
        kv_layer_groups_manager = _SlidingWindowGroupsManager()

    adapter = VLLMDataPlaneAdapter(_SlidingWindowContext())

    plan = adapter.build_retrieve_plan(_request(), _extent(adapter))

    assert plan.block_ids_by_group == ((9, 3, 7, 1), (5, 8))


def test_layout_records_declared_object_group_and_engine_format_order() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())

    layout = adapter.describe_layout("Qwen2.5-7B-Instruct", 64)

    assert layout.object_group_order == (0, 1)
    assert layout.shapes == ((2, 28, 64, 128), (2, 4, 64, 128))
    assert layout.engine_kv_formats == ("FIRST", "SECOND")


def test_layout_mismatch_fails_before_a_plan_is_returned() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())
    extent = _extent(adapter)
    incompatible = ExtentDescriptor(
        region_id=extent.region_id,
        offset=extent.offset,
        length=extent.length,
        generation=extent.generation,
        tier="cxl",
        layout_id=extent.layout_id,
        layout_fingerprint="f" * 64,
    )

    with pytest.raises(ValueError, match="layout"):
        adapter.build_retrieve_plan(_request(), incompatible)


def test_skip_cannot_consume_every_block_in_a_retrieve_group() -> None:
    adapter = VLLMDataPlaneAdapter(_CacheContext())

    with pytest.raises(ValueError, match="skip"):
        adapter.build_retrieve_plan(_request(skip_first_n_tokens=64), _extent(adapter))
