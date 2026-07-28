# SPDX-License-Identifier: Apache-2.0
"""Real CXLMemSim modeled service through the shared-tier lifecycle."""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
import os
import subprocess
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.data_plane_adapter import VLLMTransferRequest
from lmcache.v1.multiprocess.modules.cxl_shared_tier import (
    CXLSharedTierModule,
    InMemoryCXLOperationSink,
)


pytestmark = pytest.mark.no_shared_allocator


class _Registration:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def device_address(self, offset: int, length: int) -> int:
        assert offset >= 0 and length > 0
        return 0x100000 + offset

    def close(self) -> None:
        pass


class _NativeOps:
    class TransferDirection:
        H2D = "h2d"
        D2H = "d2h"

    def CudaRegionRegistration(
        self, shm_name: str, capacity: int, data_offset: int
    ) -> _Registration:
        assert shm_name == "/cxlmemsim_shared"
        assert data_offset == 4096
        return _Registration(capacity)

    def cxl_region_block_kv_transfer(self, *args: object) -> None:
        del args


class _Runtime:
    def execute(self, cache_context: object, operation) -> int:
        del cache_context
        operation()
        return 50


class _ObjectGroup:
    kernel_group_indices = (0,)


class _GroupsManager:
    object_groups = (_ObjectGroup(),)
    num_kernel_groups = 1
    num_object_groups = 1

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        assert kernel_group_idx == 0
        return 16

    def get_attn_desc(self) -> SimpleNamespace:
        return SimpleNamespace(num_chunks_in_sw=[-1])


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
        assert num_tokens == 16 and kernel_group_idx == 0
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


def _build_directory() -> Path:
    configured = os.environ.get("LMCACHE_CXLMEMSIM_BUILD_DIR")
    if configured:
        build = Path(configured)
    else:
        checkout = Path(__file__).resolve().parents[4].parent / "CXLMemSim"
        build = checkout / "build-vllm"
    required = (
        build / "cxlmemsim_server",
        build / "libcxlmemsim_modeled_client.so",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("CXLMemSim modeled-access build is unavailable")
    return build


@contextmanager
def _server(build: Path) -> Iterator[str]:
    suffix = uuid.uuid4().hex
    bulk_name = f"/lmcache_gate_d_bulk_{suffix}"
    modeled_name = f"/lmcache_gate_d_modeled_{suffix}"
    process = subprocess.Popen(
        [
            str(build / "cxlmemsim_server"),
            "--comm-mode=bulk-shm",
            f"--bulk-shm-name={bulk_name}",
            "--enable-gpu-direct-modeled-access=true",
            f"--modeled-access-shm-name={modeled_name}",
            "--capacity=1",
            "--default_latency=100",
            "--bulk-read-bandwidth=25",
            "--bulk-write-bandwidth=25",
        ],
        cwd=build.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 5
        control_path = Path("/dev/shm") / modeled_name[1:]
        while not control_path.exists():
            if process.poll() is not None:
                raise RuntimeError("CXLMemSim server exited before publication")
            if time.monotonic() >= deadline:
                raise TimeoutError("CXLMemSim modeled service did not start")
            time.sleep(0.01)
        yield modeled_name
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _request(op_id: str, key: ObjectKey) -> VLLMTransferRequest:
    return VLLMTransferRequest(
        op_id=op_id,
        instance_id=7,
        model_name="Qwen2.5-7B-Instruct",
        token_count=16,
        object_keys=(key,),
        block_ids_by_group=((1,),),
    )


def test_real_server_models_shared_tier_store_and_retrieve() -> None:
    build = _build_directory()
    sink = InMemoryCXLOperationSink()
    with _server(build) as modeled_name:
        module = CXLSharedTierModule.open(
            CXLSharedTierConfig(
                enabled=True,
                provider="cxlmemsim_shm",
                shm_name="/cxlmemsim_shared",
                capacity_bytes=1_044_480,
                alignment_bytes=64,
                model_mode="cxlmemsim",
                model_control_name=modeled_name,
                model_client_library=str(build / "libcxlmemsim_modeled_client.so"),
                model_timeout_ms=2_000,
            ),
            native_ops=_NativeOps(),
            runtime=_Runtime(),
            operation_sink=sink,
        )
        try:
            module.register_engine(7, _CacheContext(), "Qwen2.5-7B-Instruct")
            key = ObjectKey(b"real-modeled", "Qwen2.5-7B-Instruct", 0)

            stored = module.store(_request("real-modeled-store", key))
            retrieved = module.retrieve(_request("real-modeled-retrieve", key))

            assert stored.status == "ok"
            assert retrieved.status == "ok"
            assert stored.completions[0].modeled_status == "ok"
            assert retrieved.completions[0].modeled_status == "ok"
            assert all(
                completion.effective_complete_ns
                == max(
                    completion.cuda_complete_ns or 0,
                    completion.modeled_complete_ns or 0,
                )
                for completion in stored.completions + retrieved.completions
            )
            terminals = [event for event in sink.snapshot() if event.terminal]
            assert [event.direction for event in terminals] == ["store", "retrieve"]
            assert all(event.modeled_service_ns is not None for event in terminals)
        finally:
            module.close()
