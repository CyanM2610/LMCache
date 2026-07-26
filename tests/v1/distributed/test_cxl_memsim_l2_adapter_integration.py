# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration test for the real CXLMemSim bulk transport."""

# Standard
from pathlib import Path
import os
import select
import subprocess
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.cxl_memsim_l2_adapter import (
    CxlMemSimL2Adapter,
    CxlMemSimL2AdapterConfig,
)
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.platform import consume_fd

_SERVER = os.environ.get("CXLMEMSIM_SERVER")
_CLIENT_LIBRARY = os.environ.get("CXLMEMSIM_CLIENT_LIBRARY")

pytestmark = pytest.mark.skipif(
    not _SERVER or not _CLIENT_LIBRARY,
    reason="set CXLMEMSIM_SERVER and CXLMEMSIM_CLIENT_LIBRARY",
)


def _memory_obj(fill_value: float) -> MemoryObj:
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate(
        [torch.Size([2048])],
        [torch.bfloat16],
        fmt=MemoryFormat.KV_2LTD,
    )
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(fill_value)
    return obj


def _wait_for_fd(fd: int) -> None:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(10_000)
    consume_fd(fd)


def _wait_for_server_lock(
    control_name: str,
    server: subprocess.Popen[str],
    timeout: float = 10.0,
) -> None:
    lock_path = Path("/dev/shm") / f"{control_name.removeprefix('/')}.lock"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if lock_path.exists():
            return
        if server.poll() is not None:
            pytest.fail(f"CXLMemSim server exited with code {server.returncode}")
        time.sleep(0.01)
    pytest.fail(f"CXLMemSim server lock did not appear: {lock_path}")


def test_cxl_memsim_adapter_round_trips_through_real_bulk_server(
    tmp_path: Path,
) -> None:
    assert _SERVER is not None
    assert _CLIENT_LIBRARY is not None
    control_name = f"/lmcache_cxl_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    server_log = tmp_path / "cxlmemsim-server.log"
    with server_log.open("w", encoding="utf-8") as log_file:
        server = subprocess.Popen(
            [
                _SERVER,
                "--comm-mode=bulk-shm",
                f"--bulk-shm-name={control_name}",
                "--capacity=16",
                "--default_latency=100",
                "--bulk-read-bandwidth=25",
                "--bulk-write-bandwidth=25",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_server_lock(control_name, server)

        adapter = None
        source = _memory_obj(7)
        target = _memory_obj(0)
        key = ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(1),
            model_name="integration-model",
            kv_rank=0,
        )
        try:
            config = CxlMemSimL2AdapterConfig(
                client_library=_CLIENT_LIBRARY,
                control_name=control_name,
                slot_bytes=4096,
                capacity_bytes=16384,
                timeout_ms=10_000,
                num_store_workers=1,
                num_lookup_workers=1,
                num_load_workers=1,
            )
            adapter = CxlMemSimL2Adapter(config)

            store_task = adapter.submit_store_task([key], [source])
            _wait_for_fd(adapter.get_store_event_fd())
            store_result = adapter.pop_completed_store_tasks()[store_task]
            assert store_result.is_successful()
            assert store_result.bytes_transferred() == 4096

            lookup_task = adapter.submit_lookup_and_lock_task(
                [key],
                MemoryLayoutDesc(shapes=[torch.Size([2048])], dtypes=[torch.bfloat16]),
            )
            _wait_for_fd(adapter.get_lookup_and_lock_event_fd())
            lookup_result = adapter.query_lookup_and_lock_result(lookup_task)
            assert lookup_result is not None
            assert lookup_result.test(0)

            load_task = adapter.submit_load_task([key], [target])
            _wait_for_fd(adapter.get_load_event_fd())
            load_result = adapter.query_load_result(load_task)
            assert load_result is not None
            assert load_result.test(0)
            assert target.tensor is not None
            assert torch.equal(target.tensor, source.tensor)
            adapter.submit_unlock([key])

            transport = adapter.report_status()["transport"]
            assert transport["write_requests"] == 1
            assert transport["read_requests"] == 1
            assert transport["write_bytes"] == 4096
            assert transport["read_bytes"] == 4096
            assert transport["write_cachelines"] == 64
            assert transport["read_cachelines"] == 64
        except Exception:
            if server.poll() is not None:
                pytest.fail(server_log.read_text(encoding="utf-8"))
            raise
        finally:
            if adapter is not None:
                adapter.close()
            source.ref_count_down()
            target.ref_count_down()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
