# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import TargetSpec
from lmcache.v1.multiprocess.cxl.directory import (
    MultiResidencyDirectory,
    ResidencyState,
)
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def _manager(tier: str, region_id: str) -> CXLRegionManager:
    return CXLRegionManager(
        RegionHandle(
            region_id=region_id,
            shm_name=f"/{region_id}",
            capacity=4096,
            alignment=64,
            capabilities=frozenset(),
        ),
        layout_id="packed_kv_v1",
        layout_fingerprint="a" * 64,
        tier=tier,
    )


def _directory() -> MultiResidencyDirectory:
    return MultiResidencyDirectory(
        {"dram": _manager("dram", "dram0"), "cxl": _manager("cxl", "cxl0")}
    )


def _publish(directory: MultiResidencyDirectory, tier: str):
    reserved = directory.reserve_residency(
        _key(), TargetSpec(tier, True, "test"), length=256, alignment=64
    )
    directory.mark_writing(reserved.residency_id)
    return directory.publish(reserved.residency_id, None)


def test_one_object_has_independent_ready_dram_and_cxl_residencies() -> None:
    directory = _directory()
    dram = _publish(directory, "dram")
    cxl = _publish(directory, "cxl")

    assert {item.tier for item in directory.list_residencies(_key())} == {
        "dram",
        "cxl",
    }
    assert dram.residency_id != cxl.residency_id
    assert dram.descriptor != cxl.descriptor
    lease = directory.acquire_read(
        dram.residency_id, dram.generation, ttl_ns=100, now_ns=1
    )
    assert directory.get_residency(dram.residency_id).active_readers == 1
    assert directory.get_residency(cxl.residency_id).active_readers == 0
    directory.release_read(lease.lease_id, now_ns=2)


def test_replicate_publish_evict_never_mutates_ready_source() -> None:
    directory = _directory()
    source = _publish(directory, "cxl")
    target = directory.reserve_residency(
        _key(), TargetSpec("dram", True, "replicate"), length=256, alignment=64
    )
    directory.mark_writing(target.residency_id)

    assert directory.get_residency(source.residency_id) == source
    directory.abort(target.residency_id, "copy failed")
    assert directory.get_residency(source.residency_id) == source

    target = directory.reserve_residency(
        _key(), TargetSpec("dram", True, "replicate"), length=256, alignment=64
    )
    directory.mark_writing(target.residency_id)
    ready_target = directory.publish(target.residency_id, None)
    directory.begin_evict(source.residency_id)
    directory.reclaim(source.residency_id)

    assert directory.list_residencies(_key()) == (ready_target,)
    assert not any(
        name in {"move", "change_tier"} for name in dir(MultiResidencyDirectory)
    )


def test_eviction_waits_for_exact_generation_lease() -> None:
    directory = _directory()
    ready = _publish(directory, "cxl")
    lease = directory.acquire_read(
        ready.residency_id, ready.generation, ttl_ns=100, now_ns=1
    )
    directory.begin_evict(ready.residency_id)

    with pytest.raises(RuntimeError, match="active readers"):
        directory.reclaim(ready.residency_id)
    with pytest.raises(KeyError, match="READY"):
        directory.acquire_read(
            ready.residency_id, ready.generation, ttl_ns=100, now_ns=2
        )
    directory.release_read(lease.lease_id, now_ns=3)
    directory.reclaim(ready.residency_id)
    assert directory.list_residencies(_key()) == ()


def test_acquire_rejects_stale_generation() -> None:
    directory = _directory()
    ready = _publish(directory, "dram")
    with pytest.raises(RuntimeError, match="generation"):
        directory.acquire_read(
            ready.residency_id, ready.generation + 1, ttl_ns=100, now_ns=1
        )
    assert ready.state == ResidencyState.READY
