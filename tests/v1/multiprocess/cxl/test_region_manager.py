# SPDX-License-Identifier: Apache-2.0

# Standard
from multiprocessing import shared_memory
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import (
    REGION_HEADER_SIZE,
    PosixShmRegionProvider,
    RegionHandle,
    pack_region_header,
)


pytestmark = pytest.mark.no_shared_allocator
FINGERPRINT = "a" * 64


def _handle(capacity: int = 1024) -> RegionHandle:
    return RegionHandle(
        region_id="proxy0",
        shm_name="/unused",
        capacity=capacity,
        alignment=64,
        capabilities=frozenset({"cuda_host_register_v1"}),
    )


def _manager(capacity: int = 1024) -> CXLRegionManager:
    return CXLRegionManager(
        _handle(capacity),
        layout_id="packed_kv_v1",
        layout_fingerprint=FINGERPRINT,
    )


def test_region_manager_allocates_aligned_non_overlapping_extents() -> None:
    manager = _manager()

    first = manager.reserve(100, 64)
    second = manager.reserve(200, 128)

    assert first.offset == 0
    assert second.offset == 128
    assert first.offset + first.length <= second.offset


def test_region_manager_uses_best_fit_and_reclaims_aborted_reservation() -> None:
    manager = _manager()
    first = manager.reserve(128, 64)
    middle = manager.reserve(256, 64)
    manager.reserve(128, 64)
    manager.abort(middle.reservation_id, "cancelled")

    replacement = manager.reserve(192, 64)

    assert replacement.offset == 128
    assert replacement.generation == middle.generation + 1
    manager.abort(first.reservation_id, "cleanup")


def test_region_manager_reports_capacity_exhaustion_without_overlap() -> None:
    manager = _manager(256)
    manager.reserve(192, 64)

    with pytest.raises(MemoryError, match="capacity"):
        manager.reserve(128, 64)


def test_region_manager_enforces_extent_lifecycle_and_generation() -> None:
    manager = _manager()
    reservation = manager.reserve(128, 64)

    with pytest.raises(RuntimeError, match="WRITING"):
        manager.publish(reservation.reservation_id)

    writing = manager.begin_write(reservation.reservation_id)
    ready = manager.publish(reservation.reservation_id)
    assert ready == writing

    manager.begin_evict(ready)
    manager.reclaim(ready)
    replacement = manager.reserve(128, 64)
    assert replacement.offset == ready.offset
    assert replacement.generation == ready.generation + 1

    with pytest.raises(ValueError, match="stale"):
        manager.begin_evict(ready)


def test_region_manager_rejects_invalid_alignment_and_transition_order() -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="power of two"):
        manager.reserve(64, 24)

    reservation = manager.reserve(64, 64)
    manager.begin_write(reservation.reservation_id)
    with pytest.raises(RuntimeError):
        manager.begin_write(reservation.reservation_id)


def test_posix_provider_validates_header_and_returns_stable_handle() -> None:
    name = f"beluga-test-{uuid.uuid4().hex}"
    capacity = 4096
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    try:
        header = pack_region_header(capacity, 4096)
        shm.buf[: len(header)] = header
        provider = PosixShmRegionProvider(
            region_id="proxy0",
            shm_name=f"/{name}",
            expected_capacity=capacity,
        )

        first = provider.provision()
        second = provider.provision()

        assert first is second
        assert first.capacity == capacity
        assert first.capabilities == frozenset({"cuda_host_register_v1"})
        provider.close()
    finally:
        shm.close()
        shm.unlink()


def test_posix_provider_fails_closed_on_invalid_header() -> None:
    name = f"beluga-test-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + 64
    )
    try:
        shm.buf[:8] = b"NOTCXL!!"
        provider = PosixShmRegionProvider(
            region_id="proxy0", shm_name=f"/{name}", expected_capacity=64
        )

        with pytest.raises(RuntimeError, match="header"):
            provider.provision()
    finally:
        shm.close()
        shm.unlink()
