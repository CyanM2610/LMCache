# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import FrozenInstanceError
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle
from lmcache.v1.multiprocess.cxl.residency import (
    ResidencyState,
    SingleResidencyDirectory,
)


pytestmark = pytest.mark.no_shared_allocator


def _key(chunk: bytes = b"chunk") -> ObjectKey:
    return ObjectKey(chunk, "Qwen2.5-7B-Instruct", 0)


def _directory() -> SingleResidencyDirectory:
    handle = RegionHandle(
        region_id="proxy0",
        shm_name="/unused",
        capacity=4096,
        alignment=64,
        capabilities=frozenset({"cuda_host_register_v1"}),
    )
    manager = CXLRegionManager(
        handle,
        layout_id="packed_kv_v1",
        layout_fingerprint="a" * 64,
    )
    return SingleResidencyDirectory(manager)


def _publish(
    directory: SingleResidencyDirectory, key: ObjectKey | None = None
) -> tuple[ObjectKey, str]:
    object_key = key or _key()
    reserved = directory.reserve_store(object_key, length=256, alignment=64)
    writing = directory.mark_writing(reserved.residency_id)
    ready = directory.publish(reserved.residency_id)
    assert writing.state == ResidencyState.WRITING
    assert ready.state == ResidencyState.READY
    return object_key, ready.residency_id


def test_single_residency_store_follows_exact_publication_lifecycle() -> None:
    directory = _directory()
    key = _key()

    reserved = directory.reserve_store(key, length=256, alignment=64)
    assert reserved.state == ResidencyState.RESERVED
    assert directory.lookup_ready(key) is None

    writing = directory.mark_writing(reserved.residency_id)
    assert writing.descriptor is not None
    ready = directory.publish(reserved.residency_id)

    assert ready.state == ResidencyState.READY
    assert directory.lookup_ready(key) == ready
    with pytest.raises(FrozenInstanceError):
        ready.state = ResidencyState.EVICTING  # type: ignore[misc]


def test_single_residency_rejects_skipped_reversed_and_duplicate_transitions() -> None:
    directory = _directory()
    key = _key()
    reserved = directory.reserve_store(key, length=256, alignment=64)

    with pytest.raises(RuntimeError):
        directory.publish(reserved.residency_id)
    with pytest.raises(RuntimeError, match="already has"):
        directory.reserve_store(key, length=256, alignment=64)

    directory.mark_writing(reserved.residency_id)
    directory.abort(reserved.residency_id, "copy failed")
    assert directory.lookup_ready(key) is None


def test_ready_residency_supports_multiple_generation_bound_readers() -> None:
    directory = _directory()
    key, residency_id = _publish(directory)

    first = directory.acquire_read(key, ttl_ns=100, now_ns=10)
    second = directory.acquire_read(key, ttl_ns=200, now_ns=10)

    assert first.residency_id == second.residency_id == residency_id
    assert first.generation == second.generation
    assert first.lease_id != second.lease_id
    assert first.expires_at_ns == 110


def test_eviction_blocks_new_readers_and_waits_for_existing_lease() -> None:
    directory = _directory()
    key, _ = _publish(directory)
    lease = directory.acquire_read(key, ttl_ns=100, now_ns=10)

    directory.evict(key)

    assert directory.lookup_ready(key) is None
    with pytest.raises(KeyError, match="READY"):
        directory.acquire_read(key, ttl_ns=100, now_ns=11)

    directory.release_read(lease.lease_id, now_ns=12)
    replacement = directory.reserve_store(key, length=256, alignment=64)
    assert replacement.generation == lease.generation + 1


def test_expired_and_duplicate_lease_release_are_idempotent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = _directory()
    key, _ = _publish(directory)
    lease = directory.acquire_read(key, ttl_ns=10, now_ns=5)
    directory.evict(key)

    with caplog.at_level(logging.INFO):
        directory.release_read(lease.lease_id, now_ns=20)
    directory.release_read(lease.lease_id, now_ns=21)

    replacement = directory.reserve_store(key, length=256, alignment=64)
    assert replacement.generation > lease.generation
    assert "expired read lease" in caplog.text


def test_release_rejects_a_lease_id_bound_to_another_generation() -> None:
    directory = _directory()
    key, _ = _publish(directory)

    with pytest.raises(KeyError):
        directory.release_read("not-a-lease", now_ns=0)
