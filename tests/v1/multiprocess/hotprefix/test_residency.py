# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.hotprefix.admission import (
    AdmissionAction,
    HostAdmissionCandidate,
    HostAdmissionPolicy,
)
from lmcache.v1.multiprocess.hotprefix.residency import (
    HostResidencyDirectory,
    HostResidencyState,
)


def test_directory_reserves_publishes_and_deduplicates_residency() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    candidate = HostAdmissionCandidate(b"prefix-a", 100, 2, 3)

    reserved = directory.reserve(candidate, policy)
    ready = directory.publish(b"prefix-a")
    duplicate = directory.reserve(candidate, policy)

    assert reserved.action is AdmissionAction.ACCEPT
    assert ready.state is HostResidencyState.READY
    assert ready.generation == 1
    assert duplicate.action is AdmissionAction.DEDUP
    assert directory.used_bytes == 100


def test_directory_abort_releases_capacity_and_rejects_inflight_duplicate() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    candidate = HostAdmissionCandidate(b"prefix-a", 100, 2, 3)

    directory.reserve(candidate, policy)
    with pytest.raises(RuntimeError, match="already in progress"):
        directory.reserve(candidate, policy)
    directory.abort(b"prefix-a")

    assert directory.used_bytes == 0
    assert directory.snapshot() == ()


def test_generation_bound_read_lease_prevents_host_replacement() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    cold = HostAdmissionCandidate(b"prefix-cold", 100, 1, 1)
    hot = HostAdmissionCandidate(b"prefix-hot", 100, 5, 5)
    directory.reserve(cold, policy)
    directory.publish(cold.prefix_id)

    with pytest.raises(RuntimeError, match="generation changed"):
        directory.acquire(cold.prefix_id, 2, b"wrong-generation")
    lease = directory.acquire(cold.prefix_id, 1, b"ticket-a")
    rejected = directory.reserve(hot, policy)

    assert lease.generation == 1
    assert rejected.action is AdmissionAction.REJECT
    assert rejected.reason == "insufficient_reclaimable_capacity"
    with pytest.raises(RuntimeError, match="active read lease"):
        directory.evict(cold.prefix_id)

    directory.release(lease.ticket_id)
    accepted = directory.reserve(hot, policy)
    assert accepted.action is AdmissionAction.ACCEPT
    assert accepted.evict_prefixes == (cold.prefix_id,)


def test_aborted_replacement_restores_ready_victim() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    cold = HostAdmissionCandidate(b"prefix-cold", 100, 1, 1)
    hot = HostAdmissionCandidate(b"prefix-hot", 100, 5, 5)
    directory.reserve(cold, policy)
    original = directory.publish(cold.prefix_id)

    directory.reserve(hot, policy)
    assert directory.get(cold.prefix_id) is None
    directory.abort(hot.prefix_id)

    assert directory.get(cold.prefix_id) == original
    assert directory.used_bytes == 100


def test_expired_read_lease_no_longer_blocks_replacement() -> None:
    now = [0.0]
    directory = HostResidencyDirectory(
        capacity_bytes=100,
        lease_ttl_seconds=5.0,
        clock=lambda: now[0],
    )
    policy = HostAdmissionPolicy(frequency_threshold=1)
    cold = HostAdmissionCandidate(b"prefix-cold", 100, 1, 1)
    hot = HostAdmissionCandidate(b"prefix-hot", 100, 5, 5)
    directory.reserve(cold, policy)
    directory.publish(cold.prefix_id)
    directory.acquire(cold.prefix_id, 1, b"ticket")

    now[0] = 6.0
    accepted = directory.reserve(hot, policy)

    assert accepted.action is AdmissionAction.ACCEPT
    assert accepted.evict_prefixes == (cold.prefix_id,)
    assert directory.release(b"ticket", missing_ok=True) is None


def test_physical_miss_invalidates_only_the_reported_generation() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    candidate = HostAdmissionCandidate(b"prefix", 100, 2, 2)
    directory.reserve(candidate, policy, generation=7)
    directory.publish(candidate.prefix_id)

    assert directory.invalidate(candidate.prefix_id, 6) is False
    assert directory.get(candidate.prefix_id) is not None
    assert directory.invalidate(candidate.prefix_id, 7) is True
    assert directory.get(candidate.prefix_id) is None


def test_invalidation_tombstones_generation_until_readers_release() -> None:
    directory = HostResidencyDirectory(capacity_bytes=100)
    policy = HostAdmissionPolicy(frequency_threshold=1)
    candidate = HostAdmissionCandidate(b"prefix", 100, 2, 2)
    directory.reserve(candidate, policy, generation=7)
    directory.publish(candidate.prefix_id)
    lease = directory.acquire(candidate.prefix_id, 7, b"reader")

    assert directory.invalidate(candidate.prefix_id, 7) is True
    invalid = directory.get(candidate.prefix_id)
    assert invalid is not None
    assert invalid.state is HostResidencyState.INVALID

    directory.release(lease.ticket_id)
    assert directory.get(candidate.prefix_id) is None
