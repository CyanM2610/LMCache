# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import (
    FetchDecision,
    RecomputeDecision,
    StorePlacementPlan,
    TargetSpec,
)
from lmcache.v1.multiprocess.cxl.directory import MultiResidencyDirectory
from lmcache.v1.multiprocess.cxl.observations import (
    PlacementSnapshot,
    RequestObservation,
    ResidencyObservation,
    TierObservation,
)
from lmcache.v1.multiprocess.cxl.placement import BoundLookup, LookupCoordinator
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle
from lmcache.v1.multiprocess.cxl.tickets import TicketManager


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def _ready() -> tuple[MultiResidencyDirectory, object]:
    manager = CXLRegionManager(
        RegionHandle("cxl", "/unused", 4096, 64, frozenset()),
        layout_id="packed_kv_v1",
        layout_fingerprint="a" * 64,
    )
    directory = MultiResidencyDirectory({"cxl": manager})
    reserved = directory.reserve_residency(
        _key(), TargetSpec("cxl", True, "test"), length=256, alignment=64
    )
    directory.mark_writing(reserved.residency_id)
    return directory, directory.publish(reserved.residency_id, None)


def test_bind_reserves_lease_and_queue_then_terminal_releases_once() -> None:
    directory, ready = _ready()
    manager = TicketManager(directory, ticket_ttl_ns=1_000)
    ticket = manager.bind_fetch(
        FetchDecision(ready.residency_id, 100, "fetch"), "request", 10
    )

    assert manager.queue_bytes("cxl") == 256
    assert manager.validate(ticket, 11).residency.residency_id == ready.residency_id
    manager.complete(ticket, "ok")
    manager.complete(ticket, "ok")
    assert manager.queue_bytes("cxl") == 0
    assert directory.get_residency(ready.residency_id).active_readers == 0


def test_policy_to_bind_race_rejects_evicted_residency() -> None:
    directory, ready = _ready()
    decision = FetchDecision(ready.residency_id, 100, "advisory")
    directory.begin_evict(ready.residency_id)
    directory.reclaim(ready.residency_id)

    with pytest.raises(RuntimeError, match="bind"):
        TicketManager(directory).bind_fetch(decision, "request", 10)


def test_expiry_releases_lease_and_queue_exactly_once() -> None:
    directory, ready = _ready()
    manager = TicketManager(directory, ticket_ttl_ns=10)
    ticket = manager.bind_fetch(
        FetchDecision(ready.residency_id, 1, "fetch"), "request", 5
    )

    assert manager.expire(14) == ()
    assert manager.expire(15) == (ticket,)
    assert manager.expire(16) == ()
    assert manager.queue_bytes("cxl") == 0
    with pytest.raises(RuntimeError, match="terminal"):
        manager.validate(ticket, 16)


def test_lookup_coordinator_replans_once_before_returning_positive_match() -> None:
    directory, ready = _ready()
    calls = 0

    class RacingPolicy:
        def plan_store(self, snapshot: PlacementSnapshot) -> StorePlacementPlan:
            return StorePlacementPlan(
                snapshot.request.object_key,
                (TargetSpec("cxl", True, "test"),),
                "test",
            )

        def decide_lookup(self, snapshot: PlacementSnapshot):
            nonlocal calls
            calls += 1
            if calls == 1:
                directory.begin_evict(ready.residency_id)
                directory.reclaim(ready.residency_id)
                return FetchDecision(ready.residency_id, 100, "stale")
            return RecomputeDecision("fresh snapshot has no source")

    def snapshot() -> PlacementSnapshot:
        residencies = ()
        if calls == 0:
            residencies = (
                ResidencyObservation(
                    ready.residency_id,
                    _key(),
                    "cxl",
                    "ready",
                    256,
                    ready.generation,
                    "a" * 64,
                    1,
                    0,
                    0,
                    False,
                ),
            )
        return PlacementSnapshot(
            1,
            RequestObservation("request", 0, _key(), 256, 256, None, 1_000, "a" * 64),
            residencies,
            (TierObservation("cxl", 4096, 256, 0, 1_000_000, 100),),
        )

    coordinator = LookupCoordinator(RacingPolicy(), TicketManager(directory))
    result = coordinator.decide_and_bind("request", snapshot, now_ns=1)

    assert isinstance(result, RecomputeDecision)
    assert calls == 2
    assert not isinstance(result, BoundLookup)
