# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import (
    FetchDecision,
    RecomputeDecision,
    TargetSpec,
)
from lmcache.v1.multiprocess.cxl.directory import MultiResidencyDirectory
from lmcache.v1.multiprocess.cxl.fallback import (
    FallbackCoordinator,
    LoadCancelled,
)
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle
from lmcache.v1.multiprocess.cxl.tickets import TicketManager


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def _setup() -> tuple[TicketManager, dict[str, object]]:
    managers = {
        tier: CXLRegionManager(
            RegionHandle(tier, f"/{tier}", 4096, 64, frozenset()),
            layout_id="packed_kv_v1",
            layout_fingerprint="a" * 64,
            tier=tier,
        )
        for tier in ("dram", "cxl")
    }
    directory = MultiResidencyDirectory(managers)
    ready = {}
    for tier in ("dram", "cxl"):
        item = directory.reserve_residency(
            _key(), TargetSpec(tier, True, "test"), length=256, alignment=64
        )
        directory.mark_writing(item.residency_id)
        ready[tier] = directory.publish(item.residency_id, None)
    return TicketManager(directory), ready


def _run(
    first_tier: str,
    outcomes: dict[str, object],
    alternate: Callable[[frozenset[str], int], object] | None = None,
):
    tickets, ready = _setup()
    first = tickets.bind_fetch(
        FetchDecision(ready[first_tier].residency_id, 10, "first"),
        "request",
        1,
    )
    writes = []
    invalidated = []

    def transfer(ticket: object, overwrite: bool) -> None:
        bound = tickets.validate(ticket, 2)
        tier = bound.residency.tier
        writes.append((tier, overwrite))
        outcome = outcomes[tier]
        if isinstance(outcome, BaseException):
            raise outcome
        if not outcome:
            raise RuntimeError(f"{tier} failed")

    def choose(excluded: frozenset[str], now_ns: int):
        if alternate is not None:
            return alternate(excluded, now_ns)
        other = "dram" if first_tier == "cxl" else "cxl"
        return FetchDecision(ready[other].residency_id, 10, "alternate")

    result = FallbackCoordinator(tickets, invalidate=invalidated.extend).execute(
        first,
        transfer=transfer,
        choose_alternate=choose,
        destination_block_ids=(3, 4),
        deadline_ns=1_000,
        now_ns=lambda: 2,
    )
    return result, writes, invalidated


@pytest.mark.parametrize("first", ["cxl", "dram"])
def test_exactly_one_alternate_residency_fallback(first: str) -> None:
    other = "dram" if first == "cxl" else "cxl"
    result, writes, invalidated = _run(first, {first: False, other: True})

    assert result.status == "ok"
    assert result.attempts == 2
    assert result.fallback_from == first
    assert result.fallback_to == other
    assert writes == [(first, True), (other, True)]
    assert invalidated == []


def test_second_failure_delegates_to_recompute_and_invalidates_blocks() -> None:
    result, writes, invalidated = _run("cxl", {"cxl": False, "dram": False})

    assert result.status == "recompute"
    assert result.attempts == 2
    assert invalidated == [3, 4]
    assert len(writes) == 2


def test_no_or_deadline_infeasible_alternate_does_not_retry() -> None:
    result, writes, invalidated = _run(
        "cxl",
        {"cxl": False, "dram": True},
        alternate=lambda excluded, now: RecomputeDecision("no alternate"),
    )
    assert result.status == "recompute"
    assert result.attempts == 1
    assert invalidated == [3, 4]
    assert len(writes) == 1


def test_cancellation_never_falls_back_or_exposes_partial_hbm() -> None:
    result, writes, invalidated = _run(
        "cxl", {"cxl": LoadCancelled("cancelled"), "dram": True}
    )
    assert result.status == "cancelled"
    assert result.attempts == 1
    assert invalidated == [3, 4]
    assert writes == [("cxl", True)]
