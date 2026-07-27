# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import hashlib

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
from lmcache.v1.multiprocess.cxl.fallback import FallbackCoordinator
from lmcache.v1.multiprocess.cxl.observations import (
    RequestObservation,
    TierObservation,
)
from lmcache.v1.multiprocess.cxl.placement import (
    BoundLookup,
    LookupCoordinator,
    StoreCoordinator,
    TargetTransferCompletion,
    snapshot_from_directory,
)
from lmcache.v1.multiprocess.cxl.policies import CostAwarePolicy
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle
from lmcache.v1.multiprocess.cxl.tickets import TicketManager


pytestmark = pytest.mark.no_shared_allocator
PAYLOAD = bytes(range(256))
CHECKSUM = hashlib.sha256(PAYLOAD).hexdigest()


def _key() -> ObjectKey:
    return ObjectKey(b"gate-e", "model", 0)


def _directory() -> MultiResidencyDirectory:
    managers = {
        tier: CXLRegionManager(
            RegionHandle(tier, f"/{tier}", 4096, 64, frozenset()),
            layout_id="packed_kv_v1",
            layout_fingerprint="a" * 64,
            tier=tier,
        )
        for tier in ("dram", "cxl")
    }
    return MultiResidencyDirectory(managers)


@dataclass
class _Source:
    payload: bytes = PAYLOAD
    released: bool = False

    def release(self) -> None:
        self.released = True


class _StoreExecutor:
    def __init__(self, failing_tiers: frozenset[str] = frozenset()) -> None:
        self.failing_tiers = failing_tiers
        self.payloads: dict[str, bytes] = {}

    def transfer(self, target, residency, source):
        if target.tier in self.failing_tiers:
            return TargetTransferCompletion(target.tier, False, "injected")
        self.payloads[residency.residency_id] = source.payload
        return TargetTransferCompletion(target.tier, True, None)


def _store(
    directory: MultiResidencyDirectory,
    targets: tuple[TargetSpec, ...],
    failing_tiers: frozenset[str] = frozenset(),
):
    source = _Source()
    executor = _StoreExecutor(failing_tiers)
    result = StoreCoordinator(directory, executor).execute(
        StorePlacementPlan(_key(), targets, "scenario"),
        source,
        length=len(PAYLOAD),
        alignment=64,
    )
    assert source.released
    return result, executor.payloads


def _snapshot(
    directory: MultiResidencyDirectory,
    *,
    dram_queue: int,
    cxl_queue: int,
    recompute_ns: int,
):
    request = RequestObservation(
        "request",
        0,
        _key(),
        len(PAYLOAD),
        256,
        None,
        recompute_ns,
        "a" * 64,
    )
    tiers = (
        TierObservation("dram", 4096, 256, dram_queue, 1_000_000, 0),
        TierObservation("cxl", 4096, 256, cxl_queue, 500_000, 100),
    )
    return snapshot_from_directory(directory, request, tiers, 1)


def _both_ready():
    directory = _directory()
    result, payloads = _store(
        directory,
        (
            TargetSpec("dram", True, "hot replica"),
            TargetSpec("cxl", True, "shared replica"),
        ),
    )
    return directory, result, payloads


def test_gate_e_store_dram_required_cxl_optional_partial_success() -> None:
    directory = _directory()
    result, payloads = _store(
        directory,
        (
            TargetSpec("dram", True, "required hot"),
            TargetSpec("cxl", False, "optional shared"),
        ),
        frozenset({"cxl"}),
    )

    assert result.required_satisfied and result.partial_success
    assert [item.tier for item in result.successful_residencies] == ["dram"]
    assert hashlib.sha256(next(iter(payloads.values()))).hexdigest() == CHECKSUM


def test_gate_e_store_cxl_required() -> None:
    directory = _directory()
    result, payloads = _store(directory, (TargetSpec("cxl", True, "required shared"),))

    assert result.required_satisfied and not result.partial_success
    assert result.successful_residencies[0].tier == "cxl"
    assert hashlib.sha256(next(iter(payloads.values()))).hexdigest() == CHECKSUM


@pytest.mark.parametrize(
    ("dram_queue", "cxl_queue", "expected_tier"),
    [(0, 0, "dram"), (2_000, 0, "cxl")],
)
def test_gate_e_cost_policy_selects_dram_or_cxl_with_bound_ticket(
    dram_queue: int, cxl_queue: int, expected_tier: str
) -> None:
    directory, _, _ = _both_ready()
    tickets = TicketManager(directory)
    snapshot = _snapshot(
        directory,
        dram_queue=dram_queue,
        cxl_queue=cxl_queue,
        recompute_ns=1_000_000,
    )
    result = LookupCoordinator(
        CostAwarePolicy(cuda_bandwidth_bytes_per_s=1_000_000), tickets
    ).decide_and_bind("request", lambda: snapshot, now_ns=1)

    assert isinstance(result, BoundLookup)
    bound = tickets.validate(result.ticket, 2)
    assert bound.residency.tier == expected_tier
    tickets.complete(result.ticket, "ok")


def test_gate_e_cost_policy_selects_recompute() -> None:
    directory, _, _ = _both_ready()
    snapshot = _snapshot(
        directory,
        dram_queue=2_000,
        cxl_queue=2_000,
        recompute_ns=1,
    )
    result = LookupCoordinator(
        CostAwarePolicy(cuda_bandwidth_bytes_per_s=1_000_000),
        TicketManager(directory),
    ).decide_and_bind("request", lambda: snapshot, now_ns=1)

    assert isinstance(result, RecomputeDecision)


@pytest.mark.parametrize("dram_succeeds", [True, False])
def test_gate_e_one_fallback_or_final_recompute_has_no_partial_execution(
    dram_succeeds: bool,
) -> None:
    directory, result, payloads = _both_ready()
    ready = {item.tier: item for item in result.successful_residencies}
    tickets = TicketManager(directory)
    initial = tickets.bind_fetch(
        FetchDecision(ready["cxl"].residency_id, 10, "initial CXL"),
        "request",
        1,
    )
    destination = bytearray(b"stale" * 64)
    invalidated: list[int] = []
    attention_calls = 0

    def transfer(ticket, full_overwrite: bool) -> None:
        assert full_overwrite
        tier = tickets.validate(ticket, 2).residency.tier
        destination[:] = b"partial"
        if tier == "cxl" or not dram_succeeds:
            raise RuntimeError(f"{tier} injected failure")
        destination[:] = payloads[ready[tier].residency_id]

    fallback = FallbackCoordinator(tickets, invalidate=invalidated.extend).execute(
        initial,
        transfer=transfer,
        choose_alternate=lambda excluded, now: FetchDecision(
            ready["dram"].residency_id, 10, "alternate DRAM"
        ),
        destination_block_ids=(7, 8),
        deadline_ns=1_000,
        now_ns=lambda: 2,
    )

    if dram_succeeds:
        assert fallback.status == "ok" and fallback.attempts == 2
        assert hashlib.sha256(destination).hexdigest() == CHECKSUM
        attention_calls += 1
        assert invalidated == [] and attention_calls == 1
    else:
        assert fallback.status == "recompute" and fallback.attempts == 2
        assert invalidated == [7, 8]
        assert attention_calls == 0
