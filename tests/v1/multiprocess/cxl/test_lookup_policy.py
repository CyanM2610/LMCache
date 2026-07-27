# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import FetchDecision, RecomputeDecision
from lmcache.v1.multiprocess.cxl.observations import (
    PlacementSnapshot,
    RequestObservation,
    ResidencyObservation,
    TierObservation,
)
from lmcache.v1.multiprocess.cxl.policies import (
    AlwaysFetchPolicy,
    AlwaysRecomputePolicy,
    CostAwarePolicy,
    FixedL1FirstPolicy,
)


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def _snapshot(
    *,
    dram_queue: int = 0,
    cxl_queue: int = 0,
    recompute_ns: int = 20_000_000,
    deadline_ns: int | None = None,
    dram_state: str = "ready",
    cxl_layout: str = "a" * 64,
) -> PlacementSnapshot:
    residencies = (
        ResidencyObservation(
            "dram-r",
            _key(),
            "dram",
            dram_state,
            1_000,
            1,
            "a" * 64,
            1,
            1,
            0,
            False,
        ),
        ResidencyObservation(
            "cxl-r",
            _key(),
            "cxl",
            "ready",
            1_000,
            1,
            cxl_layout,
            1,
            1,
            0,
            False,
        ),
    )
    return PlacementSnapshot(
        100,
        RequestObservation(
            "request",
            1,
            _key(),
            1_000,
            256,
            deadline_ns,
            recompute_ns,
            "a" * 64,
        ),
        residencies,
        (
            TierObservation("dram", 1_000_000, 1_000, dram_queue, 1_000_000, 10),
            TierObservation("cxl", 1_000_000, 1_000, cxl_queue, 100_000, 100),
        ),
    )


def test_explicit_baselines_have_distinct_behavior() -> None:
    snapshot = _snapshot(dram_queue=1_000_000)
    assert FixedL1FirstPolicy().decide_lookup(snapshot).residency_id == "dram-r"
    assert AlwaysFetchPolicy().decide_lookup(snapshot).residency_id == "cxl-r"
    assert isinstance(
        AlwaysRecomputePolicy().decide_lookup(snapshot), RecomputeDecision
    )


def test_cost_policy_has_dram_cxl_and_recompute_crossovers() -> None:
    policy = CostAwarePolicy(cuda_bandwidth_bytes_per_s=1_000_000)

    dram = policy.decide_lookup(_snapshot())
    cxl = policy.decide_lookup(_snapshot(dram_queue=2_000_000))
    recompute = policy.decide_lookup(
        _snapshot(dram_queue=2_000_000, cxl_queue=2_000_000, recompute_ns=50)
    )

    assert isinstance(dram, FetchDecision) and dram.residency_id == "dram-r"
    assert isinstance(cxl, FetchDecision) and cxl.residency_id == "cxl-r"
    assert isinstance(recompute, RecomputeDecision)
    reason = json.loads(cxl.reason)
    assert reason["winner"] == "cxl-r"
    assert set(reason["estimates"]) == {"cxl-r", "dram-r", "recompute"}


def test_candidate_filter_and_deadline_are_applied_before_cost() -> None:
    policy = CostAwarePolicy(cuda_bandwidth_bytes_per_s=1_000_000)
    decision = policy.decide_lookup(
        _snapshot(
            dram_state="evicting",
            cxl_layout="b" * 64,
            deadline_ns=101,
        )
    )

    assert isinstance(decision, RecomputeDecision)
    detail = json.loads(decision.reason)
    assert detail["rejections"] == {
        "cxl-r": "layout_mismatch",
        "dram-r": "state_evicting",
    }
