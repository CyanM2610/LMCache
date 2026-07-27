# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import FrozenInstanceError
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import (
    FetchDecision,
    RecomputeDecision,
    StorePlacementPlan,
    TargetSpec,
    validate_store_plan,
)
from lmcache.v1.multiprocess.cxl.observations import (
    PlacementSnapshot,
    RequestObservation,
    ResidencyObservation,
    TierObservation,
)


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def test_contracts_are_frozen_validated_and_json_primitive() -> None:
    residency = ResidencyObservation(
        residency_id="dram-1",
        object_key=_key(),
        tier="dram",
        state="ready",
        size_bytes=4096,
        generation=2,
        layout_fingerprint="a" * 64,
        last_access_ns=10,
        access_count=3,
        active_readers=1,
        pinned=False,
    )
    request = RequestObservation(
        request_id="request-1",
        instance_id=7,
        object_key=_key(),
        required_bytes=4096,
        external_matched_tokens=256,
        deadline_ns=100_000,
        recompute_estimate_ns=50_000,
        layout_fingerprint="a" * 64,
    )
    snapshot = PlacementSnapshot(
        timestamp_ns=1,
        request=request,
        residencies=(residency,),
        tiers=(TierObservation("dram", 8192, 4096, 0, 1_000_000, 100),),
    )
    encoded = json.dumps(snapshot.to_primitive(), sort_keys=True)

    assert "chunk_hash" in encoded
    assert "pointer" not in encoded.lower()
    with pytest.raises(FrozenInstanceError):
        residency.generation = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: TierObservation("dram", 1, 2, 0, 1, 0),
            "used_bytes",
        ),
        (
            lambda: ResidencyObservation(
                "r",
                _key(),
                "cxl",
                "ready",
                -1,
                1,
                "a" * 64,
                0,
                0,
                0,
                False,
            ),
            "size_bytes",
        ),
        (lambda: FetchDecision("r", 0, "fast"), "estimated_completion_ns"),
        (lambda: RecomputeDecision(""), "reason"),
    ],
)
def test_contracts_reject_invalid_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_store_plan_rejects_empty_duplicate_and_unavailable_targets() -> None:
    with pytest.raises(ValueError, match="targets"):
        StorePlacementPlan(_key(), (), "empty")
    with pytest.raises(ValueError, match="duplicate"):
        StorePlacementPlan(
            _key(),
            (TargetSpec("dram", True, "a"), TargetSpec("dram", False, "b")),
            "duplicate",
        )
    plan = StorePlacementPlan(_key(), (TargetSpec("cxl", True, "required"),), "test")
    with pytest.raises(ValueError, match="unavailable"):
        validate_store_plan(plan, frozenset({"dram"}))
