# SPDX-License-Identifier: Apache-2.0
"""Explainable baseline and cost-aware Gate E placement policies."""

# Future
from __future__ import annotations

# Standard
from typing import Protocol
import json
import math

# Local
from .actions import (
    FetchDecision,
    LookupDecision,
    RecomputeDecision,
    StorePlacementPlan,
    TargetSpec,
    Tier,
)
from .observations import PlacementSnapshot, ResidencyObservation, TierObservation


class PlacementPolicy(Protocol):
    """Stable policy seam containing domain objects only."""

    def plan_store(self, snapshot: PlacementSnapshot) -> StorePlacementPlan: ...

    def decide_lookup(self, snapshot: PlacementSnapshot) -> LookupDecision: ...


def _duration_ns(byte_count: int, bytes_per_second: int) -> int:
    return math.ceil(byte_count * 1_000_000_000 / bytes_per_second)


def _eligible(
    snapshot: PlacementSnapshot,
) -> tuple[list[ResidencyObservation], dict[str, str]]:
    tiers = {tier.tier for tier in snapshot.tiers}
    eligible: list[ResidencyObservation] = []
    rejections: dict[str, str] = {}
    for residency in snapshot.residencies:
        reason = None
        if residency.state != "ready":
            reason = f"state_{residency.state}"
        elif residency.layout_fingerprint != snapshot.request.layout_fingerprint:
            reason = "layout_mismatch"
        elif residency.tier not in tiers:
            reason = "tier_unreachable"
        if reason is None:
            eligible.append(residency)
        else:
            rejections[residency.residency_id] = reason
    return eligible, rejections


class _BasePolicy:
    def __init__(self, store_tier: Tier = "cxl") -> None:
        if store_tier not in ("dram", "cxl"):
            raise ValueError("store_tier is unsupported")
        self._store_tier = store_tier

    def plan_store(self, snapshot: PlacementSnapshot) -> StorePlacementPlan:
        return StorePlacementPlan(
            snapshot.request.object_key,
            (TargetSpec(self._store_tier, True, "policy baseline"),),
            f"required {self._store_tier} baseline",
        )


class FixedL1FirstPolicy(_BasePolicy):
    """Explicit DRAM-first baseline independent of estimated cost."""

    def decide_lookup(self, snapshot: PlacementSnapshot) -> LookupDecision:
        candidates, rejections = _eligible(snapshot)
        for tier in ("dram", "cxl"):
            found = next((item for item in candidates if item.tier == tier), None)
            if found is not None:
                return FetchDecision(
                    found.residency_id,
                    1,
                    json.dumps(
                        {
                            "baseline": "fixed_l1_first",
                            "winner": found.residency_id,
                            "rejections": rejections,
                        },
                        sort_keys=True,
                    ),
                )
        return RecomputeDecision(
            json.dumps(
                {"baseline": "fixed_l1_first", "rejections": rejections},
                sort_keys=True,
            )
        )


class AlwaysRecomputePolicy(_BasePolicy):
    """Explicit no-external-fetch baseline."""

    def decide_lookup(self, snapshot: PlacementSnapshot) -> LookupDecision:
        del snapshot
        return RecomputeDecision("always_recompute baseline")


class CostAwarePolicy(_BasePolicy):
    """Choose the minimum feasible DRAM/CXL proxy/recompute estimate."""

    def __init__(
        self,
        *,
        cuda_bandwidth_bytes_per_s: int = 25_000_000_000,
        cuda_latency_ns: int = 0,
        store_tier: Tier = "cxl",
    ) -> None:
        super().__init__(store_tier)
        if cuda_bandwidth_bytes_per_s <= 0 or cuda_latency_ns < 0:
            raise ValueError("CUDA estimates must be positive/non-negative")
        self._cuda_bandwidth = cuda_bandwidth_bytes_per_s
        self._cuda_latency_ns = cuda_latency_ns

    def decide_lookup(self, snapshot: PlacementSnapshot) -> LookupDecision:
        candidates, rejections = _eligible(snapshot)
        tier_by_name = {tier.tier: tier for tier in snapshot.tiers}
        estimates: dict[str, int] = {
            "recompute": snapshot.request.recompute_estimate_ns
        }
        feasible: list[tuple[int, ResidencyObservation]] = []
        for residency in candidates:
            estimate = self._fetch_estimate(
                snapshot.request.required_bytes, tier_by_name[residency.tier]
            )
            estimates[residency.residency_id] = estimate
            if (
                snapshot.request.deadline_ns is not None
                and snapshot.timestamp_ns + estimate > snapshot.request.deadline_ns
            ):
                rejections[residency.residency_id] = "deadline_infeasible"
                continue
            feasible.append((estimate, residency))

        winner: str = "recompute"
        winner_estimate = snapshot.request.recompute_estimate_ns
        if feasible:
            fetch_estimate, fetch_residency = min(
                feasible, key=lambda item: (item[0], item[1].residency_id)
            )
            recompute_feasible = (
                snapshot.request.deadline_ns is None
                or snapshot.timestamp_ns + winner_estimate
                <= snapshot.request.deadline_ns
            )
            if fetch_estimate < winner_estimate or not recompute_feasible:
                winner = fetch_residency.residency_id
                winner_estimate = fetch_estimate

        reason = json.dumps(
            {
                "policy": "cost_aware_v1",
                "winner": winner,
                "estimates": estimates,
                "rejections": rejections,
            },
            sort_keys=True,
        )
        if winner == "recompute":
            return RecomputeDecision(reason)
        return FetchDecision(winner, winner_estimate, reason)

    def _fetch_estimate(self, byte_count: int, tier: TierObservation) -> int:
        cuda_ns = _duration_ns(byte_count, self._cuda_bandwidth) + self._cuda_latency_ns
        queue_ns = _duration_ns(tier.queued_bytes, tier.estimated_bandwidth_bytes_per_s)
        if tier.tier == "dram":
            return cuda_ns + queue_ns
        modeled_service_ns = (
            _duration_ns(byte_count, tier.estimated_bandwidth_bytes_per_s)
            + tier.estimated_latency_ns
        )
        return max(cuda_ns, queue_ns + modeled_service_ns)


class AlwaysFetchPolicy(CostAwarePolicy):
    """Fetch the lowest raw eligible source even when recompute is cheaper."""

    def decide_lookup(self, snapshot: PlacementSnapshot) -> LookupDecision:
        candidates, rejections = _eligible(snapshot)
        if not candidates:
            return RecomputeDecision(
                json.dumps(
                    {"baseline": "always_fetch", "rejections": rejections},
                    sort_keys=True,
                )
            )
        tier_by_name = {tier.tier: tier for tier in snapshot.tiers}
        estimates = {
            item.residency_id: self._fetch_estimate(
                snapshot.request.required_bytes, tier_by_name[item.tier]
            )
            for item in candidates
        }
        winner = min(estimates, key=lambda item: (estimates[item], item))
        return FetchDecision(
            winner,
            estimates[winner],
            json.dumps(
                {
                    "baseline": "always_fetch",
                    "winner": winner,
                    "estimates": estimates,
                    "rejections": rejections,
                },
                sort_keys=True,
            ),
        )
