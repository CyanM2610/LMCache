# SPDX-License-Identifier: Apache-2.0
"""Serializable placement decisions produced by Gate E policies."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

# First Party
from lmcache.v1.distributed.api import ObjectKey


Tier = Literal["dram", "cxl"]


@dataclass(frozen=True)
class TargetSpec:
    """One independent target of a STORE placement decision."""

    tier: Tier
    required: bool
    reason: str

    def __post_init__(self) -> None:
        if self.tier not in ("dram", "cxl"):
            raise ValueError("target tier must be 'dram' or 'cxl'")
        if not self.reason:
            raise ValueError("target reason must not be empty")


@dataclass(frozen=True)
class StorePlacementPlan:
    """Immutable set of required and optional STORE targets."""

    object_key: ObjectKey
    targets: tuple[TargetSpec, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("store plan targets must not be empty")
        tiers = [target.tier for target in self.targets]
        if len(set(tiers)) != len(tiers):
            raise ValueError("store plan contains a duplicate target tier")
        if not self.reason:
            raise ValueError("store plan reason must not be empty")

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "object_key": asdict(self.object_key.to_encoded_object_key()),
            "targets": [asdict(target) for target in self.targets],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FetchDecision:
    """Advisory choice of one immutable residency."""

    residency_id: str
    estimated_completion_ns: int
    reason: str

    def __post_init__(self) -> None:
        if not self.residency_id:
            raise ValueError("residency_id must not be empty")
        if self.estimated_completion_ns <= 0:
            raise ValueError("estimated_completion_ns must be positive")
        if not self.reason:
            raise ValueError("fetch reason must not be empty")

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class RecomputeDecision:
    """Explicit choice to avoid an external fetch."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("recompute reason must not be empty")

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


LookupDecision: TypeAlias = FetchDecision | RecomputeDecision


def validate_store_plan(
    plan: StorePlacementPlan, available_tiers: frozenset[Tier]
) -> None:
    """Fail closed when a plan names an unavailable execution tier."""
    unavailable = [
        target.tier for target in plan.targets if target.tier not in available_tiers
    ]
    if unavailable:
        raise ValueError(f"store target tier is unavailable: {unavailable[0]}")
