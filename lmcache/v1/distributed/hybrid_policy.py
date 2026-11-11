# SPDX-License-Identifier: Apache-2.0
"""Replaceable placement decisions; ownership and protection stay in runtime."""

# Standard
from dataclasses import dataclass
from typing import Literal, Protocol

Medium = Literal["dram", "cxl_sim", "cxl_dax"]


@dataclass(frozen=True)
class EvictionCandidate:
    """Runtime snapshot: stable identity, location, physical size and protection."""

    object_id: str
    medium: Medium
    physical_bytes: int
    last_access_ns: int
    evictable: bool


class PlacementPolicy(Protocol):
    """Pure decisions with no allocator, mutation, clock, or future request access."""

    def pool_order(self) -> tuple[Medium, ...]:
        """Return pools to try for new objects, in preference order."""
        ...

    def victims(self, candidates: list[EvictionCandidate]) -> list[str]:
        """Return permitted victim IDs in deterministic preference order."""
        ...


class DramFirstPolicy:
    """DRAM then CXL; evict only legal CXL residents, LRU with stable ID ties."""

    def __init__(self, cxl_medium: Medium = "cxl_sim") -> None:
        """Select the CXL backing label without changing placement decisions."""
        if cxl_medium not in ("cxl_sim", "cxl_dax"):
            raise ValueError("expected a CXL medium")
        self.cxl_medium = cxl_medium

    def pool_order(self) -> tuple[Medium, ...]:
        """Use free DRAM first; do not migrate existing residents."""
        return ("dram", self.cxl_medium)

    def victims(self, candidates: list[EvictionCandidate]) -> list[str]:
        """Order eligible CXL residents; never return pinned or DRAM objects.

        Args:
            candidates: Current-state snapshots supplied by the runtime.
        Returns:
            Stable object IDs, from least to most recently accessed.
        """
        eligible = [
            entry
            for entry in candidates
            if entry.medium == self.cxl_medium and entry.evictable
        ]
        return [
            entry.object_id
            for entry in sorted(eligible, key=lambda e: (e.last_access_ns, e.object_id))
        ]


class ForcedPoolPolicy(DramFirstPolicy):
    """Diagnostic placement hook. Results must not be called normal-policy data."""

    def __init__(self, medium: Medium) -> None:
        """Select one pool for all new allocations, without allowing migration.

        Args:
            medium: The only permitted destination.
        Raises:
            ValueError: If medium is unknown.
        """
        if medium not in ("dram", "cxl_sim", "cxl_dax"):
            raise ValueError("unknown hybrid medium")
        super().__init__("cxl_dax" if medium == "cxl_dax" else "cxl_sim")
        self.medium = medium

    def pool_order(self) -> tuple[Medium, ...]:
        """Return only the forced medium."""
        return (self.medium,)

    def victims(self, candidates: list[EvictionCandidate]) -> list[str]:
        """Keep the same legal CXL-only eviction rule, even in diagnostics."""
        return super().victims(candidates) if self.medium != "dram" else []


class AllPoolsLRUPolicy(DramFirstPolicy):
    """Prefer free DRAM, then CXL; reclaim legal objects from either pool."""

    def victims(self, candidates: list[EvictionCandidate]) -> list[str]:
        """Order only legal objects by runtime access time and stable identity."""
        eligible = [entry for entry in candidates if entry.evictable]
        return [
            entry.object_id
            for entry in sorted(eligible, key=lambda e: (e.last_access_ns, e.object_id))
        ]
