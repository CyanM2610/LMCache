# SPDX-License-Identifier: Apache-2.0

"""Selective shared-Host admission policy for multi-instance HotPrefix."""

# Standard
from dataclasses import dataclass
from enum import Enum

PrefixId = bytes


class AdmissionAction(Enum):
    """Terminal policy action for one HBM eviction candidate."""

    DEDUP = "dedup"
    REJECT = "reject"
    ACCEPT = "accept"


@dataclass(frozen=True)
class HostAdmissionCandidate:
    """One evicted logical prefix proposed for shared Host storage."""

    prefix_id: PrefixId
    size_bytes: int
    frequency: int
    clock: int

    def __post_init__(self) -> None:
        if not self.prefix_id:
            raise ValueError("prefix_id must not be empty")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.frequency < 0 or self.clock < 0:
            raise ValueError("candidate hotness fields must be non-negative")

    @property
    def hotness(self) -> int:
        """Return the Global Hotness used by Host admission."""
        return self.frequency * self.clock


@dataclass(frozen=True)
class HostResidencyObservation:
    """Policy-safe view of one shared Host residency."""

    prefix_id: PrefixId
    size_bytes: int
    frequency: int
    clock: int
    ready: bool
    evictable: bool = True

    def __post_init__(self) -> None:
        if not self.prefix_id:
            raise ValueError("prefix_id must not be empty")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.frequency < 0 or self.clock < 0:
            raise ValueError("residency hotness fields must be non-negative")

    @property
    def hotness(self) -> int:
        """Return the current Global Hotness of this residency."""
        return self.frequency * self.clock


@dataclass(frozen=True)
class HostAdmissionDecision:
    """Explainable DEDUP, REJECT, or ACCEPT result."""

    action: AdmissionAction
    reason: str
    evict_prefixes: tuple[PrefixId, ...] = ()


class HostAdmissionPolicy:
    """Apply HotPrefix threshold and coldest-residency replacement.

    Args:
        frequency_threshold: Minimum fleet-wide frequency eligible for storage.
    """

    def __init__(self, *, frequency_threshold: int) -> None:
        if frequency_threshold < 0:
            raise ValueError("frequency_threshold must be non-negative")
        self._frequency_threshold = frequency_threshold

    def decide(
        self,
        candidate: HostAdmissionCandidate,
        residencies: tuple[HostResidencyObservation, ...],
        *,
        capacity_bytes: int,
        used_bytes: int,
    ) -> HostAdmissionDecision:
        """Choose shared-Host admission without mutating residency state.

        Args:
            candidate: Evicted prefix and its latest Global Hotness facts.
            residencies: Current shared-Host residency observations.
            capacity_bytes: Total Host tier capacity.
            used_bytes: Currently allocated Host tier bytes.

        Returns:
            Explainable admission decision and cold prefixes to evict.

        Raises:
            ValueError: If capacity observations are inconsistent.
        """
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if used_bytes < 0 or used_bytes > capacity_bytes:
            raise ValueError("used_bytes must be within Host capacity")
        if any(
            item.ready and item.prefix_id == candidate.prefix_id for item in residencies
        ):
            return HostAdmissionDecision(AdmissionAction.DEDUP, "ready_residency")
        if candidate.frequency < self._frequency_threshold:
            return HostAdmissionDecision(
                AdmissionAction.REJECT, "frequency_below_threshold"
            )
        if candidate.size_bytes > capacity_bytes:
            return HostAdmissionDecision(
                AdmissionAction.REJECT, "candidate_exceeds_host_capacity"
            )

        bytes_needed = max(0, used_bytes + candidate.size_bytes - capacity_bytes)
        if bytes_needed == 0:
            return HostAdmissionDecision(AdmissionAction.ACCEPT, "capacity_available")

        ready = sorted(
            (item for item in residencies if item.ready and item.evictable),
            key=lambda item: (item.hotness, item.prefix_id),
        )
        victims: list[HostResidencyObservation] = []
        reclaimed = 0
        for residency in ready:
            victims.append(residency)
            reclaimed += residency.size_bytes
            if reclaimed >= bytes_needed:
                break
        if reclaimed < bytes_needed:
            return HostAdmissionDecision(
                AdmissionAction.REJECT, "insufficient_reclaimable_capacity"
            )
        if any(candidate.hotness <= victim.hotness for victim in victims):
            return HostAdmissionDecision(
                AdmissionAction.REJECT, "not_hotter_than_replacement"
            )
        return HostAdmissionDecision(
            AdmissionAction.ACCEPT,
            "replace_colder_residency",
            tuple(victim.prefix_id for victim in victims),
        )
