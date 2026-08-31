# SPDX-License-Identifier: Apache-2.0

"""Capacity reservations for HotPrefix shared Host residencies."""

# Standard
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Mapping
import time

# First Party
from lmcache.v1.multiprocess.hotprefix.admission import (
    AdmissionAction,
    HostAdmissionCandidate,
    HostAdmissionDecision,
    HostAdmissionPolicy,
    HostResidencyObservation,
)

PrefixId = bytes


class HostResidencyState(Enum):
    """Publication state for one immutable Host residency."""

    RESERVED = "reserved"
    READY = "ready"
    INVALID = "invalid"


@dataclass(frozen=True)
class HostResidency:
    """One capacity-accounted shared Host residency."""

    prefix_id: PrefixId
    size_bytes: int
    generation: int
    frequency: int
    clock: int
    state: HostResidencyState


@dataclass(frozen=True)
class HostReadLease:
    """Generation-bound read authorization for one immutable residency."""

    ticket_id: bytes
    prefix_id: PrefixId
    generation: int
    size_bytes: int
    expires_at: float


class HostResidencyDirectory:
    """Reserve and publish one immutable Host residency per LogicalPrefix.

    Args:
        capacity_bytes: Total capacity of the active HotPrefix Host Tier.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int,
        lease_ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self._capacity_bytes = capacity_bytes
        self._lease_ttl_seconds = lease_ttl_seconds
        self._clock = clock
        self._residencies: dict[PrefixId, HostResidency] = {}
        self._generations: dict[PrefixId, int] = {}
        self._leases: dict[bytes, HostReadLease] = {}
        self._rollback_victims: dict[PrefixId, tuple[HostResidency, ...]] = {}

    @property
    def used_bytes(self) -> int:
        """Return bytes held by reserved and READY residencies."""
        return sum(item.size_bytes for item in self._residencies.values())

    def reserve(
        self,
        candidate: HostAdmissionCandidate,
        policy: HostAdmissionPolicy,
        *,
        hotness_by_prefix: Mapping[PrefixId, tuple[int, int]] | None = None,
        generation: int | None = None,
    ) -> HostAdmissionDecision:
        """Apply selective admission and reserve capacity on ACCEPT.

        Args:
            candidate: Evicted HBM prefix proposed for storage.
            policy: Pure HotPrefix admission policy.
            hotness_by_prefix: Optional current Global frequency/clock overrides
                for READY replacement candidates.
            generation: Scheduler-proposed generation shared by all MP servers.

        Returns:
            DEDUP, REJECT, or ACCEPT with evicted cold prefixes.

        Raises:
            RuntimeError: If the same prefix already has an in-flight reserve or
                a selected victim becomes unsafe.
        """
        existing = self._residencies.get(candidate.prefix_id)
        if existing is not None and existing.state is HostResidencyState.RESERVED:
            raise RuntimeError("admission for prefix is already in progress")
        observations = tuple(
            HostResidencyObservation(
                item.prefix_id,
                item.size_bytes,
                (
                    hotness_by_prefix[item.prefix_id][0]
                    if hotness_by_prefix is not None
                    and item.prefix_id in hotness_by_prefix
                    else item.frequency
                ),
                (
                    hotness_by_prefix[item.prefix_id][1]
                    if hotness_by_prefix is not None
                    and item.prefix_id in hotness_by_prefix
                    else item.clock
                ),
                item.state is HostResidencyState.READY,
                not self._has_active_lease(item.prefix_id),
            )
            for item in self._residencies.values()
        )
        decision = policy.decide(
            candidate,
            observations,
            capacity_bytes=self._capacity_bytes,
            used_bytes=self.used_bytes,
        )
        if decision.action is not AdmissionAction.ACCEPT:
            return decision
        rollback_victims: list[HostResidency] = []
        for prefix_id in decision.evict_prefixes:
            victim = self._residencies.get(prefix_id)
            if victim is None or victim.state is not HostResidencyState.READY:
                raise RuntimeError("admission victim is no longer READY")
            if self._has_active_lease(prefix_id):
                raise RuntimeError("admission victim acquired a read lease")
            rollback_victims.append(self._residencies.pop(prefix_id))
        if generation is None:
            generation = self._generations.get(candidate.prefix_id, 0) + 1
        elif generation <= 0:
            raise ValueError("generation must be positive")
        self._generations[candidate.prefix_id] = generation
        self._residencies[candidate.prefix_id] = HostResidency(
            candidate.prefix_id,
            candidate.size_bytes,
            generation,
            candidate.frequency,
            candidate.clock,
            HostResidencyState.RESERVED,
        )
        self._rollback_victims[candidate.prefix_id] = tuple(rollback_victims)
        return decision

    def publish(self, prefix_id: PrefixId) -> HostResidency:
        """Publish a completely written reserved residency as READY.

        Args:
            prefix_id: Reserved LogicalPrefix to publish.

        Returns:
            The immutable READY residency. Repeated publication is idempotent.
        """
        residency = self._residencies[prefix_id]
        if residency.state is HostResidencyState.READY:
            return residency
        if residency.state is not HostResidencyState.RESERVED:
            raise RuntimeError("only a reserved residency can be published")
        ready = replace(residency, state=HostResidencyState.READY)
        self._residencies[prefix_id] = ready
        self._rollback_victims.pop(prefix_id, None)
        return ready

    def abort(self, prefix_id: PrefixId) -> None:
        """Release an incomplete reservation and restore replacement victims.

        Args:
            prefix_id: Reserved LogicalPrefix to abort. Missing prefixes are a
                successful no-op for retry safety.
        """
        residency = self._residencies.get(prefix_id)
        if residency is None:
            return
        if residency.state is not HostResidencyState.RESERVED:
            raise RuntimeError("only a reserved residency can be aborted")
        del self._residencies[prefix_id]
        for victim in self._rollback_victims.pop(prefix_id, ()):
            if victim.prefix_id in self._residencies:
                raise RuntimeError("rollback victim prefix was reused")
            self._residencies[victim.prefix_id] = victim

    def evict(self, prefix_id: PrefixId) -> HostResidency:
        """Remove and return a READY shared Host residency.

        Args:
            prefix_id: LogicalPrefix to remove.

        Returns:
            The removed residency.

        Raises:
            RuntimeError: If the residency is not READY or has an active lease.
        """
        residency = self._residencies[prefix_id]
        if residency.state is not HostResidencyState.READY:
            raise RuntimeError("only a READY residency can be evicted")
        if self._has_active_lease(prefix_id):
            raise RuntimeError("cannot evict a residency with an active read lease")
        return self._residencies.pop(prefix_id)

    def invalidate(self, prefix_id: PrefixId, generation: int) -> bool:
        """Remove a stale READY generation after a physical data-plane miss.

        Args:
            prefix_id: LogicalPrefix whose payload was missing.
            generation: Exact generation verified by the failed transfer.

        Returns:
            ``True`` only when that exact generation was removed.
        """
        residency = self._residencies.get(prefix_id)
        if residency is None:
            return False
        if residency.generation != generation:
            return False
        if residency.state is not HostResidencyState.READY:
            raise RuntimeError("cannot invalidate an in-flight Host STORE")
        if self._has_active_lease(prefix_id):
            self._residencies[prefix_id] = replace(
                residency,
                state=HostResidencyState.INVALID,
            )
        else:
            del self._residencies[prefix_id]
        return True

    def acquire(
        self,
        prefix_id: PrefixId,
        generation: int,
        ticket_id: bytes,
    ) -> HostReadLease:
        """Acquire a repeatable read lease for an exact READY generation.

        Args:
            prefix_id: Promotion source LogicalPrefix.
            generation: Candidate generation observed by the planner.
            ticket_id: Client-generated idempotency key shared by all servers.

        Returns:
            A renewable generation-bound read lease.

        Raises:
            RuntimeError: If the source is absent, not READY, or changed.
        """
        if not ticket_id:
            raise ValueError("ticket_id must not be empty")
        self._purge_expired_leases()
        existing = self._leases.get(ticket_id)
        if existing is not None:
            if existing.prefix_id != prefix_id or existing.generation != generation:
                raise RuntimeError("ticket_id is already bound to another residency")
            return existing
        residency = self._residencies.get(prefix_id)
        if residency is None or residency.state is not HostResidencyState.READY:
            raise RuntimeError("promotion source is not READY")
        if residency.generation != generation:
            raise RuntimeError("promotion source generation changed")
        lease = HostReadLease(
            ticket_id,
            prefix_id,
            generation,
            residency.size_bytes,
            self._clock() + self._lease_ttl_seconds,
        )
        self._leases[ticket_id] = lease
        return lease

    def release(
        self, ticket_id: bytes, *, missing_ok: bool = False
    ) -> HostReadLease | None:
        """Release and return a previously acquired read lease.

        Args:
            ticket_id: Lease identifier to release.
            missing_ok: Treat an absent or expired ticket as an idempotent no-op.

        Returns:
            The released lease, or ``None`` for a permitted missing ticket.
        """
        self._purge_expired_leases()
        try:
            lease = self._leases.pop(ticket_id)
        except KeyError as error:
            if missing_ok:
                return None
            raise RuntimeError("promotion ticket is not active") from error
        self._remove_unleased_invalid_residency(lease.prefix_id)
        return lease

    def renew(self, ticket_id: bytes) -> HostReadLease:
        """Extend an active promotion lease from the server's current clock.

        Args:
            ticket_id: Active lease identifier.

        Returns:
            The renewed lease with a new expiry.

        Raises:
            RuntimeError: If the ticket is absent or already expired.
        """
        self._purge_expired_leases()
        try:
            lease = self._leases[ticket_id]
        except KeyError as error:
            raise RuntimeError("promotion ticket is not active") from error
        renewed = replace(
            lease,
            expires_at=self._clock() + self._lease_ttl_seconds,
        )
        self._leases[ticket_id] = renewed
        return renewed

    def snapshot(self) -> tuple[HostResidency, ...]:
        """Return all residencies in deterministic prefix order.

        Returns:
            Immutable residency snapshots sorted by LogicalPrefix identifier.
        """
        return tuple(
            sorted(self._residencies.values(), key=lambda item: item.prefix_id)
        )

    def get(self, prefix_id: PrefixId) -> HostResidency | None:
        """Return one residency when present.

        Args:
            prefix_id: LogicalPrefix identifier.

        Returns:
            The current reservation/READY residency, or ``None``.
        """
        return self._residencies.get(prefix_id)

    def _has_active_lease(self, prefix_id: PrefixId) -> bool:
        self._purge_expired_leases()
        return any(lease.prefix_id == prefix_id for lease in self._leases.values())

    def _purge_expired_leases(self) -> None:
        now = self._clock()
        expired = [
            ticket_id
            for ticket_id, lease in self._leases.items()
            if lease.expires_at <= now
        ]
        for ticket_id in expired:
            lease = self._leases.pop(ticket_id)
            self._remove_unleased_invalid_residency(lease.prefix_id)

    def _remove_unleased_invalid_residency(self, prefix_id: PrefixId) -> None:
        residency = self._residencies.get(prefix_id)
        if residency is None or residency.state is not HostResidencyState.INVALID:
            return
        if any(lease.prefix_id == prefix_id for lease in self._leases.values()):
            return
        del self._residencies[prefix_id]
