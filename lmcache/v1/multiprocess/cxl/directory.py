# SPDX-License-Identifier: Apache-2.0
"""Independent immutable DRAM/CXL residency directory for Gate E."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import threading
import time
import uuid

# First Party
from lmcache.v1.distributed.api import ObjectKey

# Local
from .actions import TargetSpec, Tier
from .contracts import ExtentDescriptor
from .region_manager import CXLRegionManager
from .residency import ReadLease


class ResidencyState(str, Enum):
    """Lifecycle state of an immutable residency."""

    RESERVED = "reserved"
    WRITING = "writing"
    READY = "ready"
    EVICTING = "evicting"


@dataclass(frozen=True)
class Residency:
    """Caller-facing immutable snapshot of one tier replica."""

    residency_id: str
    object_key: ObjectKey
    tier: Tier
    descriptor: ExtentDescriptor | None
    state: ResidencyState
    generation: int
    last_access_ns: int
    access_count: int
    active_readers: int
    pinned: bool


@dataclass
class _Entry:
    residency_id: str
    object_key: ObjectKey
    tier: Tier
    reservation_id: str
    generation: int
    state: ResidencyState
    descriptor: ExtentDescriptor | None = None
    last_access_ns: int = 0
    access_count: int = 0
    pinned: bool = False
    leases: set[str] = field(default_factory=set)


class MultiResidencyDirectory:
    """Map one ObjectKey to independently managed immutable tier replicas."""

    def __init__(self, managers: Mapping[Tier, CXLRegionManager]) -> None:
        if not managers:
            raise ValueError("at least one residency manager is required")
        self._managers = dict(managers)
        self._by_key: dict[ObjectKey, dict[str, _Entry]] = {}
        self._by_id: dict[str, _Entry] = {}
        self._leases: dict[str, tuple[_Entry, ReadLease]] = {}
        self._retired_leases: set[str] = set()
        self._lock = threading.RLock()

    @property
    def available_tiers(self) -> frozenset[Tier]:
        """Return tiers with an executable allocator."""
        return frozenset(self._managers)

    def list_residencies(self, object_key: ObjectKey) -> tuple[Residency, ...]:
        """Return stable snapshots of all replicas for one object."""
        with self._lock:
            entries = self._by_key.get(object_key, {}).values()
            return tuple(
                self._snapshot(entry)
                for entry in sorted(entries, key=lambda item: item.residency_id)
            )

    def get_residency(self, residency_id: str) -> Residency:
        """Return a snapshot by immutable residency identity."""
        with self._lock:
            return self._snapshot(self._get_entry(residency_id))

    def reserve_residency(
        self,
        object_key: ObjectKey,
        target_spec: TargetSpec,
        *,
        length: int,
        alignment: int,
    ) -> Residency:
        """Reserve a new extent without changing existing READY replicas."""
        if target_spec.tier not in self._managers:
            raise ValueError(f"store target tier is unavailable: {target_spec.tier}")
        manager = self._managers[target_spec.tier]
        with self._lock:
            reservation = manager.reserve(length, alignment)
            entry = _Entry(
                residency_id=uuid.uuid4().hex,
                object_key=object_key,
                tier=target_spec.tier,
                reservation_id=reservation.reservation_id,
                generation=reservation.generation,
                state=ResidencyState.RESERVED,
                last_access_ns=time.monotonic_ns(),
            )
            self._by_key.setdefault(object_key, {})[entry.residency_id] = entry
            self._by_id[entry.residency_id] = entry
            return self._snapshot(entry)

    def mark_writing(self, residency_id: str) -> Residency:
        """Bind the reserved descriptor and enter WRITING."""
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.RESERVED)
            entry.descriptor = self._managers[entry.tier].begin_write(
                entry.reservation_id
            )
            entry.state = ResidencyState.WRITING
            return self._snapshot(entry)

    def publish(self, residency_id: str, completion: Any | None) -> Residency:
        """Publish one successful target without affecting sibling replicas."""
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.WRITING)
            if completion is not None and getattr(completion, "success", None) is False:
                raise RuntimeError("cannot publish a failed transfer completion")
            entry.descriptor = self._managers[entry.tier].publish(entry.reservation_id)
            entry.state = ResidencyState.READY
            entry.last_access_ns = time.monotonic_ns()
            return self._snapshot(entry)

    def abort(self, residency_id: str, reason: str) -> None:
        """Abort one unpublished target and leave all siblings unchanged."""
        with self._lock:
            entry = self._get_entry(residency_id)
            if entry.state not in (ResidencyState.RESERVED, ResidencyState.WRITING):
                raise RuntimeError(f"cannot abort residency in {entry.state.value}")
            self._managers[entry.tier].abort(entry.reservation_id, reason)
            self._remove_entry(entry)

    def acquire_read(
        self,
        residency_id: str,
        expected_generation: int,
        ttl_ns: int,
        *,
        now_ns: int | None = None,
    ) -> ReadLease:
        """Atomically compare state/generation and reserve one reader."""
        if ttl_ns <= 0:
            raise ValueError("lease TTL must be positive")
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            entry = self._by_id.get(residency_id)
            if entry is None or entry.state != ResidencyState.READY:
                raise KeyError("residency has no READY generation")
            if entry.generation != expected_generation:
                raise RuntimeError("residency generation changed before binding")
            lease = ReadLease(
                lease_id=uuid.uuid4().hex,
                residency_id=residency_id,
                generation=entry.generation,
                expires_at_ns=now + ttl_ns,
            )
            entry.leases.add(lease.lease_id)
            entry.last_access_ns = now
            entry.access_count += 1
            self._leases[lease.lease_id] = (entry, lease)
            return lease

    def validate_lease(self, lease: ReadLease, now_ns: int) -> Residency:
        """Validate an issued lease against the exact current residency."""
        with self._lock:
            pair = self._leases.get(lease.lease_id)
            if pair is None or pair[1] != lease:
                raise RuntimeError("read lease is terminal or unknown")
            entry = pair[0]
            if lease.expires_at_ns <= now_ns:
                raise RuntimeError("read lease expired")
            if entry.generation != lease.generation:
                raise RuntimeError("read lease generation is stale")
            if entry.state not in (ResidencyState.READY, ResidencyState.EVICTING):
                raise RuntimeError("read lease residency is not readable")
            return self._snapshot(entry)

    def release_read(self, lease_id: str, *, now_ns: int | None = None) -> None:
        """Release one lease exactly once."""
        del now_ns
        with self._lock:
            if lease_id in self._retired_leases:
                return
            pair = self._leases.pop(lease_id, None)
            if pair is None:
                raise KeyError("unknown read lease")
            entry, _ = pair
            entry.leases.discard(lease_id)
            self._retired_leases.add(lease_id)

    def begin_evict(self, residency_id: str) -> None:
        """Block new readers while existing generation leases drain."""
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.READY)
            if entry.descriptor is None:
                raise RuntimeError("READY residency has no descriptor")
            self._managers[entry.tier].begin_evict(entry.descriptor)
            entry.state = ResidencyState.EVICTING

    def reclaim(self, residency_id: str) -> None:
        """Reclaim an EVICTING residency after all readers release."""
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.EVICTING)
            if entry.leases:
                raise RuntimeError("cannot reclaim residency with active readers")
            if entry.descriptor is None:
                raise RuntimeError("EVICTING residency has no descriptor")
            self._managers[entry.tier].reclaim(entry.descriptor)
            self._remove_entry(entry)

    @staticmethod
    def _snapshot(entry: _Entry) -> Residency:
        return Residency(
            residency_id=entry.residency_id,
            object_key=entry.object_key,
            tier=entry.tier,
            descriptor=entry.descriptor,
            state=entry.state,
            generation=entry.generation,
            last_access_ns=entry.last_access_ns,
            access_count=entry.access_count,
            active_readers=len(entry.leases),
            pinned=entry.pinned,
        )

    @staticmethod
    def _require_state(entry: _Entry, expected: ResidencyState) -> None:
        if entry.state != expected:
            raise RuntimeError(
                f"residency must be {expected.value}, got {entry.state.value}"
            )

    def _get_entry(self, residency_id: str) -> _Entry:
        try:
            return self._by_id[residency_id]
        except KeyError as error:
            raise KeyError("unknown residency") from error

    def _remove_entry(self, entry: _Entry) -> None:
        self._by_id.pop(entry.residency_id, None)
        entries = self._by_key.get(entry.object_key)
        if entries is None:
            return
        entries.pop(entry.residency_id, None)
        if not entries:
            self._by_key.pop(entry.object_key, None)
