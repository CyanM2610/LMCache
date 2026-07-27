# SPDX-License-Identifier: Apache-2.0
"""Single-residency publication and read-lease lifecycle for Gates B-C."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
import uuid

# First Party
from lmcache.v1.distributed.api import ObjectKey

# Local
from .contracts import ExtentDescriptor
from .region_manager import CXLRegionManager


logger = logging.getLogger(__name__)


class ResidencyState(str, Enum):
    """Allowed states of one immutable tier residency."""

    FREE = "free"
    RESERVED = "reserved"
    WRITING = "writing"
    READY = "ready"
    EVICTING = "evicting"


@dataclass(frozen=True)
class Residency:
    """Immutable caller-facing snapshot of a single CXL residency."""

    residency_id: str
    object_key: ObjectKey
    descriptor: ExtentDescriptor | None
    state: ResidencyState
    generation: int


@dataclass(frozen=True)
class ReadLease:
    """Generation-bound permission to read one READY residency."""

    lease_id: str
    residency_id: str
    generation: int
    expires_at_ns: int


@dataclass
class _ResidencyEntry:
    residency_id: str
    object_key: ObjectKey
    reservation_id: str
    generation: int
    state: ResidencyState
    descriptor: ExtentDescriptor | None = None
    leases: set[str] = field(default_factory=set)


class SingleResidencyDirectory:
    """Map each ObjectKey to at most one CXL residency for Gates B-C."""

    def __init__(self, region_manager: CXLRegionManager) -> None:
        self._region_manager = region_manager
        self._by_key: dict[ObjectKey, _ResidencyEntry] = {}
        self._by_id: dict[str, _ResidencyEntry] = {}
        self._leases: dict[str, tuple[_ResidencyEntry, ReadLease]] = {}
        self._retired_leases: set[str] = set()
        self._lock = threading.RLock()

    def reserve_store(
        self, object_key: ObjectKey, *, length: int, alignment: int
    ) -> Residency:
        """Reserve the only allowed CXL residency for an object.

        Args:
            object_key: Logical packed KV object identity.
            length: Payload bytes to reserve.
            alignment: Required extent alignment.

        Returns:
            A RESERVED residency snapshot.

        Raises:
            RuntimeError: If the object already owns a residency.
        """
        with self._lock:
            if object_key in self._by_key:
                raise RuntimeError("ObjectKey already has a CXL residency")
            reservation = self._region_manager.reserve(length, alignment)
            entry = _ResidencyEntry(
                residency_id=uuid.uuid4().hex,
                object_key=object_key,
                reservation_id=reservation.reservation_id,
                generation=reservation.generation,
                state=ResidencyState.RESERVED,
            )
            self._by_key[object_key] = entry
            self._by_id[entry.residency_id] = entry
            return self._snapshot(entry)

    def mark_writing(self, residency_id: str) -> Residency:
        """Bind an extent and transition a RESERVED residency to WRITING.

        Args:
            residency_id: Identifier returned by :meth:`reserve_store`.

        Returns:
            A WRITING residency carrying its extent descriptor.

        Raises:
            KeyError: If the residency is unknown.
            RuntimeError: If the residency is not RESERVED.
        """
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.RESERVED)
            entry.descriptor = self._region_manager.begin_write(entry.reservation_id)
            entry.state = ResidencyState.WRITING
            return self._snapshot(entry)

    def publish(self, residency_id: str) -> Residency:
        """Publish a completed WRITING residency as immutable READY data.

        Args:
            residency_id: Identifier of the completed write.

        Returns:
            The READY residency snapshot.

        Raises:
            KeyError: If the residency is unknown.
            RuntimeError: If the residency is not WRITING.
        """
        with self._lock:
            entry = self._get_entry(residency_id)
            self._require_state(entry, ResidencyState.WRITING)
            entry.descriptor = self._region_manager.publish(entry.reservation_id)
            entry.state = ResidencyState.READY
            return self._snapshot(entry)

    def abort(self, residency_id: str, reason: str) -> None:
        """Abort an unpublished residency and release its extent.

        Args:
            residency_id: Identifier of the incomplete residency.
            reason: Non-empty diagnostic reason.

        Raises:
            KeyError: If the residency is unknown.
            ValueError: If the diagnostic reason is empty.
            RuntimeError: If the residency is already published or evicting.
        """
        with self._lock:
            entry = self._get_entry(residency_id)
            if entry.state not in (ResidencyState.RESERVED, ResidencyState.WRITING):
                raise RuntimeError(f"cannot abort residency in {entry.state.value}")
            self._region_manager.abort(entry.reservation_id, reason)
            self._remove_entry(entry)

    def lookup_ready(self, object_key: ObjectKey) -> Residency | None:
        """Return the READY residency for an object, if one is published.

        Args:
            object_key: Logical object identity.

        Returns:
            A READY snapshot, otherwise None.
        """
        with self._lock:
            entry = self._by_key.get(object_key)
            if entry is None or entry.state != ResidencyState.READY:
                return None
            return self._snapshot(entry)

    def acquire_read(
        self,
        object_key: ObjectKey,
        *,
        ttl_ns: int,
        now_ns: int | None = None,
    ) -> ReadLease:
        """Acquire a generation-bound read lease on READY data.

        Args:
            object_key: Logical object identity.
            ttl_ns: Positive lease duration in nanoseconds.
            now_ns: Optional deterministic clock value for tests.

        Returns:
            The immutable read lease.

        Raises:
            ValueError: If the TTL is not positive.
            KeyError: If no READY residency is available.
        """
        if ttl_ns <= 0:
            raise ValueError("lease TTL must be positive")
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            entry = self._by_key.get(object_key)
            if entry is None or entry.state != ResidencyState.READY:
                raise KeyError("ObjectKey has no READY residency")
            self._expire_leases(entry, now)
            lease = ReadLease(
                lease_id=uuid.uuid4().hex,
                residency_id=entry.residency_id,
                generation=entry.generation,
                expires_at_ns=now + ttl_ns,
            )
            entry.leases.add(lease.lease_id)
            self._leases[lease.lease_id] = (entry, lease)
            return lease

    def release_read(self, lease_id: str, *, now_ns: int | None = None) -> None:
        """Release a read lease exactly once; repeated releases are harmless.

        Args:
            lease_id: Identifier returned by :meth:`acquire_read`.
            now_ns: Optional deterministic clock value used to classify expiry.

        Raises:
            KeyError: If the lease identifier was never issued.
        """
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if lease_id in self._retired_leases:
                return
            pair = self._leases.pop(lease_id, None)
            if pair is None:
                raise KeyError("unknown read lease")
            entry, lease = pair
            entry.leases.discard(lease_id)
            self._retired_leases.add(lease_id)
            if lease.expires_at_ns <= now:
                logger.info(
                    "Released expired read lease %s for residency %s generation %d",
                    lease_id,
                    lease.residency_id,
                    lease.generation,
                )
            self._finish_evict_if_idle(entry)

    def evict(self, object_key: ObjectKey) -> None:
        """Block new readers and reclaim after all active readers release.

        Args:
            object_key: Logical object identity to evict.

        Raises:
            KeyError: If the object has no residency.
            RuntimeError: If the residency is not READY.
        """
        with self._lock:
            entry = self._by_key.get(object_key)
            if entry is None:
                raise KeyError("ObjectKey has no residency")
            self._require_state(entry, ResidencyState.READY)
            if entry.descriptor is None:
                raise RuntimeError("READY residency has no extent descriptor")
            self._region_manager.begin_evict(entry.descriptor)
            entry.state = ResidencyState.EVICTING
            self._finish_evict_if_idle(entry)

    @staticmethod
    def _snapshot(entry: _ResidencyEntry) -> Residency:
        return Residency(
            residency_id=entry.residency_id,
            object_key=entry.object_key,
            descriptor=entry.descriptor,
            state=entry.state,
            generation=entry.generation,
        )

    @staticmethod
    def _require_state(entry: _ResidencyEntry, required: ResidencyState) -> None:
        if entry.state != required:
            raise RuntimeError(
                f"residency must be {required.value}, got {entry.state.value}"
            )

    def _get_entry(self, residency_id: str) -> _ResidencyEntry:
        try:
            return self._by_id[residency_id]
        except KeyError as error:
            raise KeyError("unknown residency") from error

    def _expire_leases(self, entry: _ResidencyEntry, now_ns: int) -> None:
        expired = [
            lease_id
            for lease_id in entry.leases
            if self._leases[lease_id][1].expires_at_ns <= now_ns
        ]
        for lease_id in expired:
            self._leases.pop(lease_id, None)
            entry.leases.discard(lease_id)
            self._retired_leases.add(lease_id)

    def _finish_evict_if_idle(self, entry: _ResidencyEntry) -> None:
        if entry.state != ResidencyState.EVICTING or entry.leases:
            return
        if entry.descriptor is None:
            raise RuntimeError("EVICTING residency has no extent descriptor")
        self._region_manager.reclaim(entry.descriptor)
        self._remove_entry(entry)

    def _remove_entry(self, entry: _ResidencyEntry) -> None:
        self._by_key.pop(entry.object_key, None)
        self._by_id.pop(entry.residency_id, None)
