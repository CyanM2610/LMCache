# SPDX-License-Identifier: Apache-2.0
"""Extent allocation and generation management for one CXL proxy region."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
import threading
import uuid

# Local
from .contracts import ExtentDescriptor
from .region_provider import RegionHandle


class _ExtentState(str, Enum):
    RESERVED = "RESERVED"
    WRITING = "WRITING"
    READY = "READY"
    EVICTING = "EVICTING"


@dataclass(frozen=True)
class ExtentReservation:
    """Allocator reservation before an extent becomes writable."""

    reservation_id: str
    offset: int
    length: int
    alignment: int
    generation: int


@dataclass
class _Allocation:
    reservation: ExtentReservation
    descriptor: ExtentDescriptor
    state: _ExtentState


class CXLRegionManager:
    """Sole allocator and lifecycle owner for extents in one region."""

    def __init__(
        self,
        handle: RegionHandle,
        *,
        layout_id: str,
        layout_fingerprint: str,
    ) -> None:
        self._handle = handle
        self._layout_id = layout_id
        self._layout_fingerprint = layout_fingerprint
        self._free: list[tuple[int, int]] = [(0, handle.capacity)]
        self._allocations: dict[str, _Allocation] = {}
        self._allocation_by_offset: dict[int, _Allocation] = {}
        self._generation_by_offset: dict[int, int] = {}
        self._lock = threading.RLock()

    def reserve(self, length: int, alignment: int) -> ExtentReservation:
        """Reserve the deterministic best-fit aligned free range.

        Args:
            length: Required payload bytes.
            alignment: Required power-of-two extent alignment.

        Returns:
            An immutable reservation carrying the new generation.

        Raises:
            ValueError: If length or alignment is invalid.
            MemoryError: If no free extent can satisfy the request.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a power of two")
        effective_alignment = max(alignment, self._handle.alignment)
        with self._lock:
            candidates: list[tuple[int, int, int, int]] = []
            for index, (free_offset, free_length) in enumerate(self._free):
                aligned_offset = self._align_up(free_offset, effective_alignment)
                consumed = aligned_offset - free_offset + length
                if consumed <= free_length:
                    candidates.append(
                        (free_length - consumed, aligned_offset, index, consumed)
                    )
            if not candidates:
                raise MemoryError("CXL region capacity is exhausted")
            _, offset, free_index, consumed = min(candidates)
            free_offset, free_length = self._free.pop(free_index)
            prefix = offset - free_offset
            suffix = free_length - consumed
            if prefix:
                self._free.append((free_offset, prefix))
            if suffix:
                self._free.append((offset + length, suffix))
            self._free.sort()

            generation = self._generation_by_offset.get(offset, 0) + 1
            self._generation_by_offset[offset] = generation
            reservation = ExtentReservation(
                reservation_id=uuid.uuid4().hex,
                offset=offset,
                length=length,
                alignment=effective_alignment,
                generation=generation,
            )
            descriptor = ExtentDescriptor(
                region_id=self._handle.region_id,
                offset=offset,
                length=length,
                generation=generation,
                tier="cxl",
                layout_id=self._layout_id,
                layout_fingerprint=self._layout_fingerprint,
            )
            allocation = _Allocation(
                reservation=reservation,
                descriptor=descriptor,
                state=_ExtentState.RESERVED,
            )
            self._allocations[reservation.reservation_id] = allocation
            self._allocation_by_offset[offset] = allocation
            return reservation

    def begin_write(self, reservation_id: str) -> ExtentDescriptor:
        """Transition a reservation from RESERVED to WRITING.

        Args:
            reservation_id: Identifier returned by :meth:`reserve`.

        Returns:
            The descriptor to bind into a transfer plan.

        Raises:
            ValueError: If the reservation is unknown or stale.
            RuntimeError: If the reservation is not RESERVED.
        """
        with self._lock:
            allocation = self._get_reservation(reservation_id)
            self._require_state(allocation, _ExtentState.RESERVED)
            allocation.state = _ExtentState.WRITING
            return allocation.descriptor

    def publish(self, reservation_id: str) -> ExtentDescriptor:
        """Publish a successfully written extent as immutable READY data.

        Args:
            reservation_id: Identifier of a WRITING reservation.

        Returns:
            The immutable READY extent descriptor.

        Raises:
            ValueError: If the reservation is unknown or stale.
            RuntimeError: If the reservation is not WRITING.
        """
        with self._lock:
            allocation = self._get_reservation(reservation_id)
            self._require_state(allocation, _ExtentState.WRITING)
            allocation.state = _ExtentState.READY
            return allocation.descriptor

    def abort(self, reservation_id: str, reason: str) -> None:
        """Abort a RESERVED or WRITING allocation and reclaim its bytes.

        Args:
            reservation_id: Identifier of the incomplete allocation.
            reason: Non-empty diagnostic reason for the abort.

        Raises:
            ValueError: If the reason is empty or reservation is stale.
            RuntimeError: If the allocation is already published or evicting.
        """
        if not reason:
            raise ValueError("abort reason must not be empty")
        with self._lock:
            allocation = self._get_reservation(reservation_id)
            if allocation.state not in (_ExtentState.RESERVED, _ExtentState.WRITING):
                raise RuntimeError(
                    f"cannot abort extent in {allocation.state.value} state"
                )
            self._release(allocation)

    def begin_evict(self, descriptor: ExtentDescriptor) -> None:
        """Make a READY descriptor unavailable for new users.

        Args:
            descriptor: Exact current-generation READY descriptor.

        Raises:
            ValueError: If the descriptor is unknown or stale.
            RuntimeError: If the extent is not READY.
        """
        with self._lock:
            allocation = self._resolve_descriptor(descriptor)
            self._require_state(allocation, _ExtentState.READY)
            allocation.state = _ExtentState.EVICTING

    def reclaim(self, descriptor: ExtentDescriptor) -> None:
        """Return an EVICTING extent to the allocator.

        Args:
            descriptor: Exact current-generation EVICTING descriptor.

        Raises:
            ValueError: If the descriptor is unknown or stale.
            RuntimeError: If the extent is not EVICTING.
        """
        with self._lock:
            allocation = self._resolve_descriptor(descriptor)
            self._require_state(allocation, _ExtentState.EVICTING)
            self._release(allocation)

    def validate_descriptor(self, descriptor: ExtentDescriptor) -> None:
        """Validate that a descriptor is current and executable.

        Args:
            descriptor: Exact current-generation WRITING or READY descriptor.

        Raises:
            ValueError: If the descriptor is unknown or stale.
            RuntimeError: If its lifecycle state cannot execute a transfer.
        """
        with self._lock:
            allocation = self._resolve_descriptor(descriptor)
            if allocation.state not in (_ExtentState.WRITING, _ExtentState.READY):
                raise RuntimeError(
                    f"extent is not executable in {allocation.state.value} state"
                )

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) & -alignment

    @staticmethod
    def _require_state(allocation: _Allocation, required: _ExtentState) -> None:
        if allocation.state != required:
            raise RuntimeError(
                f"extent must be {required.value}, got {allocation.state.value}"
            )

    def _get_reservation(self, reservation_id: str) -> _Allocation:
        try:
            return self._allocations[reservation_id]
        except KeyError as error:
            raise ValueError("unknown or stale reservation") from error

    def _resolve_descriptor(self, descriptor: ExtentDescriptor) -> _Allocation:
        allocation = self._allocation_by_offset.get(descriptor.offset)
        if allocation is None or allocation.descriptor != descriptor:
            raise ValueError("unknown or stale extent descriptor")
        return allocation

    def _release(self, allocation: _Allocation) -> None:
        reservation = allocation.reservation
        self._allocations.pop(reservation.reservation_id, None)
        self._allocation_by_offset.pop(reservation.offset, None)
        self._free.append((reservation.offset, reservation.length))
        self._free.sort()
        coalesced: list[tuple[int, int]] = []
        for offset, length in self._free:
            if coalesced and coalesced[-1][0] + coalesced[-1][1] == offset:
                previous_offset, previous_length = coalesced[-1]
                coalesced[-1] = (previous_offset, previous_length + length)
            else:
                coalesced.append((offset, length))
        self._free = coalesced
