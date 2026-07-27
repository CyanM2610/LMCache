# SPDX-License-Identifier: Apache-2.0
"""Compare-and-reserve transfer tickets for Gate E FETCH decisions."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Literal
import threading
import uuid

# Local
from .actions import FetchDecision, Tier
from .directory import MultiResidencyDirectory, Residency
from .residency import ReadLease


@dataclass(frozen=True)
class TransferTicket:
    """Generation-bound permission to execute one selected residency fetch."""

    op_id: str
    residency_id: str
    generation: int
    lease_id: str
    expires_at_ns: int
    estimated_completion_ns: int


@dataclass(frozen=True)
class BoundResidency:
    """Validated ticket and immutable source snapshot."""

    ticket: TransferTicket
    residency: Residency


@dataclass
class _TicketEntry:
    ticket: TransferTicket
    request_id: str
    lease: ReadLease
    tier: Tier
    queued_bytes: int
    terminal_outcome: str | None = None


class TicketManager:
    """Atomically bind advisory decisions to leases and queue reservations."""

    def __init__(
        self,
        directory: MultiResidencyDirectory,
        *,
        ticket_ttl_ns: int = 30_000_000_000,
    ) -> None:
        if ticket_ttl_ns <= 0:
            raise ValueError("ticket_ttl_ns must be positive")
        self._directory = directory
        self._ticket_ttl_ns = ticket_ttl_ns
        self._entries: dict[str, _TicketEntry] = {}
        self._queue_bytes: dict[Tier, int] = {"dram": 0, "cxl": 0}
        self._lock = threading.RLock()

    def bind_fetch(
        self, decision: FetchDecision, request_id: str, now_ns: int
    ) -> TransferTicket:
        """Compare state/generation and acquire a lease before returning."""
        if not request_id or now_ns < 0:
            raise ValueError("request_id and non-negative now_ns are required")
        with self._lock:
            try:
                residency = self._directory.get_residency(decision.residency_id)
                if residency.descriptor is None:
                    raise RuntimeError("selected residency has no descriptor")
                ttl = max(self._ticket_ttl_ns, decision.estimated_completion_ns)
                lease = self._directory.acquire_read(
                    residency.residency_id,
                    residency.generation,
                    ttl,
                    now_ns=now_ns,
                )
            except (KeyError, RuntimeError) as error:
                raise RuntimeError("FETCH decision could not bind") from error
            ticket = TransferTicket(
                op_id=uuid.uuid4().hex,
                residency_id=residency.residency_id,
                generation=residency.generation,
                lease_id=lease.lease_id,
                expires_at_ns=lease.expires_at_ns,
                estimated_completion_ns=decision.estimated_completion_ns,
            )
            queued_bytes = residency.descriptor.length
            self._entries[ticket.op_id] = _TicketEntry(
                ticket=ticket,
                request_id=request_id,
                lease=lease,
                tier=residency.tier,
                queued_bytes=queued_bytes,
            )
            self._queue_bytes[residency.tier] += queued_bytes
            return ticket

    def validate(self, ticket: TransferTicket, now_ns: int) -> BoundResidency:
        """Return the exact source while its ticket remains live."""
        with self._lock:
            entry = self._matching_entry(ticket)
            if entry.terminal_outcome is not None:
                raise RuntimeError("transfer ticket is terminal")
            if ticket.expires_at_ns <= now_ns:
                self._finish(entry, "expired")
                raise RuntimeError("transfer ticket expired")
            residency = self._directory.validate_lease(entry.lease, now_ns)
            return BoundResidency(ticket, residency)

    def complete(self, ticket: TransferTicket, outcome: Literal["ok", "error"]) -> None:
        """Release ticket resources once after data-path completion."""
        with self._lock:
            entry = self._matching_entry(ticket)
            if entry.terminal_outcome is not None:
                return
            self._finish(entry, outcome)

    def cancel(self, ticket: TransferTicket, reason: str) -> None:
        """Release a ticket after cancellation."""
        if not reason:
            raise ValueError("ticket cancellation reason must not be empty")
        with self._lock:
            entry = self._matching_entry(ticket)
            if entry.terminal_outcome is not None:
                return
            self._finish(entry, f"cancelled:{reason}")

    def expire(self, now_ns: int) -> tuple[TransferTicket, ...]:
        """Expire all live tickets whose bounded TTL elapsed."""
        expired: list[TransferTicket] = []
        with self._lock:
            for entry in self._entries.values():
                if (
                    entry.terminal_outcome is None
                    and entry.ticket.expires_at_ns <= now_ns
                ):
                    expired.append(entry.ticket)
                    self._finish(entry, "expired")
        return tuple(expired)

    def queue_bytes(self, tier: Tier) -> int:
        """Return bytes reserved by live tickets for one tier."""
        with self._lock:
            return self._queue_bytes[tier]

    def request_id(self, ticket: TransferTicket) -> str:
        """Return the request identity bound to a ticket."""
        with self._lock:
            return self._matching_entry(ticket).request_id

    def _matching_entry(self, ticket: TransferTicket) -> _TicketEntry:
        entry = self._entries.get(ticket.op_id)
        if entry is None or entry.ticket != ticket:
            raise RuntimeError("unknown transfer ticket")
        return entry

    def _finish(self, entry: _TicketEntry, outcome: str) -> None:
        entry.terminal_outcome = outcome
        self._queue_bytes[entry.tier] -= entry.queued_bytes
        self._directory.release_read(entry.lease.lease_id)
