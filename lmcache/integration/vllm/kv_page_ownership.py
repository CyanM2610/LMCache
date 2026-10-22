# SPDX-License-Identifier: Apache-2.0
"""Order physical KV page reuse after conflicting remote transfers finish."""

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import time

Page = tuple[int, int]


class TerminalFuture(Protocol):
    """Completion covers remote physical access, including its device event."""

    def query(self) -> bool:
        """Return whether a terminal response and device completion are ready."""
        ...

    def result(self, timeout: float | None = None) -> Any:
        """Wait for terminal completion; raise if it cannot be established."""
        ...


def pages(block_ids: Mapping[int, Sequence[int]]) -> frozenset[Page]:
    """Snapshot physical identities without conflating different engine groups."""
    return frozenset(
        (group, block) for group, blocks in block_ids.items() for block in blocks
    )


@dataclass(frozen=True)
class PendingAccess:
    """One submitted transfer; request IDs do not determine physical ownership."""

    future: TerminalFuture
    pages: frozenset[Page]
    writes: bool


class PendingKVPageAccess:
    """Worker-thread-owned hazards, retained until proven physical completion.

    Reclamation and scheduling remain native. Before submitting a conflicting
    read/write, the worker waits only for transfers touching those actual pages.
    Timeout or an ambiguous transport exception retains the obligation and
    prevents the new access. A terminal failed copy can safely release a hazard;
    its original future remains available to normal request error handling.
    """

    def __init__(self) -> None:
        self._pending: list[PendingAccess] = []

    @property
    def pending_count(self) -> int:
        """Return retained transfer obligations, including unobserved completions."""
        return len(self._pending)

    def record(
        self,
        future: TerminalFuture,
        block_ids: Mapping[int, Sequence[int]],
        *,
        writes: bool,
    ) -> None:
        """Retain a submitted transfer and immutable engine-group/page identities."""
        self._pending.append(PendingAccess(future, pages(block_ids), writes))

    def reap(self) -> None:
        """Discard observed terminal hazards; preserve ambiguous failures.

        Raises:
            Exception: If a ready future cannot establish terminal completion.
        """
        for access in tuple(self._pending):
            if access.future.query():
                access.future.result(timeout=0)
                self._pending.remove(access)

    def wait(
        self,
        read_blocks: Mapping[int, Sequence[int]],
        write_blocks: Mapping[int, Sequence[int]],
        *,
        timeout: float,
    ) -> None:
        """Order this access after conflicting submitted transfers only.

        Args:
            read_blocks: Pages the new operation will read, per engine group.
            write_blocks: Pages the new operation may overwrite, per group.
            timeout: Total maximum host wait in seconds for this access.
        Raises:
            TimeoutError: If physical completion is not observed in time.
            Exception: Original future failure; its protection is retained.
        """
        reads, writes = pages(read_blocks), pages(write_blocks)
        deadline = time.monotonic() + timeout
        for access in tuple(self._pending):
            conflict = access.pages & (reads | writes if access.writes else writes)
            if not conflict:
                continue
            access.future.result(timeout=max(0.0, deadline - time.monotonic()))
            self._pending.remove(access)
