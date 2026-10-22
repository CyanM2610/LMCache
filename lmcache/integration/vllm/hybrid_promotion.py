# SPDX-License-Identifier: Apache-2.0
"""Independent, terminally owned Promotion controls for one dense KV group."""

# Standard
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import uuid

# First Party
from lmcache.integration.vllm.kv_page_ownership import TerminalFuture

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import (
        BlockPool,
        ExternalPrefix,
        ExternalPrefixReservation,
    )

    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )


@dataclass(frozen=True)
class PromotionCommand:
    """Immutable scheduler -> worker control, separate from frontend requests."""

    control_id: str
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    num_cached_blocks: int
    block_size: int
    chunk_size: int


@dataclass(frozen=True)
class PromotionReceipt:
    """Worker terminal completion; false proves failure without pending DMA."""

    control_id: str
    success: bool


class PromotionCompletions:
    """Worker-only futures, never mixed into native request completion sets."""

    def __init__(self) -> None:
        """Create an empty terminal tracker with strong IPC-event ownership."""
        self._pending: dict[str, tuple[TerminalFuture | None, Any]] = {}

    @property
    def pending(self) -> bool:
        """Whether terminal status remains unreported, including immediate drops."""
        return bool(self._pending)

    def contains(self, control_id: str) -> bool:
        """Return whether this control already has an unreported submission."""
        return control_id in self._pending

    def record(
        self, control_id: str, future: TerminalFuture | None, keepalive: Any
    ) -> None:
        """Retain a future and IPC event, or a known never-submitted failure.

        Args:
            control_id: Independent control identity.
            future: Terminal future; None only when submission never occurred.
            keepalive: IPC event/operation storage needed until physical completion.
        Raises:
            ValueError: Duplicate still-pending control identity.
        """
        if control_id in self._pending:
            raise ValueError("duplicate promotion submission")
        self._pending[control_id] = (future, keepalive)

    def poll(self) -> tuple[PromotionReceipt, ...]:
        """Produce receipts only after terminal bool results; keep uncertain copies.

        Returns:
            New terminal receipts. No frontend request IDs are returned here.
        Raises:
            Exception: Query/result/transport error, retaining all unreported
                futures and their IPC events. A timeout is not a failed copy.
        """
        receipts = []
        for control_id, (future, _) in self._pending.items():
            if future is not None and not future.query():
                continue
            success = False if future is None else future.result(timeout=0)
            if not isinstance(success, bool):
                raise ValueError("promotion future lacks a terminal boolean result")
            receipts.append(PromotionReceipt(control_id, success))
        # Commit removals only after the whole poll succeeds, so a later error
        # cannot discard earlier receipts before the worker output is built.
        for receipt in receipts:
            del self._pending[receipt.control_id]
        return tuple(receipts)


@dataclass
class PendingPromotion:
    """Scheduler-thread ownership from source lookup through output adoption."""

    prefix: ExternalPrefix
    max_new_blocks: int
    reservation: ExternalPrefixReservation | None = None
    cancelled: bool = False
    settled: bool = False


class PromotionScheduler:
    """Coordinate native HBM ownership and independently leased host sources.

    ``prepare`` runs after native frontend allocation. Once metadata is captured,
    cancellation waits for a worker receipt even if the LOAD has not submitted
    yet. Only FIFO output adoption can publish a successful copy. Native request
    finish/abort callbacks do not operate on these control IDs or reservations.
    """

    def __init__(
        self,
        pool: BlockPool,
        adapter: LMCacheMPSchedulerAdapter,
        *,
        max_pending: int = 1,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Bind runtime owners; caller must enforce TP1/dense-group geometry.

        Args:
            pool: Native pool, accessed only from its scheduler thread.
            adapter: Terminal-protocol scheduler adapter with confirmed cleanup.
            max_pending: Bounded number of independent outstanding controls.
            event_sink: Optional synchronous diagnostic sink; must not raise.
        Raises:
            ValueError: Nonpositive pending bound.
        """
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._pool = pool
        self._adapter = adapter
        self._max_pending = max_pending
        self._event_sink = event_sink
        self._pending: dict[str, PendingPromotion] = {}

    @property
    def has_pending_work(self) -> bool:
        """Keep native empty worker iterations alive until receipt adoption."""
        return bool(self._pending)

    def enqueue(self, prefix: ExternalPrefix, max_new_blocks: int) -> str | None:
        """Lease a known host prefix without allocating or changing frontend work.

        Args:
            prefix: Immutable native snapshot; must end at a host chunk boundary.
            max_new_blocks: Positive HBM allocation limit for this control.
        Returns:
            Unique control ID, or None for a duplicate/full pending queue.
        Raises:
            ValueError: Invalid budget or host-chunk alignment.
            Exception: Transport failure; ownership remains cancellable/drainable.
        """
        chunk_size = self._adapter.lmcache_tokens_per_chunk
        if (
            max_new_blocks <= 0
            or not prefix.token_ids
            or chunk_size <= 0
            or len(prefix.token_ids) % chunk_size
        ):
            raise ValueError("promotion requires a positive budget and whole chunks")
        if len(self._pending) >= self._max_pending or any(
            item.prefix == prefix for item in self._pending.values()
        ):
            return None
        control_id = "lmcache-promotion-" + uuid.uuid4().hex
        self._pending[control_id] = PendingPromotion(prefix, max_new_blocks)
        self._adapter.maybe_submit_lookup_request(control_id, list(prefix.token_ids))
        self._emit("lookup_submitted", control_id, tokens=len(prefix.token_ids))
        return control_id

    def prepare(self) -> tuple[PromotionCommand, ...]:
        """Capture controls after source readiness and native capacity preflight.

        Returns:
            Newly captured commands, emitted exactly once. Missing host chunks,
            full HBM hits or insufficient capacity are rejected without eviction.
        Raises:
            Exception: Original lookup/allocation failure; ownership is retained.
        """
        commands = []
        for control_id, pending in tuple(self._pending.items()):
            if pending.settled:
                self._retire(control_id, "cleanup_retry")
                continue
            if pending.reservation is not None:
                continue
            matched = self._adapter.check_lookup_result(control_id)
            if matched is None:
                continue
            prefix = pending.prefix
            if matched < len(prefix.token_ids):
                self._retire(control_id, "host_miss")
                continue
            reservation = self._pool.reserve_external_prefix(
                prefix, pending.max_new_blocks
            )
            if reservation is None:
                self._retire(control_id, "hbm_hit_or_capacity")
                continue
            pending.reservation = reservation
            self._adapter.cleanup_lookup_result(control_id)
            commands.append(
                PromotionCommand(
                    control_id,
                    prefix.token_ids,
                    reservation.block_ids,
                    reservation.num_cached_blocks,
                    prefix.block_size,
                    self._adapter.lmcache_tokens_per_chunk,
                )
            )
            self._emit(
                "metadata_captured",
                control_id,
                block_ids=reservation.block_ids,
                num_cached_blocks=reservation.num_cached_blocks,
            )
        return tuple(commands)

    def cancel(self, control_id: str) -> None:
        """Cancel selection; captured metadata retains HBM until terminal adoption.

        Args:
            control_id: Existing control, or a retired ID (idempotent no-op).
        """
        pending = self._pending.get(control_id)
        if pending is None:
            return
        pending.cancelled = True
        self._emit("cancel_requested", control_id)
        if pending.reservation is None or pending.settled:
            self._retire(control_id, "cancel_before_capture")

    def adopt(self, receipt: PromotionReceipt) -> None:
        """Consume terminal output once; never use native finished_recving IDs.

        Args:
            receipt: TP1 worker response after physical terminal completion.
        Raises:
            ValueError: Unexpected/duplicate receipt, or lost HBM ownership.
        """
        pending = self._pending.get(receipt.control_id)
        if pending is None or pending.reservation is None or pending.settled:
            raise ValueError("unexpected or duplicate promotion receipt")
        publish = receipt.success and not pending.cancelled
        self._pool.finish_external_prefix(pending.reservation, publish=publish)
        pending.settled = True
        self._emit(
            "terminal_adopted",
            receipt.control_id,
            success=receipt.success,
            published=publish,
        )
        self._retire(receipt.control_id, "complete")

    def _retire(self, control_id: str, reason: str) -> None:
        self._adapter.end_session(control_id)
        self._adapter.cleanup_lookup_result(control_id)
        del self._pending[control_id]
        self._emit("retired", control_id, reason=reason)

    def _emit(self, event: str, control_id: str, **fields: Any) -> None:
        if self._event_sink is not None:
            self._event_sink({"event": event, "control_id": control_id, **fields})
