# SPDX-License-Identifier: Apache-2.0
"""One bounded alternate-residency fallback before vLLM recompute."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

# Local
from .actions import FetchDecision, LookupDecision
from .tickets import TicketManager, TransferTicket


class LoadCancelled(RuntimeError):
    """Signal that a fetch was cancelled and must not retry."""


@dataclass(frozen=True)
class FallbackResult:
    """Terminal bounded-fallback result consumed by connector translation."""

    status: Literal["ok", "recompute", "cancelled"]
    attempts: int
    fallback_from: str | None
    fallback_to: str | None
    reason: str


class FallbackCoordinator:
    """Try an initial ticket and at most one freshly bound alternate."""

    def __init__(
        self,
        tickets: TicketManager,
        *,
        invalidate: Callable[[tuple[int, ...]], None],
    ) -> None:
        self._tickets = tickets
        self._invalidate = invalidate

    def execute(
        self,
        initial: TransferTicket,
        *,
        transfer: Callable[[TransferTicket, bool], None],
        choose_alternate: Callable[[frozenset[str], int], LookupDecision],
        destination_block_ids: tuple[int, ...],
        deadline_ns: int | None,
        now_ns: Callable[[], int],
    ) -> FallbackResult:
        """Fully overwrite HBM on each attempt; never expose partial KV."""
        first_bound = self._tickets.validate(initial, now_ns())
        first_tier = first_bound.residency.tier
        request_id = self._tickets.request_id(initial)
        first_error_text = ""
        try:
            transfer(initial, True)
        except LoadCancelled as error:
            self._tickets.cancel(initial, str(error))
            self._invalidate(destination_block_ids)
            return FallbackResult("cancelled", 1, None, None, str(error))
        except Exception as first_error:
            first_error_text = str(first_error) or type(first_error).__name__
            self._tickets.complete(initial, "error")
        else:
            self._tickets.complete(initial, "ok")
            return FallbackResult("ok", 1, None, None, "initial fetch succeeded")

        current_ns = now_ns()
        alternate = choose_alternate(
            frozenset({first_bound.residency.residency_id}), current_ns
        )
        if not isinstance(alternate, FetchDecision):
            self._invalidate(destination_block_ids)
            return FallbackResult("recompute", 1, first_tier, None, alternate.reason)
        if (
            deadline_ns is not None
            and current_ns + alternate.estimated_completion_ns > deadline_ns
        ):
            self._invalidate(destination_block_ids)
            return FallbackResult(
                "recompute", 1, first_tier, None, "alternate exceeds deadline"
            )
        if alternate.residency_id == first_bound.residency.residency_id:
            self._invalidate(destination_block_ids)
            return FallbackResult(
                "recompute", 1, first_tier, None, "alternate repeated failed source"
            )

        try:
            second = self._tickets.bind_fetch(alternate, request_id, current_ns)
            second_bound = self._tickets.validate(second, now_ns())
        except RuntimeError as error:
            self._invalidate(destination_block_ids)
            return FallbackResult("recompute", 1, first_tier, None, str(error))

        second_tier = second_bound.residency.tier
        try:
            transfer(second, True)
        except LoadCancelled as error:
            self._tickets.cancel(second, str(error))
            self._invalidate(destination_block_ids)
            return FallbackResult("cancelled", 2, first_tier, second_tier, str(error))
        except Exception as error:
            self._tickets.complete(second, "error")
            self._invalidate(destination_block_ids)
            return FallbackResult("recompute", 2, first_tier, second_tier, str(error))
        self._tickets.complete(second, "ok")
        return FallbackResult("ok", 2, first_tier, second_tier, first_error_text)
