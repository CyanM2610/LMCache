# SPDX-License-Identifier: Apache-2.0
"""Aggregate metrics for HotPrefix control and residency events."""

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

_REASONS = {
    "none",
    "prefix_absent",
    "ready_residency",
    "frequency_below_threshold",
    "candidate_exceeds_host_capacity",
    "capacity_available",
    "insufficient_reclaimable_capacity",
    "not_hotter_than_replacement",
    "replace_colder_residency",
}


class HotPrefixMetricsSubscriber(EventSubscriber):
    """Consume bounded HotPrefix EventBus events into OTel metrics."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.hotprefix")
        self._handler_duration = meter.create_histogram(
            "lmcache_mp.hotprefix_handler_duration",
            description=(
                "HotPrefix handler milliseconds split by total, lock wait, and body."
            ),
            unit="ms",
        )
        self._decisions = meter.create_counter(
            "lmcache_mp.hotprefix_decisions",
            description="HotPrefix Global policy decisions.",
            unit="decisions",
        )
        self._residency_changes = meter.create_counter(
            "lmcache_mp.hotprefix_residency_changes",
            description="HotPrefix logical residency state transitions.",
            unit="changes",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.HOTPREFIX_CONTROL_END: self._on_control_end,
            EventType.HOTPREFIX_DECISION: self._on_decision,
            EventType.HOTPREFIX_RESIDENCY_CHANGED: self._on_residency_changed,
        }

    def _on_control_end(self, event: Event) -> None:
        method = str(event.metadata.get("method", "unknown"))
        outcome = str(event.metadata.get("outcome", "failure"))
        for phase, key in (
            ("total", "duration_ns"),
            ("lock_wait", "lock_wait_ns"),
            ("handler_body", "handler_body_ns"),
        ):
            self._handler_duration.record(
                int(event.metadata.get(key, 0)) / 1e6,
                attributes={
                    "method": method,
                    "outcome": outcome,
                    "phase": phase,
                },
            )

    def _on_decision(self, event: Event) -> None:
        reason = str(event.metadata.get("reason", "none"))
        self._decisions.add(
            1,
            attributes={
                "kind": str(event.metadata.get("kind", "unknown")),
                "action": str(event.metadata.get("action", "unknown")),
                "reason": reason if reason in _REASONS else "other",
            },
        )

    def _on_residency_changed(self, event: Event) -> None:
        self._residency_changes.add(
            1,
            attributes={
                "old_state": str(event.metadata.get("old_state", "unknown")),
                "new_state": str(event.metadata.get("new_state", "unknown")),
            },
        )
