# SPDX-License-Identifier: Apache-2.0
"""OTel spans for HotPrefix control handlers."""

# Standard
from typing import Any
import os

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.hotprefix")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class HotPrefixTracingSubscriber(EventSubscriber):
    """Create one sampled span per HotPrefix control handler call."""

    def __init__(self) -> None:
        self._pending: dict[str, Any] = {}
        self._run_id = os.environ.get("HOTPREFIX_RUN_ID", "")

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.HOTPREFIX_CONTROL_START: self._on_start,
            EventType.HOTPREFIX_CONTROL_END: self._on_end,
        }

    def shutdown(self) -> None:
        """Close spans left open during EventBus shutdown."""
        for span in self._pending.values():
            span.end()
        self._pending.clear()

    def _on_start(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        method = str(event.metadata.get("method", "unknown"))
        span = _tracer.start_span(
            f"hotprefix.{method}",
            start_time=int(event.timestamp * 1e9),
        )
        span.set_attribute("hotprefix.method", method)
        span.set_attribute("hotprefix.operation_id", event.session_id)
        if self._run_id:
            span.set_attribute("hotprefix.run_id", self._run_id)
        self._pending[event.session_id] = span

    def _on_end(self, event: Event) -> None:
        span = self._pending.pop(event.session_id, None)
        if span is None:
            return
        for key in (
            "outcome",
            "request_bytes",
            "response_bytes",
            "duration_ns",
            "lock_wait_ns",
            "handler_body_ns",
        ):
            span.set_attribute(f"hotprefix.{key}", event.metadata.get(key, 0))
        span.end(end_time=int(event.timestamp * 1e9))
