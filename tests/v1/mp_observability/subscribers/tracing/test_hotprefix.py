# SPDX-License-Identifier: Apache-2.0

"""Tests for HotPrefixTracingSubscriber."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.tracing.hotprefix import (
    HotPrefixTracingSubscriber,
)
import lmcache.v1.mp_observability.subscribers.tracing.hotprefix as tracing_module


def test_control_events_form_one_terminal_span(monkeypatch) -> None:
    tracer = MagicMock()
    span = tracer.start_span.return_value
    monkeypatch.setattr(tracing_module, "_HAS_OTEL", True)
    monkeypatch.setattr(tracing_module, "_tracer", tracer)
    subscriber = HotPrefixTracingSubscriber()

    subscriber._on_start(
        Event(
            event_type=EventType.HOTPREFIX_CONTROL_START,
            session_id="operation-1",
            timestamp=1.0,
            metadata={"method": "access"},
        )
    )
    subscriber._on_end(
        Event(
            event_type=EventType.HOTPREFIX_CONTROL_END,
            session_id="operation-1",
            timestamp=2.0,
            metadata={"outcome": "success", "duration_ns": 10},
        )
    )

    tracer.start_span.assert_called_once()
    span.end.assert_called_once_with(end_time=2_000_000_000)
    assert not subscriber._pending
