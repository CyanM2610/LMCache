# SPDX-License-Identifier: Apache-2.0

"""Tests for HotPrefixMetricsSubscriber."""

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.metrics.hotprefix import (
    HotPrefixMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    read_tagged_counters,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import histogram_count


def test_control_end_records_three_duration_phases() -> None:
    subscriber = HotPrefixMetricsSubscriber()
    before = histogram_count("lmcache_mp.hotprefix_handler_duration")

    subscriber._on_control_end(
        Event(
            event_type=EventType.HOTPREFIX_CONTROL_END,
            metadata={
                "method": "access",
                "outcome": "success",
                "duration_ns": 10_000,
                "lock_wait_ns": 2_000,
                "handler_body_ns": 8_000,
            },
        )
    )

    assert histogram_count("lmcache_mp.hotprefix_handler_duration") - before == 3


def test_decision_reason_is_bounded() -> None:
    subscriber = HotPrefixMetricsSubscriber()
    before = read_tagged_counters()

    subscriber._on_decision(
        Event(
            event_type=EventType.HOTPREFIX_DECISION,
            metadata={
                "kind": "admission",
                "action": "reject",
                "reason": "exception text must not become a label",
            },
        )
    )
    delta = counter_delta(before, read_tagged_counters())

    assert (
        counter_value(
            delta,
            "lmcache_mp.hotprefix_decisions",
            kind="admission",
            action="reject",
            reason="other",
        )
        == 1
    )
