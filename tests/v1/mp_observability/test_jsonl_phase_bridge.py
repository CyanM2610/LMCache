# SPDX-License-Identifier: Apache-2.0
"""JSONL-only observation must export completed native transfer sections."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import json

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.config import ObservabilityConfig, init_observability
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from lmcache.v1.mp_observability.subscribers import transfer_phase_sampler as sampler


def test_jsonl_only_exports_phase_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public config exports samples and loss counts without metrics/tracing."""
    samples = [(0, 1, 0, 0.25, 4096, "transfer-7", 1.0, 1.00025)]
    monkeypatch.setattr(sampler, "_HAS_TRANSFER_PHASE_TIMING", True)
    monkeypatch.setattr(
        sampler,
        "_device_ops",
        SimpleNamespace(pop_completed_phase_timings=lambda: samples),
    )
    path = tmp_path / "events.jsonl"
    bus = init_observability(
        ObservabilityConfig(
            metrics_enabled=False,
            tracing_enabled=False,
            logging_enabled=False,
            event_jsonl=str(path),
        ),
        start_prometheus_http_server=False,
    )
    try:
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                metadata={"transfer_key": "transfer-7"},
            )
        )
        bus.stop()
    finally:
        init_event_bus(EventBusConfig(enabled=False))
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    phases = [r for r in rows if r.get("event") == "mp.transfer.phase_samples"]
    assert len(phases) == 1
    assert phases[0]["metadata"]["samples"] == [list(samples[0])]
    assert phases[0]["metadata"]["ended_transfer_key"] == "transfer-7"
    assert rows[-1]["bus_dropped_events"] == rows[-1]["bus_queue_depth"] == 0
    assert not rows[-1]["subscriber_errors"]


def test_host_phase_nesting_and_publish_failure_preserve_native_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inclusive host spans retain parents and audit diagnostic publication errors."""
    # First Party
    from lmcache.v1.mp_observability import host_phase

    events: list[Event] = []
    monkeypatch.setenv("L2_HYBRID_PHASE_TIMING", "1")
    monkeypatch.setattr(
        host_phase, "get_event_bus", lambda: SimpleNamespace(publish=events.append)
    )

    @host_phase.observed_mp_call("inner")
    def inner() -> int:
        return 42

    @host_phase.observed_mp_call("outer")
    def outer() -> int:
        return inner()

    assert outer() == 42
    child, parent = [event.metadata for event in events]
    assert child["parent_span_id"] == parent["span_id"]
    assert child["begin_wall_ns"] >= parent["begin_wall_ns"]
    assert child["end_wall_ns"] <= parent["end_wall_ns"]

    def broken(event: Event) -> None:
        raise RuntimeError("publication failed")

    monkeypatch.setattr(
        host_phase, "get_event_bus", lambda: SimpleNamespace(publish=broken)
    )
    before = len(host_phase.host_phase_errors())
    assert inner() == 42
    assert len(host_phase.host_phase_errors()) == before + 1
