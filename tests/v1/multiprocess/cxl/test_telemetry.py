# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.telemetry import (
    InMemoryPolicyEventSink,
    PolicyEvent,
    PolicyMetrics,
)


pytestmark = pytest.mark.no_shared_allocator


def _event(**changes) -> PolicyEvent:
    values = {
        "timestamp_ns": 10,
        "run_id": "run",
        "op_id": "op",
        "request_id": "request",
        "object_id": "sha256:abc",
        "residency_id": "residency",
        "instance_id": 1,
        "event": "lookup_decision",
        "tier": "cxl",
        "path": "cxl_direct",
        "payload_bytes": 4096,
        "tokens": 256,
        "state": "ready",
        "generation": 2,
        "layout_fingerprint": "a" * 64,
        "candidate_estimates_ns": (("cxl", 100), ("recompute", 200)),
        "chosen_action": "fetch",
        "reason_code": "minimum_cost",
        "reason": "structured reason",
        "queue_ns": 10,
        "cuda_estimate_ns": 90,
        "modeled_estimate_ns": 100,
        "recompute_estimate_ns": 200,
        "cuda_actual_ns": 95,
        "modeled_actual_ns": 105,
        "effective_actual_ns": 105,
        "ticket_id": "ticket",
        "fallback_from": None,
        "fallback_to": None,
        "required": True,
        "partial_success": False,
        "invalidated_blocks": 0,
        "terminal_error": None,
    }
    values.update(changes)
    return PolicyEvent(**values)


def test_json_event_preserves_high_cardinality_evidence_without_payloads() -> None:
    sink = InMemoryPolicyEventSink()
    event = _event()
    sink.record(event)
    primitive = sink.snapshot()[0].to_primitive()

    assert primitive["request_id"] == "request"
    assert primitive["candidate_estimates_ns"] == {"cxl": 100, "recompute": 200}
    serialized = json.dumps(primitive)
    for forbidden in ("prompt", "token_ids", "pointer", "cuda_event", "block_ids"):
        assert forbidden not in serialized


def test_aggregate_metrics_exclude_unbounded_identifiers() -> None:
    metrics = PolicyMetrics()
    metrics.record(_event())
    metrics.record(
        _event(
            event="fallback",
            chosen_action="recompute",
            reason_code="transfer_error",
            fallback_from="cxl",
            fallback_to="dram",
            tokens=0,
            invalidated_blocks=4,
            terminal_error="injected",
        )
    )

    snapshot = metrics.snapshot()
    serialized = json.dumps(snapshot, sort_keys=True)
    assert snapshot["decision_count"]["fetch:minimum_cost"] == 1
    assert snapshot["fallback_count"] == 1
    assert snapshot["invalidated_blocks"] == 4
    for forbidden in (
        "request",
        "residency",
        "sha256:abc",
        "structured reason",
    ):
        assert forbidden not in serialized


def test_event_rejects_inconsistent_or_sensitive_fields() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _event(payload_bytes=-1)
    with pytest.raises(ValueError, match="terminal_error"):
        _event(chosen_action="fetch", terminal_error="")
