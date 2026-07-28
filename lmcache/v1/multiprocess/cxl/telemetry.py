# SPDX-License-Identifier: Apache-2.0
"""Pointer-free Gate E JSONL evidence and bounded-cardinality aggregates."""

# Future
from __future__ import annotations

# Standard
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
import json
import logging
import threading

# Local
from .actions import Tier


PolicyEventType = Literal[
    "store",
    "retrieve",
    "lookup_decision",
    "ticket",
    "fallback",
    "load_error",
]
PolicyAction = Literal["store", "fetch", "recompute", "evict", "cancel"]
ReasonCode = Literal[
    "minimum_cost",
    "baseline",
    "no_candidate",
    "layout_rejection",
    "deadline",
    "ticket_conflict",
    "ticket_expiry",
    "transfer_error",
    "cancelled",
    "required_target_error",
    "optional_target_error",
]


@dataclass(frozen=True)
class PolicyEvent:
    """Complete high-cardinality event retained only in JSONL evidence."""

    timestamp_ns: int
    run_id: str
    op_id: str
    request_id: str
    object_id: str
    residency_id: str | None
    instance_id: int
    event: PolicyEventType
    tier: Tier | None
    path: str
    payload_bytes: int
    tokens: int
    state: str | None
    generation: int | None
    layout_fingerprint: str | None
    candidate_estimates_ns: tuple[tuple[str, int], ...]
    chosen_action: PolicyAction
    reason_code: ReasonCode
    reason: str
    queue_ns: int | None
    cuda_estimate_ns: int | None
    modeled_estimate_ns: int | None
    recompute_estimate_ns: int | None
    cuda_actual_ns: int | None
    modeled_actual_ns: int | None
    effective_actual_ns: int | None
    ticket_id: str | None
    fallback_from: Tier | None
    fallback_to: Tier | None
    required: bool | None
    partial_success: bool
    invalidated_blocks: int
    terminal_error: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported policy event schema")
        if not all(
            (self.run_id, self.op_id, self.request_id, self.object_id, self.path)
        ):
            raise ValueError("policy event identities and path must not be empty")
        if (
            min(
                self.timestamp_ns,
                self.instance_id,
                self.payload_bytes,
                self.tokens,
                self.invalidated_blocks,
            )
            < 0
        ):
            raise ValueError("policy event counters must be non-negative")
        if not self.reason:
            raise ValueError("policy event reason must not be empty")
        if self.terminal_error == "":
            raise ValueError("terminal_error must be non-empty when supplied")
        times = (
            self.queue_ns,
            self.cuda_estimate_ns,
            self.modeled_estimate_ns,
            self.recompute_estimate_ns,
            self.cuda_actual_ns,
            self.modeled_actual_ns,
            self.effective_actual_ns,
        )
        if any(value is not None and value < 0 for value in times):
            raise ValueError("policy event times must be non-negative")
        if any(value < 0 for _, value in self.candidate_estimates_ns):
            raise ValueError("candidate estimates must be non-negative")
        names = [name for name, _ in self.candidate_estimates_ns]
        if len(names) != len(set(names)):
            raise ValueError("candidate estimate names must be unique")
        if self.chosen_action == "fetch" and not self.ticket_id:
            raise ValueError("positive fetch event requires a ticket_id")

    def to_primitive(self) -> dict[str, Any]:
        """Return recursively JSON-compatible evidence fields."""
        value = asdict(self)
        value["candidate_estimates_ns"] = dict(self.candidate_estimates_ns)
        return value


class PolicyEventSink(Protocol):
    """Existing-observability extension point for policy evidence."""

    def record(self, event: PolicyEvent) -> None: ...


class InMemoryPolicyEventSink:
    """Bounded event sink used by deterministic tests and proof runs."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._events: deque[PolicyEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, event: PolicyEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[PolicyEvent, ...]:
        with self._lock:
            return tuple(self._events)


class JSONLPolicyEventSink:
    """Append crash-local, high-cardinality policy evidence as JSONL."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: PolicyEvent) -> None:
        line = json.dumps(event.to_primitive(), sort_keys=True) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as output:
            output.write(line)


class LoggingPolicyEventSink:
    """Emit policy JSON through the existing LMCache logging pipeline."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def record(self, event: PolicyEvent) -> None:
        self._logger.debug(
            "gate_e_policy=%s",
            json.dumps(event.to_primitive(), sort_keys=True),
        )


class PolicyMetrics:
    """In-process aggregates with only bounded labels suitable for export."""

    def __init__(self) -> None:
        self._event_count: Counter[str] = Counter()
        self._tier_operations: Counter[str] = Counter()
        self._decisions: Counter[str] = Counter()
        self._external_tokens = 0
        self._recomputed_tokens = 0
        self._queue_bytes: dict[str, int] = {}
        self._latencies: dict[str, deque[int]] = {
            "cuda": deque(maxlen=4096),
            "modeled": deque(maxlen=4096),
            "effective": deque(maxlen=4096),
        }
        self._fallback_count = 0
        self._ticket_conflicts = 0
        self._ticket_expiries = 0
        self._partial_success = 0
        self._layout_rejections = 0
        self._invalidated_blocks = 0
        self._lock = threading.Lock()

    def record(self, event: PolicyEvent) -> None:
        with self._lock:
            self._event_count[event.event] += 1
            if event.event in ("store", "retrieve") and event.tier is not None:
                self._tier_operations[f"{event.event}:{event.tier}"] += 1
            if event.event == "lookup_decision":
                self._decisions[f"{event.chosen_action}:{event.reason_code}"] += 1
            if event.chosen_action == "fetch":
                self._external_tokens += event.tokens
            elif event.chosen_action == "recompute":
                self._recomputed_tokens += event.tokens
            if event.tier is not None and event.payload_bytes:
                self._queue_bytes[event.tier] = event.payload_bytes
            for name, value in (
                ("cuda", event.cuda_actual_ns),
                ("modeled", event.modeled_actual_ns),
                ("effective", event.effective_actual_ns),
            ):
                if value is not None:
                    self._latencies[name].append(value)
            self._fallback_count += int(event.event == "fallback")
            self._ticket_conflicts += int(event.reason_code == "ticket_conflict")
            self._ticket_expiries += int(event.reason_code == "ticket_expiry")
            self._partial_success += int(event.partial_success)
            self._layout_rejections += int(event.reason_code == "layout_rejection")
            self._invalidated_blocks += event.invalidated_blocks

    def snapshot(self) -> dict[str, Any]:
        """Return Prometheus-safe aggregate values without unbounded IDs."""
        with self._lock:
            return {
                "event_count": dict(self._event_count),
                "tier_operation_count": dict(self._tier_operations),
                "decision_count": dict(self._decisions),
                "external_tokens": self._external_tokens,
                "recomputed_tokens": self._recomputed_tokens,
                "queue_bytes": dict(self._queue_bytes),
                "latency_ns": {
                    name: tuple(values) for name, values in self._latencies.items()
                },
                "fallback_count": self._fallback_count,
                "ticket_conflict_count": self._ticket_conflicts,
                "ticket_expiry_count": self._ticket_expiries,
                "partial_success_count": self._partial_success,
                "layout_rejection_count": self._layout_rejections,
                "invalidated_blocks": self._invalidated_blocks,
            }
