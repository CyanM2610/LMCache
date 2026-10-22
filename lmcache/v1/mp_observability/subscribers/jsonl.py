# SPDX-License-Identifier: Apache-2.0
"""Opt-in loss-auditable raw events, using occurrence time from the event bus."""

# Standard
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json
import os
import time

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)
from lmcache.v1.mp_observability.host_phase import host_phase_errors


def encode_value(value: Any) -> Any:
    """Encode identities/layouts without losing byte hashes or dataclass fields.

    Args:
        value: A metadata value passed by JSON's default encoder.
    Returns:
        JSON-compatible typed representation; opaque values retain type and repr.
    """
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return {"type": type(value).__qualname__, "repr": repr(value)}


class JsonlSubscriber(EventSubscriber):
    """Write all bus events and final drop/error counts on the drain thread.

    This trace is diagnostic. File I/O never occurs in the publisher, but full
    event collection can perturb timings and must be compared with trace-off.
    """

    def __init__(self, path: str, bus: EventBus) -> None:
        """Open a fresh output, refusing to truncate previous evidence.

        Args:
            path: Output JSONL file (parent must already exist).
            bus: Bus whose drop and callback-error counters will be recorded.
        Raises:
            OSError: If path exists or cannot be created.
        """
        self.bus = bus
        self.output = Path(path).open("x", buffering=1)
        self.count = 0
        self._write(
            {
                "record": "header",
                "schema_version": 1,
                "pid": os.getpid(),
                "clock": "unix_seconds_float64",
                "clock_anchor_wall_ns": time.time_ns(),
                "clock_anchor_monotonic_ns": time.monotonic_ns(),
                "timestamp_semantics": "bus publication or native stream host callback",
                "absolute_dma_timestamps": None,
                "absolute_dma_null_reason": "host callbacks do not give DMA boundaries",
            }
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return subscriptions for every defined event type, without sampling."""
        return {event_type: self.on_event for event_type in EventType}

    def on_event(self, event: Event) -> None:
        """Serialize event identity, metadata and its original occurrence time.

        Args:
            event: Event delivered by the bus, already timestamped at source.
        Raises:
            OSError: If output fails; EventBus retains callback-error counts.
        """
        self._write(
            {
                "record": "event",
                "sequence": self.count,
                "event": event.event_type.value,
                "timestamp_s": event.timestamp,
                "session_id": event.session_id,
                "metadata": event.metadata,
            }
        )
        self.count += 1

    def shutdown(self) -> None:
        """Flush a terminal accounting record after the bus's final drain."""
        if self.output.closed:
            return
        try:
            self._write(
                {
                    "record": "footer",
                    "events_written": self.count,
                    "bus_dropped_events": self.bus.dropped_events_count(),
                    "subscriber_errors": self.bus.subscriber_exception_counts(),
                    "bus_queue_depth": self.bus.queue_depth(),
                    "host_phase_errors": host_phase_errors(),
                }
            )
        finally:
            self.output.close()

    def _write(self, record: dict[str, Any]) -> None:
        self.output.write(json.dumps(record, default=encode_value) + "\n")
