# SPDX-License-Identifier: Apache-2.0
"""Stable domain-only observations consumed by Gate E policies."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from typing import Any, Literal
import re

# First Party
from lmcache.v1.distributed.api import ObjectKey

# Local
from .actions import Tier


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ResidencyObservationState = Literal["ready", "writing", "evicting"]


@dataclass(frozen=True)
class ResidencyObservation:
    """Immutable policy view of one independently managed replica."""

    residency_id: str
    object_key: ObjectKey
    tier: Tier
    state: ResidencyObservationState
    size_bytes: int
    generation: int
    layout_fingerprint: str
    last_access_ns: int
    access_count: int
    active_readers: int
    pinned: bool

    def __post_init__(self) -> None:
        if not self.residency_id:
            raise ValueError("residency_id must not be empty")
        if self.tier not in ("dram", "cxl"):
            raise ValueError("residency tier is unsupported")
        if self.state not in ("ready", "writing", "evicting"):
            raise ValueError("residency state is unsupported")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.generation <= 0:
            raise ValueError("generation must be positive")
        if not _SHA256_RE.fullmatch(self.layout_fingerprint):
            raise ValueError("layout_fingerprint must be a lowercase SHA-256")
        if min(self.last_access_ns, self.access_count, self.active_readers) < 0:
            raise ValueError("residency counters must be non-negative")

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["object_key"] = asdict(self.object_key.to_encoded_object_key())
        return value


@dataclass(frozen=True)
class TierObservation:
    """Capacity and queue view for one placement tier."""

    tier: Tier
    capacity_bytes: int
    used_bytes: int
    queued_bytes: int
    estimated_bandwidth_bytes_per_s: int
    estimated_latency_ns: int

    def __post_init__(self) -> None:
        if self.tier not in ("dram", "cxl"):
            raise ValueError("tier is unsupported")
        if (
            min(
                self.capacity_bytes,
                self.used_bytes,
                self.queued_bytes,
                self.estimated_latency_ns,
            )
            < 0
        ):
            raise ValueError("tier size, queue, and time fields must be non-negative")
        if self.used_bytes > self.capacity_bytes:
            raise ValueError("used_bytes must not exceed capacity_bytes")
        if self.estimated_bandwidth_bytes_per_s <= 0:
            raise ValueError("estimated bandwidth must be positive")


@dataclass(frozen=True)
class RequestObservation:
    """Request facts needed for fetch-versus-recompute decisions."""

    request_id: str
    instance_id: int
    object_key: ObjectKey
    required_bytes: int
    external_matched_tokens: int
    deadline_ns: int | None
    recompute_estimate_ns: int
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.instance_id < 0:
            raise ValueError("instance_id must be non-negative")
        if self.required_bytes <= 0:
            raise ValueError("required_bytes must be positive")
        if self.external_matched_tokens < 0:
            raise ValueError("external_matched_tokens must be non-negative")
        if self.deadline_ns is not None and self.deadline_ns < 0:
            raise ValueError("deadline_ns must be non-negative")
        if self.recompute_estimate_ns <= 0:
            raise ValueError("recompute_estimate_ns must be positive")
        if not _SHA256_RE.fullmatch(self.layout_fingerprint):
            raise ValueError("layout_fingerprint must be a lowercase SHA-256")

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["object_key"] = asdict(self.object_key.to_encoded_object_key())
        return value


@dataclass(frozen=True)
class PlacementSnapshot:
    """Atomic policy input for one object and request."""

    timestamp_ns: int
    request: RequestObservation
    residencies: tuple[ResidencyObservation, ...]
    tiers: tuple[TierObservation, ...]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if any(item.object_key != self.request.object_key for item in self.residencies):
            raise ValueError("snapshot residencies must match the request object")
        tier_names = [item.tier for item in self.tiers]
        if len(set(tier_names)) != len(tier_names):
            raise ValueError("snapshot contains duplicate tier observations")

    def to_primitive(self) -> dict[str, Any]:
        """Return recursively JSON-compatible primitives."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "request": self.request.to_primitive(),
            "residencies": [item.to_primitive() for item in self.residencies],
            "tiers": [asdict(item) for item in self.tiers],
        }
