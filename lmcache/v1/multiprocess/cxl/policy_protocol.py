# SPDX-License-Identifier: Apache-2.0
"""Versioned external lookup metadata and fail-closed response contracts."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Literal
import re


GATE_E_PROTOCOL_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GateERequestEnvelope:
    """Backward-separate request metadata for policy lookup."""

    protocol_version: int
    request_id: str
    deadline_ns: int | None
    recompute_estimate_ns: int
    layout_fingerprint: str
    ticket_id: str | None = None

    def __post_init__(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.deadline_ns is not None and self.deadline_ns < 0:
            raise ValueError("deadline_ns must be non-negative")
        if self.recompute_estimate_ns <= 0:
            raise ValueError("recompute_estimate_ns must be positive")
        if not _SHA256_RE.fullmatch(self.layout_fingerprint):
            raise ValueError("layout_fingerprint must be a lowercase SHA-256")
        if self.ticket_id == "":
            raise ValueError("ticket_id must be non-empty when supplied")


@dataclass(frozen=True)
class GateELookupResponse:
    """Policy lookup result that cannot represent an unbound positive hit."""

    protocol_version: int
    status: Literal["fetch", "recompute", "unsupported"]
    matched_tokens: int
    ticket_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.protocol_version != GATE_E_PROTOCOL_VERSION:
            raise ValueError("lookup response protocol version is unsupported")
        if self.status not in ("fetch", "recompute", "unsupported"):
            raise ValueError("lookup response status is unsupported")
        if self.matched_tokens < 0:
            raise ValueError("matched_tokens must be non-negative")
        if not self.reason:
            raise ValueError("lookup response reason must not be empty")
        if self.status == "fetch":
            if self.matched_tokens <= 0 or not self.ticket_ids:
                raise ValueError("positive match requires at least one bound ticket")
            if any(not ticket_id for ticket_id in self.ticket_ids):
                raise ValueError("ticket IDs must not be empty")
        elif self.matched_tokens != 0 or self.ticket_ids:
            raise ValueError("non-fetch response cannot expose matches or tickets")

    @classmethod
    def unsupported(cls, reason: str) -> GateELookupResponse:
        """Return an explicit unsupported-feature response."""
        return cls(GATE_E_PROTOCOL_VERSION, "unsupported", 0, (), reason)

    @classmethod
    def recompute(cls, reason: str) -> GateELookupResponse:
        """Return an explicit zero-match recompute response."""
        return cls(GATE_E_PROTOCOL_VERSION, "recompute", 0, (), reason)


class GateEConnectorTranslator:
    """Translate server decisions without trusting advisory positive matches."""

    @staticmethod
    def matched_tokens(response: GateELookupResponse) -> int:
        """Return tokens only when the response carries bound tickets."""
        if response.status != "fetch":
            return 0
        if not response.ticket_ids:
            raise RuntimeError("positive external match is not ticket-bound")
        return response.matched_tokens
