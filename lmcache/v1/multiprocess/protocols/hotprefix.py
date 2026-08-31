# SPDX-License-Identifier: Apache-2.0

"""HotPrefix protocol definitions."""

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

HOTPREFIX_STORE_REQUEST_PREFIX = "__hotprefix_store__:"
HOTPREFIX_PROMOTION_REQUEST_PREFIX = "__hotprefix_promotion__:"


def is_hotprefix_store_request(request_id: str) -> bool:
    """Return whether an internal request carries eviction STORE payload.

    Args:
        request_id: Internal connector operation ID.

    Returns:
        ``True`` for HotPrefix eviction STORE operations.
    """
    return request_id.startswith(HOTPREFIX_STORE_REQUEST_PREFIX)


def is_hotprefix_promotion_request(request_id: str) -> bool:
    """Return whether a request carries a promotion RETRIEVE chunk.

    Args:
        request_id: Internal connector operation ID.

    Returns:
        ``True`` for HotPrefix promotion chunk operations.
    """
    return request_id.startswith(HOTPREFIX_PROMOTION_REQUEST_PREFIX)


class HotPrefixAccessResponse(msgspec.Struct, frozen=True):
    """Committed Global Hotness Epoch and canonical matched path."""

    epoch: int
    global_matched_tokens: int
    path: list[bytes]


class HotPrefixAdmissionResponse(msgspec.Struct, frozen=True):
    """Selective Host admission result and reserved generation."""

    action: str
    reason: str
    evict_prefixes: list[bytes]
    generation: int | None = None


class HotPrefixHostCandidate(msgspec.Struct, frozen=True):
    """READY shared Host source visible to a target instance planner."""

    prefix_id: bytes
    size_bytes: int
    generation: int
    frequency: int
    clock: int


class HotPrefixTransferTicket(msgspec.Struct, frozen=True):
    """Generation-bound authorization for one Host-to-HBM promotion read."""

    ticket_id: bytes
    prefix_id: bytes
    generation: int
    size_bytes: int


REQUEST_NAMES = [
    "HOT_PREFIX_ACCESS",
    "HOT_PREFIX_ADMIT",
    "HOT_PREFIX_PUBLISH",
    "HOT_PREFIX_ABORT",
    "HOT_PREFIX_CANDIDATES",
    "HOT_PREFIX_ACQUIRE",
    "HOT_PREFIX_RELEASE",
    "HOT_PREFIX_RENEW",
    "HOT_PREFIX_INVALIDATE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return protocol definitions for HotPrefix control operations.

    Returns:
        Request-name to payload/response/handler definitions.
    """
    return {
        "HOT_PREFIX_ACCESS": ProtocolDefinition(
            payload_classes=[int, int, bytes, list[int], int],
            response_class=HotPrefixAccessResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_ADMIT": ProtocolDefinition(
            payload_classes=[bytes, bytes, int, int],
            response_class=HotPrefixAdmissionResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_PUBLISH": ProtocolDefinition(
            payload_classes=[bytes, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_ABORT": ProtocolDefinition(
            payload_classes=[bytes, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_CANDIDATES": ProtocolDefinition(
            payload_classes=[bytes, list[bytes]],
            response_class=list[HotPrefixHostCandidate],
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_ACQUIRE": ProtocolDefinition(
            payload_classes=[bytes, bytes, int, bytes],
            response_class=HotPrefixTransferTicket,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_RELEASE": ProtocolDefinition(
            payload_classes=[bytes, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_RENEW": ProtocolDefinition(
            payload_classes=[bytes, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "HOT_PREFIX_INVALIDATE": ProtocolDefinition(
            payload_classes=[bytes, bytes, int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
    }
