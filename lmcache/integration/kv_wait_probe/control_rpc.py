# SPDX-License-Identifier: Apache-2.0
"""Passive request/RPC correlation; no protocol, polling, or future changes."""

# Standard
import hashlib
import os
from typing import Any

# Third Party
import msgspec

# First Party
from vllm.v1.kv_wait_trace import emit

ENABLED = os.environ.get("KV_WAIT_CONTROL_TRACE") == "1"
OPERATIONS = {"LOOKUP", "LOOKUP_GUARDED", "QUERY_PREFETCH_STATUS", "RETRIEVE"}


def client_event(
    name: str,
    client: int,
    uid: int,
    operation: Any,
    payloads: list[Any] | None = None,
    **fields: Any,
) -> None:
    """Record a client boundary, preserving request IDs from existing payloads."""
    if not ENABLED or operation.name not in OPERATIONS:
        return
    request_id = None
    if payloads:
        first = payloads[0]
        request_id = (
            first if isinstance(first, str) else getattr(first, "request_id", None)
        )
    emit(
        name,
        request_id,
        client_id=client,
        rpc_uid=uid,
        operation=operation.name,
        **fields,
    )


def fingerprint(payloads: list[bytes]) -> str | None:
    """Hash existing wire bytes to join client/server without changing identity."""
    if not ENABLED:
        return None
    digest = hashlib.blake2s(digest_size=16)
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def server_event(name: str, frames: list[bytes], **fields: Any) -> None:
    """Record server route/UID; request association is joined offline."""
    if not ENABLED:
        return
    from lmcache.v1.multiprocess.protocol import RequestType

    operation = RequestType(msgspec.msgpack.decode(frames[2]))
    if operation.name not in OPERATIONS:
        return
    emit(
        name,
        rpc_route=frames[0].hex(),
        rpc_uid=msgspec.msgpack.decode(frames[1]),
        operation=operation.name,
        **fields,
    )
