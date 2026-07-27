# SPDX-License-Identifier: Apache-2.0
"""Canonical identity helpers for packed KV layouts and payloads."""

# Standard
import hashlib
import json

# Local
from .contracts import PackedLayoutSpec


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "float": "float32",
    "fp32": "float32",
    "torch.float32": "float32",
}


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower()
    return _DTYPE_ALIASES.get(normalized, normalized.removeprefix("torch."))


def canonical_layout_bytes(spec: PackedLayoutSpec) -> bytes:
    """Encode layout metadata into deterministic UTF-8 JSON.

    Args:
        spec: Complete packed-layout metadata.

    Returns:
        Canonical JSON bytes with normalized dtype names.
    """
    value = {
        "dtypes": [_normalize_dtype(dtype) for dtype in spec.dtypes],
        "engine_kv_formats": list(spec.engine_kv_formats),
        "layout_id": spec.layout_id,
        "layout_version": spec.layout_version,
        "model_name": spec.model_name,
        "object_group_order": list(spec.object_group_order),
        "shapes": [list(shape) for shape in spec.shapes],
        "token_count": spec.token_count,
    }
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def layout_fingerprint(spec: PackedLayoutSpec) -> str:
    """Compute the SHA-256 identity of canonical layout metadata.

    Args:
        spec: Complete packed-layout metadata.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    return hashlib.sha256(canonical_layout_bytes(spec)).hexdigest()


def layouts_compatible(producer: PackedLayoutSpec, consumer: PackedLayoutSpec) -> bool:
    """Check whether producer bytes have the consumer's exact layout.

    Args:
        producer: Layout recorded when publishing an extent.
        consumer: Layout required by the retrieving engine.

    Returns:
        True only when both canonical layout identities are equal.
    """
    return layout_fingerprint(producer) == layout_fingerprint(consumer)


def payload_checksum(payload: bytes) -> str:
    """Compute a payload checksum separate from layout compatibility.

    Args:
        payload: Packed KV payload bytes.

    Returns:
        A 64-character lowercase SHA-256 digest.
    """
    return hashlib.sha256(payload).hexdigest()
