# SPDX-License-Identifier: Apache-2.0
"""Immutable contracts shared by the CXL proxy data plane."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping
import re

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, ObjectKey


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UINT64_MAX = (1 << 64) - 1
TransferDirection = Literal["store", "retrieve"]


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class PackedLayoutSpec:
    """Canonical metadata needed to interpret a packed KV extent."""

    layout_id: str
    layout_version: int
    model_name: str
    token_count: int
    object_group_order: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    engine_kv_formats: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.layout_id:
            raise ValueError("layout_id must not be empty")
        if self.layout_version <= 0:
            raise ValueError("layout_version must be positive")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not self.object_group_order or any(
            group_id < 0 for group_id in self.object_group_order
        ):
            raise ValueError("object_group_order must contain non-negative IDs")
        if not self.shapes or any(
            not shape or any(dimension <= 0 for dimension in shape)
            for shape in self.shapes
        ):
            raise ValueError("shapes must contain positive dimensions")
        field_lengths = {
            len(self.object_group_order),
            len(self.shapes),
            len(self.dtypes),
            len(self.engine_kv_formats),
        }
        if len(field_lengths) != 1:
            raise ValueError("packed layout group metadata lengths must match")
        if any(not dtype for dtype in self.dtypes):
            raise ValueError("dtypes must not contain empty values")
        if any(not kv_format for kv_format in self.engine_kv_formats):
            raise ValueError("engine_kv_formats must not contain empty values")

    @classmethod
    def from_primitive(cls, value: Mapping[str, Any]) -> PackedLayoutSpec:
        """Construct a packed layout from JSON-decoded values.

        Args:
            value: Mapping carrying the complete layout contract.

        Returns:
            A validated immutable layout specification.

        Raises:
            KeyError: If a required field is absent.
            TypeError: If nested values are not iterable as required.
            ValueError: If conversion or contract validation fails.
        """
        return cls(
            layout_id=str(value["layout_id"]),
            layout_version=int(value["layout_version"]),
            model_name=str(value["model_name"]),
            token_count=int(value["token_count"]),
            object_group_order=tuple(int(item) for item in value["object_group_order"]),
            shapes=tuple(
                tuple(int(dimension) for dimension in shape)
                for shape in value["shapes"]
            ),
            dtypes=tuple(str(dtype) for dtype in value["dtypes"]),
            engine_kv_formats=tuple(
                str(kv_format) for kv_format in value["engine_kv_formats"]
            ),
        )


@dataclass(frozen=True)
class ExtentDescriptor:
    """Serializable identity of an immutable extent in the CXL proxy region."""

    region_id: str
    offset: int
    length: int
    generation: int
    tier: Literal["cxl"]
    layout_id: str
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must not be empty")
        if self.offset < 0 or self.length <= 0:
            raise ValueError("extent offset must be non-negative and length positive")
        if self.offset > _UINT64_MAX or self.length > _UINT64_MAX:
            raise ValueError("extent fields must fit uint64")
        if self.offset + self.length > _UINT64_MAX:
            raise ValueError("extent end must fit uint64")
        if self.generation <= 0 or self.generation > _UINT64_MAX:
            raise ValueError("generation must be in the uint64 positive range")
        if self.tier != "cxl":
            raise ValueError("ExtentDescriptor tier must be 'cxl'")
        if not self.layout_id:
            raise ValueError("layout_id must not be empty")
        _validate_sha256(self.layout_fingerprint, "layout_fingerprint")


@dataclass(frozen=True)
class TransferPlan:
    """One versioned HBM-to-extent or extent-to-HBM transfer operation."""

    plan_version: int
    op_id: str
    direction: TransferDirection
    instance_id: int
    object_keys: tuple[ObjectKey, ...]
    block_ids_by_group: tuple[tuple[int, ...], ...]
    extent: ExtentDescriptor
    payload_checksum_expected: str | None

    def __post_init__(self) -> None:
        if self.plan_version <= 0:
            raise ValueError("plan_version must be positive")
        if not self.op_id:
            raise ValueError("op_id must not be empty")
        if self.direction not in ("store", "retrieve"):
            raise ValueError("direction must be 'store' or 'retrieve'")
        if self.instance_id < 0:
            raise ValueError("instance_id must be non-negative")
        if not self.object_keys:
            raise ValueError("object_keys must not be empty")
        if not self.block_ids_by_group or any(
            not group for group in self.block_ids_by_group
        ):
            raise ValueError("block_ids_by_group must contain non-empty groups")
        if any(block_id < 0 for group in self.block_ids_by_group for block_id in group):
            raise ValueError("block IDs must be non-negative")
        if self.payload_checksum_expected is not None:
            _validate_sha256(
                self.payload_checksum_expected, "payload_checksum_expected"
            )

    def to_primitive(self) -> dict[str, Any]:
        """Convert the plan to JSON-compatible primitive values.

        Returns:
            A dictionary containing no Python object identity or process-local
            handles.
        """
        return {
            "plan_version": self.plan_version,
            "op_id": self.op_id,
            "direction": self.direction,
            "instance_id": self.instance_id,
            "object_keys": [
                asdict(object_key.to_encoded_object_key())
                for object_key in self.object_keys
            ],
            "block_ids_by_group": [list(group) for group in self.block_ids_by_group],
            "extent": asdict(self.extent),
            "payload_checksum_expected": self.payload_checksum_expected,
        }

    @classmethod
    def from_primitive(cls, value: Mapping[str, Any]) -> TransferPlan:
        """Construct a validated plan from JSON-decoded primitive values.

        Args:
            value: Mapping produced by :meth:`to_primitive` and optionally
                round-tripped through JSON.

        Returns:
            The validated immutable transfer plan.

        Raises:
            KeyError: If a required field is absent.
            TypeError: If a nested record is not mapping-shaped.
            ValueError: If any contract invariant is invalid.
        """
        extent_value = value["extent"]
        if not isinstance(extent_value, Mapping):
            raise TypeError("extent must be a mapping")
        object_keys: list[ObjectKey] = []
        for item in value["object_keys"]:
            if not isinstance(item, Mapping):
                raise TypeError("object key must be a mapping")
            object_keys.append(EncodedObjectKey(**item).to_object_key())
        return cls(
            plan_version=int(value["plan_version"]),
            op_id=str(value["op_id"]),
            direction=value["direction"],
            instance_id=int(value["instance_id"]),
            object_keys=tuple(object_keys),
            block_ids_by_group=tuple(
                tuple(int(block_id) for block_id in group)
                for group in value["block_ids_by_group"]
            ),
            extent=ExtentDescriptor(**extent_value),
            payload_checksum_expected=value.get("payload_checksum_expected"),
        )


@dataclass(frozen=True)
class DataCompletion:
    """Terminal or pending state of the physical CUDA data operation."""

    op_id: str
    status: Literal["pending", "ok", "error"]
    complete_ns: int | None
    elapsed_ns: int | None
    error: str | None

    def __post_init__(self) -> None:
        if not self.op_id:
            raise ValueError("op_id must not be empty")
        if self.status not in ("pending", "ok", "error"):
            raise ValueError("invalid data completion status")
        for value in (self.complete_ns, self.elapsed_ns):
            if value is not None and value < 0:
                raise ValueError("completion timestamps must be non-negative")
        if self.status == "pending" and (
            self.complete_ns is not None or self.elapsed_ns is not None or self.error
        ):
            raise ValueError("pending completion cannot contain terminal fields")
        if self.status == "ok" and (
            self.complete_ns is None
            or self.elapsed_ns is None
            or self.error is not None
        ):
            raise ValueError("successful completion requires timestamps and no error")
        if self.status == "error" and not self.error:
            raise ValueError("error completion requires an error message")


@dataclass(frozen=True)
class CompositeCompletion:
    """Observable CUDA and modeled components of one logical operation."""

    op_id: str
    cuda_status: Literal["pending", "ok", "error"]
    modeled_status: Literal[
        "pending", "ok", "error", "cancelled", "not_required"
    ]
    cuda_elapsed_ns: int | None
    modeled_queue_ns: int | None
    modeled_service_ns: int | None
    cuda_complete_ns: int | None = None
    modeled_complete_ns: int | None = None
    effective_complete_ns: int | None = None
    effective_elapsed_ns: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.op_id:
            raise ValueError("op_id must not be empty")
        if self.cuda_status not in ("pending", "ok", "error"):
            raise ValueError("invalid CUDA completion status")
        if self.modeled_status not in (
            "pending",
            "ok",
            "error",
            "cancelled",
            "not_required",
        ):
            raise ValueError("invalid modeled completion status")
        for value in (
            self.cuda_elapsed_ns,
            self.modeled_queue_ns,
            self.modeled_service_ns,
            self.cuda_complete_ns,
            self.modeled_complete_ns,
            self.effective_complete_ns,
            self.effective_elapsed_ns,
        ):
            if value is not None and value < 0:
                raise ValueError("completion durations must be non-negative")
        if self.effective_complete_ns is not None and (
            self.cuda_status != "ok"
            or self.modeled_status not in ("ok", "not_required")
        ):
            raise ValueError("effective completion requires successful branches")
        if self.effective_elapsed_ns is not None and self.effective_complete_ns is None:
            raise ValueError("effective elapsed requires an effective completion")
        failed = self.cuda_status == "error" or self.modeled_status in (
            "error",
            "cancelled",
        )
        if failed and not self.error:
            raise ValueError("failed composite completion requires an error")
        if not failed and self.error is not None:
            raise ValueError("successful composite completion cannot have an error")
