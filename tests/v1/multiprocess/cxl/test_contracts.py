# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import fields
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.contracts import (
    CompositeCompletion,
    DataCompletion,
    ExtentDescriptor,
    TransferPlan,
)


pytestmark = pytest.mark.no_shared_allocator
FINGERPRINT = "a" * 64


def _extent(**overrides: object) -> ExtentDescriptor:
    values = {
        "region_id": "proxy0",
        "offset": 4096,
        "length": 8192,
        "generation": 1,
        "tier": "cxl",
        "layout_id": "packed_kv_v1",
        "layout_fingerprint": FINGERPRINT,
    }
    values.update(overrides)
    return ExtentDescriptor(**values)  # type: ignore[arg-type]


def _object_key() -> ObjectKey:
    return ObjectKey(b"chunk", "Qwen2.5-7B-Instruct", 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region_id", ""),
        ("offset", -1),
        ("length", 0),
        ("generation", 0),
        ("tier", "dram"),
        ("layout_id", ""),
        ("layout_fingerprint", "A" * 64),
        ("layout_fingerprint", "a" * 63),
        ("layout_fingerprint", "g" * 64),
    ],
)
def test_extent_descriptor_rejects_invalid_identity_or_bounds(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _extent(**{field: value})


def test_extent_descriptor_rejects_uint64_overflow() -> None:
    with pytest.raises(ValueError, match="uint64"):
        _extent(offset=(1 << 64) - 1, length=2)


def test_extent_descriptor_never_exposes_process_local_identity() -> None:
    field_names = {field.name for field in fields(ExtentDescriptor)}
    assert not field_names & {
        "address",
        "block_ids",
        "device_pointer",
        "fd",
        "pointer",
        "raw_pointer",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plan_version", 0),
        ("op_id", ""),
        ("direction", "copy"),
        ("instance_id", -1),
        ("object_keys", ()),
        ("block_ids_by_group", ()),
        ("block_ids_by_group", ((0, -1),)),
        ("payload_checksum_expected", "xyz"),
    ],
)
def test_transfer_plan_rejects_invalid_inputs(field: str, value: object) -> None:
    values = {
        "plan_version": 1,
        "op_id": "op-1",
        "direction": "store",
        "instance_id": 0,
        "object_keys": (_object_key(),),
        "block_ids_by_group": ((7, 3),),
        "extent": _extent(),
        "payload_checksum_expected": None,
    }
    values[field] = value
    with pytest.raises(ValueError):
        TransferPlan(**values)  # type: ignore[arg-type]


def test_transfer_plan_round_trips_through_json_primitives() -> None:
    plan = TransferPlan(
        plan_version=1,
        op_id="op-1",
        direction="retrieve",
        instance_id=4,
        object_keys=(_object_key(),),
        block_ids_by_group=((9, 2), (17, 11)),
        extent=_extent(),
        payload_checksum_expected="0" * 64,
    )

    encoded = json.loads(json.dumps(plan.to_primitive()))

    assert TransferPlan.from_primitive(encoded) == plan


def test_completion_contracts_validate_terminal_fields() -> None:
    data = DataCompletion(
        op_id="op-1", status="ok", complete_ns=12, elapsed_ns=7, error=None
    )
    composite = CompositeCompletion(
        op_id="op-1",
        cuda_status="ok",
        modeled_status="not_required",
        cuda_elapsed_ns=7,
        modeled_queue_ns=None,
        modeled_service_ns=None,
    )

    assert data.status == "ok"
    assert composite.modeled_status == "not_required"

    with pytest.raises(ValueError):
        DataCompletion(
            op_id="op-1",
            status="error",
            complete_ns=12,
            elapsed_ns=7,
            error=None,
        )
    with pytest.raises(ValueError):
        CompositeCompletion(
            op_id="op-1",
            cuda_status="done",
            modeled_status="not_required",
            cuda_elapsed_ns=None,
            modeled_queue_ns=None,
            modeled_service_ns=None,
        )
