# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.contracts import PackedLayoutSpec
from lmcache.v1.multiprocess.cxl.layout import (
    canonical_layout_bytes,
    layout_fingerprint,
    layouts_compatible,
    payload_checksum,
)


pytestmark = pytest.mark.no_shared_allocator


def _spec(**overrides: object) -> PackedLayoutSpec:
    values = {
        "layout_id": "packed_kv_v1",
        "layout_version": 1,
        "model_name": "Qwen2.5-7B-Instruct",
        "token_count": 256,
        "object_group_order": (0, 1),
        "shapes": ((2, 28, 256, 128), (2, 4, 256, 128)),
        "dtypes": ("torch.bfloat16", "bf16"),
        "engine_kv_formats": (
            "NL_X_TWO_NB_BS_NH_HS",
            "NL_X_TWO_NB_BS_NH_HS",
        ),
    }
    values.update(overrides)
    return PackedLayoutSpec(**values)  # type: ignore[arg-type]


def test_canonical_layout_is_independent_of_mapping_construction_order() -> None:
    first = {
        "layout_id": "packed_kv_v1",
        "layout_version": 1,
        "model_name": "Qwen2.5-7B-Instruct",
        "token_count": 256,
        "object_group_order": [0, 1],
        "shapes": [[2, 28, 256, 128], [2, 4, 256, 128]],
        "dtypes": ["torch.bfloat16", "bf16"],
        "engine_kv_formats": [
            "NL_X_TWO_NB_BS_NH_HS",
            "NL_X_TWO_NB_BS_NH_HS",
        ],
    }
    second = dict(reversed(list(first.items())))

    first_spec = PackedLayoutSpec.from_primitive(first)
    second_spec = PackedLayoutSpec.from_primitive(second)

    assert canonical_layout_bytes(first_spec) == canonical_layout_bytes(second_spec)
    assert layout_fingerprint(first_spec) == layout_fingerprint(second_spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layout_version", 2),
        ("model_name", "Qwen2.5-14B-Instruct"),
        ("token_count", 512),
        ("object_group_order", (1, 0)),
        ("shapes", ((2, 28, 256, 64), (2, 4, 256, 128))),
        ("dtypes", ("float16", "bfloat16")),
        (
            "engine_kv_formats",
            ("NL_X_NB_TWO_BS_NH_HS", "NL_X_TWO_NB_BS_NH_HS"),
        ),
    ],
)
def test_every_layout_dimension_contributes_to_fingerprint(
    field: str, value: object
) -> None:
    assert layout_fingerprint(_spec()) != layout_fingerprint(_spec(**{field: value}))


def test_layout_id_alone_does_not_establish_compatibility() -> None:
    producer = _spec()
    consumer = _spec(token_count=512)

    assert producer.layout_id == consumer.layout_id
    assert not layouts_compatible(producer, consumer)


def test_dtype_aliases_have_one_canonical_identity() -> None:
    assert layout_fingerprint(_spec()) == layout_fingerprint(
        _spec(dtypes=("bfloat16", "torch.bfloat16"))
    )


def test_payload_identity_is_separate_from_layout_identity() -> None:
    spec = _spec()

    assert layout_fingerprint(spec) == layout_fingerprint(spec)
    assert payload_checksum(b"payload-a") != payload_checksum(b"payload-b")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layout_id", ""),
        ("layout_version", 0),
        ("model_name", ""),
        ("token_count", 0),
        ("object_group_order", ()),
        ("object_group_order", (0, -1)),
        ("shapes", ()),
        ("shapes", ((2, 0),)),
        ("dtypes", ("bfloat16",)),
        ("engine_kv_formats", ("NL_X_TWO_NB_BS_NH_HS",)),
    ],
)
def test_packed_layout_rejects_incomplete_or_inconsistent_metadata(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _spec(**{field: value})
