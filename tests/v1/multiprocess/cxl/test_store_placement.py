# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cxl.actions import StorePlacementPlan, TargetSpec
from lmcache.v1.multiprocess.cxl.directory import MultiResidencyDirectory
from lmcache.v1.multiprocess.cxl.placement import (
    StoreCoordinator,
    TargetTransferCompletion,
)
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle


pytestmark = pytest.mark.no_shared_allocator


def _key() -> ObjectKey:
    return ObjectKey(b"chunk", "model", 0)


def _directory() -> MultiResidencyDirectory:
    managers = {}
    for tier in ("dram", "cxl"):
        managers[tier] = CXLRegionManager(
            RegionHandle(tier, f"/{tier}", 4096, 64, frozenset()),
            layout_id="packed_kv_v1",
            layout_fingerprint="a" * 64,
            tier=tier,
        )
    return MultiResidencyDirectory(managers)


@dataclass
class _Source:
    released: bool = False

    def release(self) -> None:
        self.released = True


class _Executor:
    def __init__(self, outcomes: dict[str, bool], source: _Source) -> None:
        self.outcomes = outcomes
        self.source = source
        self.seen: list[str] = []

    def transfer(self, target: TargetSpec, residency: object, source: object):
        assert source is self.source
        assert not self.source.released
        self.seen.append(target.tier)
        ok = self.outcomes[target.tier]
        return TargetTransferCompletion(target.tier, ok, None if ok else "failed")


@pytest.mark.parametrize(
    ("targets", "outcomes", "required", "partial"),
    [
        ((TargetSpec("dram", True, "only"),), {"dram": True}, True, False),
        ((TargetSpec("cxl", True, "only"),), {"cxl": True}, True, False),
        (
            (
                TargetSpec("dram", True, "hot"),
                TargetSpec("cxl", False, "replica"),
            ),
            {"dram": True, "cxl": False},
            True,
            True,
        ),
        (
            (TargetSpec("dram", True, "a"), TargetSpec("cxl", True, "b")),
            {"dram": True, "cxl": False},
            False,
            True,
        ),
    ],
)
def test_independent_required_optional_store_targets(
    targets: tuple[TargetSpec, ...],
    outcomes: dict[str, bool],
    required: bool,
    partial: bool,
) -> None:
    directory = _directory()
    source = _Source()
    result = StoreCoordinator(directory, _Executor(outcomes, source)).execute(
        StorePlacementPlan(_key(), targets, "matrix"),
        source,
        length=256,
        alignment=64,
    )

    assert result.required_satisfied is required
    assert result.partial_success is partial
    assert source.released
    assert {item.tier for item in result.successful_residencies} == {
        tier for tier, ok in outcomes.items() if ok
    }
    assert {item.tier for item in result.failed_targets} == {
        tier for tier, ok in outcomes.items() if not ok
    }
