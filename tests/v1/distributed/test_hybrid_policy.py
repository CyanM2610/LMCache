# SPDX-License-Identifier: Apache-2.0
"""The retention policy sees only current state and never overrides protection."""

# First Party
from lmcache.v1.distributed.hybrid_policy import (
    DramFirstPolicy,
    EvictionCandidate,
    ForcedPoolPolicy,
)


def test_victims_are_cxl_only_unprotected_lru_with_stable_ties() -> None:
    """Even the oldest DRAM/pinned entries are ineligible for replacement."""
    candidates = [
        EvictionCandidate("dram-old", "dram", 4096, 0, True),
        EvictionCandidate("cxl-pinned", "cxl_sim", 4096, 0, False),
        EvictionCandidate("b", "cxl_sim", 4096, 10, True),
        EvictionCandidate("a", "cxl_sim", 4096, 10, True),
        EvictionCandidate("recent", "cxl_sim", 4096, 11, True),
    ]
    policy = DramFirstPolicy()
    assert policy.pool_order() == ("dram", "cxl_sim")
    assert policy.victims(candidates) == ["a", "b", "recent"]
    assert ForcedPoolPolicy("dram").victims(candidates) == []
    assert ForcedPoolPolicy("cxl_sim").victims(candidates) == ["a", "b", "recent"]
