# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from benchmarks.beluga_proxy.direct_copy import build_block_map


pytestmark = pytest.mark.no_shared_allocator


def test_fragmented_benchmark_uses_a_non_contiguous_paged_block_map() -> None:
    contiguous = build_block_map(32, fragmented=False, fragment_count=8)
    fragmented = build_block_map(32, fragmented=True, fragment_count=8)

    assert contiguous == tuple(range(32))
    assert len(fragmented) == 32
    assert len(set(fragmented)) == 32
    assert any(
        right - left > 1
        for left, right in zip(fragmented, fragmented[1:], strict=False)
    )
