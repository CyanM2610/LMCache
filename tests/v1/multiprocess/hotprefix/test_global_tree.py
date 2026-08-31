# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.hotprefix.global_tree import (
    GlobalHostPrefixTree,
    PrefixAccessObservation,
)
from lmcache.v1.multiprocess.modules.hotprefix import HotPrefixModule
from lmcache.v1.multiprocess.protocols.base import RequestType

pytestmark = pytest.mark.no_shared_allocator


def test_global_tree_merges_instance_streams_and_deduplicates_events() -> None:
    tree = GlobalHostPrefixTree(
        namespace=b"model-a",
        max_value=255,
        max_age=255,
        aging_interval=100,
    )
    first = PrefixAccessObservation(1, 1, (1, 2, 3), matched_tokens=0)
    branch = PrefixAccessObservation(2, 1, (1, 2, 4), matched_tokens=2)

    first_result = tree.observe(first)
    branch_result = tree.observe(branch)
    duplicate_result = tree.observe(branch)

    by_prefix = {node.full_prefix: node for node in tree.snapshot()}
    assert first_result.epoch == 1
    assert branch_result.epoch == 2
    assert duplicate_result == branch_result
    assert by_prefix[(1, 2)].frequency == 2
    assert by_prefix[(1, 2)].clock == 255
    assert by_prefix[(1, 2)].global_hotness == 510
    assert by_prefix[(1, 2, 3)].frequency == 1
    assert by_prefix[(1, 2, 4)].frequency == 1
    assert tree.get(by_prefix[(1, 2)].prefix_id) == by_prefix[(1, 2)]


def test_hotprefix_module_exposes_multi_instance_access_handler() -> None:
    module = HotPrefixModule(
        object(),  # type: ignore[arg-type]
        aging_interval=100,
        host_capacity_bytes=100,
        frequency_threshold=1,
    )

    first = module.access(1, 1, b"model-a", [1, 2, 3], 0)
    second = module.access(2, 1, b"model-a", [1, 2, 4], 0)

    assert first.epoch == 1
    assert second.epoch == 2
    assert module.get_handlers()[0].request_type is RequestType.HOT_PREFIX_ACCESS
    assert module.report_status() == {"hotprefix_trees": 1, "hotprefix_nodes": 3}

    prefix_id = first.path[-1]
    absent = module.admit(b"model-a", b"not-observed", 100, 17)
    admitted = module.admit(b"model-a", prefix_id, 100, 17)
    assert absent.action == "reject"
    assert absent.reason == "prefix_absent_from_global_tree"
    assert admitted.action == "accept"
    assert admitted.generation == 17
    assert module.publish(b"model-a", prefix_id) is True
    assert module.publish(b"model-a", prefix_id) is True
    deduplicated = module.admit(b"model-a", prefix_id, 100, 18)
    assert deduplicated.action == "dedup"
    candidates = module.candidates(b"model-a", [prefix_id, b"missing"])
    assert [(item.prefix_id, item.generation) for item in candidates] == [
        (prefix_id, 17)
    ]
    ticket = module.acquire(b"model-a", prefix_id, 17, b"ticket-a")
    assert ticket.prefix_id == prefix_id
    assert ticket.generation == 17
    assert module.renew(b"model-a", ticket.ticket_id) is True
    assert module.release(b"model-a", ticket.ticket_id) is True
    assert module.release(b"model-a", ticket.ticket_id) is True
    assert module.invalidate(b"model-a", prefix_id, 17) is True
    assert module.candidates(b"model-a", [prefix_id]) == []
    assert module.abort(b"model-a", b"never-reserved") is True
