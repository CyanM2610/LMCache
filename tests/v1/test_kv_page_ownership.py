# SPDX-License-Identifier: Apache-2.0
"""Regression for MP transfers still owning a page reused by native scheduling."""

# Standard
from types import SimpleNamespace

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.request import Request
import pytest

# First Party
from lmcache.integration.vllm.kv_page_ownership import PendingKVPageAccess
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.v1.multiprocess.futures import MessagingFuture


def test_other_request_reuse_waits_even_without_current_preemption_hint() -> None:
    """A retired page may be reassigned in a later step whose hint is false."""
    future: MessagingFuture[bool] = MessagingFuture()
    owner = PendingKVPageAccess()
    owner.record(future, {0: [2]}, writes=False)
    connector = object.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.WORKER
    connector.worker_adapter = SimpleNamespace(
        require_terminal_completion=True,
        wait_for_kv_access=lambda reads, writes: owner.wait(reads, writes, timeout=0),
        handle_preemptions=lambda hint: None,
    )
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_compute_access({0: [1, 2]}, [16], 16, 1)
    assert not metadata.need_flush_before_forward
    with pytest.raises(TimeoutError):
        connector.handle_preemptions(metadata)
    assert owner.pending_count == 1
    future.set_result(True)
    connector.handle_preemptions(metadata)
    assert owner.pending_count == 0


def test_immutable_prefix_read_and_other_group_do_not_wait_for_store() -> None:
    """Continuing decode can read a stored prefix while writing another page."""
    future: MessagingFuture[bool] = MessagingFuture()
    owner = PendingKVPageAccess()
    original = {0: [2]}
    owner.record(future, original, writes=False)
    original[0].clear()
    owner.wait({0: [2]}, {0: [3], 1: [2]}, timeout=0)
    assert owner.pending_count == 1
    with pytest.raises(TimeoutError):
        owner.wait({}, {0: [2]}, timeout=0)


def test_compute_read_waits_for_pending_load_write() -> None:
    """Page identity alone cannot make an unfinished remote LOAD usable."""
    future: MessagingFuture[bool] = MessagingFuture()
    owner = PendingKVPageAccess()
    owner.record(future, {0: [2]}, writes=True)
    with pytest.raises(TimeoutError):
        owner.wait({0: [2]}, {}, timeout=0)
    future.set_result(True)
    owner.wait({0: [2]}, {}, timeout=0)
    assert owner.pending_count == 0


def test_separate_submissions_keep_separate_completion_obligations() -> None:
    """New submissions, including one request's next STORE, cannot erase an old one."""
    first: MessagingFuture[bool] = MessagingFuture()
    second: MessagingFuture[bool] = MessagingFuture()
    owner = PendingKVPageAccess()
    owner.record(first, {0: [2]}, writes=False)
    owner.record(second, {0: [3]}, writes=False)
    second.set_result(True)
    owner.reap()
    assert owner.pending_count == 1
    owner.wait({}, {0: [3]}, timeout=0)
    with pytest.raises(TimeoutError):
        owner.wait({}, {0: [2]}, timeout=0)


def test_terminal_copy_failure_releases_but_transport_exception_does_not() -> None:
    """Failed copy with a proven fence is distinct from unknown remote ownership."""
    terminal: MessagingFuture[bool] = MessagingFuture()
    terminal.set_result(False)
    owner = PendingKVPageAccess()
    owner.record(terminal, {0: [2]}, writes=False)
    owner.wait({}, {0: [2]}, timeout=0)
    assert owner.pending_count == 0 and terminal.result() is False
    unknown: MessagingFuture[bool] = MessagingFuture()
    unknown.set_exception(ConnectionError("lost remote response"))
    owner.record(unknown, {0: [2]}, writes=False)
    with pytest.raises(ConnectionError):
        owner.wait({}, {0: [2]}, timeout=0)
    assert owner.pending_count == 1


def test_native_positions_define_exact_write_pages_and_snapshot_allocation() -> None:
    """A partial tail writes one page; each group's geometry is independent."""
    metadata = LMCacheMPConnectorMetadata()
    blocks = {0: [8, 9, 10], 1: [4, 5, 6, 7, 8, 9]}
    metadata.add_compute_access(blocks, [16, 8], 16, 1)
    assert metadata.kv_read_blocks == {0: (8, 9), 1: (4, 5, 6)}
    assert metadata.kv_write_blocks == {0: (9,), 1: (6,)}
    blocks[0][1] = 100
    assert metadata.kv_write_blocks[0] == (9,)
    with pytest.raises(ValueError, match="physical allocation"):
        metadata.add_compute_access({0: [2]}, [16], 16, 1)


def test_connector_builds_hazards_from_native_cached_request_packet() -> None:
    """Exercise the real builder with native list-based per-group geometry."""
    request = Request("B", list(range(17)), SamplingParams(max_tokens=4), None)
    tracker = LMCacheMPRequestTracker(request)
    tracker.state = LMCacheMPRequestState.READY
    tracker.allocated_block_ids = {0: [1]}
    tracker.num_scheduled_tokens = tracker.num_stored_tokens = 16
    connector = object.__new__(LMCacheMPConnector)
    connector._require_terminal_completion = True
    connector._group_tokens_per_block = [16]
    connector.lazy_offload = False
    connector.request_trackers = {"B": tracker}
    connector.scheduler_adapter = SimpleNamespace(
        lmcache_tokens_per_chunk=16, report_block_allocations=lambda records: None
    )
    cached = CachedRequestData.make_empty()
    cached.req_ids = ["B"]
    cached.new_block_ids = [([2],)]
    cached.num_computed_tokens = [16]
    cached.num_output_tokens = [1]
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"B": 1},
        total_num_scheduled_tokens=1,
        preempted_req_ids=set(),
    )
    metadata = connector.build_connector_meta(output)
    assert metadata.kv_read_blocks == {0: (1, 2)}
    assert metadata.kv_write_blocks == {0: (2,)}
