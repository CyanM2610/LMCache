# SPDX-License-Identifier: Apache-2.0
"""Native BlockPool + Promotion ownership; fake transport, no GPU timing claims."""

# Standard
from typing import Any
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
import pytest

# First Party
from lmcache.integration.vllm.hybrid_promotion import (
    PromotionCommand,
    PromotionCompletions,
    PromotionReceipt,
    PromotionScheduler,
)
from lmcache.integration.vllm.kv_page_ownership import PendingKVPageAccess
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)


def request(tokens: list[int]) -> Request:
    """Construct native hashes through the actual Request input path."""
    init_none_hash(sha256)
    return Request(
        "real-input",
        tokens,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(2, sha256),
    )


def pool(size: int = 7, events: bool = False) -> BlockPool:
    """Return a real native pool; one page is reserved as the native null page."""
    return BlockPool(size, True, 2, enable_kv_cache_events=events)


def cache(target: BlockPool, req: Request, blocks: int) -> tuple[int, ...]:
    """Populate via native cache_full_blocks, then native tail-first release."""
    allocated = target.get_new_blocks(blocks)
    target.cache_full_blocks(req, allocated, 0, blocks, 2, 0)
    target.free_blocks(reversed(allocated))
    return tuple(b.block_id for b in allocated)


def state(target: BlockPool) -> tuple[Any, ...]:
    """Observe references, public native free order and published identities."""
    return (
        tuple((b.block_id, b.ref_cnt, b.block_hash) for b in target.blocks),
        tuple(b.block_id for b in target.free_block_queue.get_all_free_blocks()),
    )


class Lookup:
    """Controllable source transport; native HBM ownership remains real."""

    lmcache_tokens_per_chunk = 4

    def __init__(self, result: int | None = 8) -> None:
        self.result = result
        self.ended: list[str] = []
        self.cleaned: list[str] = []
        self.fail_cleanup = False

    def maybe_submit_lookup_request(self, control_id: str, tokens: list[int]) -> None:
        """Record actual source key input for identity checks."""
        self.submitted = (control_id, tokens)

    def check_lookup_result(self, control_id: str) -> int | None:
        """Return pending/miss/full-host-prefix status."""
        return self.result

    def cleanup_lookup_result(self, control_id: str) -> None:
        """Discard lookup result without releasing its independent source lease."""
        self.cleaned.append(control_id)

    def end_session(self, control_id: str) -> None:
        """Complete logical release, optionally simulating a retryable failure."""
        if self.fail_cleanup:
            self.fail_cleanup = False
            raise TimeoutError("cleanup unconfirmed")
        self.ended.append(control_id)


class Future:
    """Terminal-future test double; timeout deliberately differs from false."""

    def __init__(self, result: bool = True, ready: bool = False) -> None:
        self.value = result
        self.ready = ready
        self.error = False

    def query(self) -> bool:
        """Return only physical terminal readiness."""
        return self.ready

    def result(self, timeout: float | None = None) -> bool:
        """Require readiness, retaining uncertain completion on injected error."""
        if self.error or not self.ready:
            raise TimeoutError("terminal not established")
        return self.value


def test_snapshot_does_not_follow_later_output() -> None:
    """No live Request/token list alias may expose later generation to a policy."""
    target, req = pool(), request(list(range(8)))
    snapshot = target.snapshot_external_prefix(req, 8)
    req.append_output_token_ids([91, 92])
    assert snapshot.token_ids == tuple(range(8))
    assert len(snapshot.block_hashes) == 4
    with pytest.raises(ValueError):
        target.snapshot_external_prefix(req, 12)


def test_capacity_rejection_does_not_touch_cached_prefix() -> None:
    """Free cached pages cannot count both as prefix pins and suffix capacity."""
    target, req = pool(), request(list(range(8)))
    cache(target, req, 2)
    occupied = target.get_new_blocks(4)
    before = state(target)
    assert (
        target.reserve_external_prefix(target.snapshot_external_prefix(req, 8), 4)
        is None
    )
    assert state(target) == before
    target.free_blocks(occupied)


def test_pin_prevents_self_eviction_and_keeps_native_lru_victims() -> None:
    """Protect matching pages before allocating; remaining victims use native order."""
    target, req = pool(5, True), request(list(range(8)))
    original = cache(target, req, 2)
    other = request(list(range(20, 24)))
    victims = cache(target, other, 2)
    before_order = tuple(
        b.block_id for b in target.free_block_queue.get_all_free_blocks()
    )
    assert before_order[:2] == tuple(reversed(original))
    snapshot = target.snapshot_external_prefix(req, 8)
    reservation = target.reserve_external_prefix(snapshot, 2)
    assert reservation is not None
    assert reservation.block_ids[:2] == original
    assert reservation.block_ids[2:] == tuple(reversed(victims))
    assert target.get_cached_block(snapshot.block_hashes[2], [0]) is None
    assert target.get_num_free_blocks() == 0
    target.finish_external_prefix(reservation, publish=True)
    for index, block_hash in enumerate(snapshot.block_hashes):
        hit = target.get_cached_block(block_hash, [0])
        assert hit is not None and hit[0].block_id == reservation.block_ids[index]
    assert target.get_num_free_blocks() == 4
    stored = target.kv_event_queue[-1]
    assert stored.token_ids == list(range(4, 8))
    assert stored.group_idx == 0
    with pytest.raises(ValueError):
        target.finish_external_prefix(reservation, publish=True)


def test_pending_lookup_allocates_nothing_and_duplicate_is_bounded() -> None:
    """Source readiness precedes HBM mutation; controls deduplicate independently."""
    target, source = pool(), Lookup(None)
    runtime = PromotionScheduler(target, source)
    prefix = target.snapshot_external_prefix(request(list(range(8))), 8)
    before = state(target)
    control = runtime.enqueue(prefix, 4)
    assert control and runtime.enqueue(prefix, 4) is None
    assert runtime.prepare() == () and state(target) == before
    assert runtime.has_pending_work
    runtime.cancel(control)
    assert source.ended == [control] and not runtime.has_pending_work
    assert state(target) == before


@pytest.mark.parametrize(
    "publish, cancel", [(True, False), (False, False), (True, True)]
)
def test_captured_ownership_survives_until_output_adoption(
    publish: bool, cancel: bool
) -> None:
    """Success/failure/cancel differ only at adoption, never by early page release."""
    target, source = pool(), Lookup()
    runtime = PromotionScheduler(target, source)
    prefix = target.snapshot_external_prefix(request(list(range(8))), 8)
    control = runtime.enqueue(prefix, 4)
    assert control is not None
    commands = runtime.prepare()
    assert len(commands) == 1 and commands[0].control_id == control
    assert runtime.prepare() == ()
    free_count = target.get_num_free_blocks()
    if cancel:
        runtime.cancel(control)
    assert target.get_num_free_blocks() == free_count
    assert not source.ended and runtime.has_pending_work
    with pytest.raises(ValueError):
        target.get_new_blocks(free_count + 1)
    runtime.adopt(PromotionReceipt(control, publish))
    assert target.get_num_free_blocks() == 6 and source.ended == [control]
    assert not runtime.has_pending_work
    assert (target.get_cached_block(prefix.block_hashes[-1], [0]) is not None) == (
        publish and not cancel
    )
    with pytest.raises(ValueError):
        runtime.adopt(PromotionReceipt(control, publish))


def test_host_miss_budget_and_full_hit_do_not_evict() -> None:
    """Reject unavailable, over-budget and already-resident candidates harmlessly."""
    target, req = pool(), request(list(range(8)))
    prefix = target.snapshot_external_prefix(req, 8)
    for matched, budget in [(0, 4), (4, 4), (8, 1)]:
        source = Lookup(matched)
        runtime = PromotionScheduler(target, source)
        before = state(target)
        control = runtime.enqueue(prefix, budget)
        assert runtime.prepare() == () and state(target) == before
        assert source.ended == [control] and not runtime.has_pending_work
    cache(target, req, 4)
    before = state(target)
    runtime = PromotionScheduler(target, Lookup())
    runtime.enqueue(prefix, 4)
    assert runtime.prepare() == () and state(target) == before


def test_cleanup_retry_cannot_republish_or_allocate_again() -> None:
    """After publication, source-cleanup failure must not recreate a reservation."""
    target, source = pool(), Lookup()
    runtime = PromotionScheduler(target, source)
    prefix = target.snapshot_external_prefix(request(list(range(8))), 8)
    control = runtime.enqueue(prefix, 4)
    assert control is not None and runtime.prepare()
    source.fail_cleanup = True
    with pytest.raises(TimeoutError):
        runtime.adopt(PromotionReceipt(control, True))
    before = state(target)
    assert runtime.prepare() == ()
    assert state(target) == before and not runtime.has_pending_work


def test_completion_timeout_retains_all_unreported_receipts() -> None:
    """An error in a later future cannot discard earlier terminal receipts."""
    tracker = PromotionCompletions()
    good, uncertain = Future(True, True), Future(False, True)
    uncertain.error = True
    tracker.record("success", good, object())
    tracker.record("failure", uncertain, object())
    with pytest.raises(TimeoutError):
        tracker.poll()
    assert tracker.contains("success") and tracker.contains("failure")
    uncertain.error = False
    assert tracker.poll() == (
        PromotionReceipt("success", True),
        PromotionReceipt("failure", False),
    )
    assert tracker.poll() == () and not tracker.pending


def test_pending_future_and_never_submitted_failure_are_distinct() -> None:
    """Only a proven never-submitted drop may produce an immediate false receipt."""
    tracker, future = PromotionCompletions(), Future()
    tracker.record("pending", future, object())
    tracker.record("unhealthy", None, None)
    assert tracker.poll() == (PromotionReceipt("unhealthy", False),)
    assert tracker.pending and tracker.contains("pending")
    future.ready = True
    assert tracker.poll() == (PromotionReceipt("pending", True),)


class Worker(LMCacheMPWorkerAdapter):
    """Real submission/page barriers with a fake transport; no constructor handshake."""

    def __init__(self, future: Future, healthy: bool = True) -> None:
        self.require_terminal_completion = True
        self.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 1, 1)
        self.lmcache_tokens_per_chunk = 4
        self.blocks_in_chunk = 2
        self.promotion_completions = PromotionCompletions()
        self._uncertain_promotion = None
        self._mq_timeout = 0
        self.healthy = healthy
        self.engine_group_infos = []
        self.instance_id = 17
        self.model_name = "unit-model"
        self.kv_caches = {}
        self.page_ownership = PendingKVPageAccess()
        self.transfer_ctx = MagicMock()
        self.transfer_ctx.submit_retrieve.return_value = future
        self.retrieve_futures = {}
        self.error_block_ids = set()

    @property
    def is_healthy(self) -> bool:
        """Return controlled heartbeat health without starting a server."""
        return self.healthy

    def _ensure_heartbeat_started(self) -> None:
        return None


class SchedulerConnector(LMCacheMPConnector):
    """Public connector binding with native HBM and controllable host lookup."""

    def __init__(self, target: BlockPool, source: Lookup) -> None:
        self._promotion_enabled = True
        self._group_tokens_per_block = [2]
        self.scheduler_adapter = source
        self.lazy_offload = False
        self.bind_gpu_block_pool(target)

    @property
    def role(self) -> KVConnectorRole:
        """Keep this fixture on the real scheduler-side public methods."""
        return KVConnectorRole.SCHEDULER


def test_connector_rejects_unrepresented_identity_before_lookup() -> None:
    """Per-request LMCache configs cannot be silently lost from a source key."""
    target, source = pool(), Lookup()
    connector = SchedulerConnector(target, source)
    req = request(list(range(8)))
    req.sampling_params.extra_args = {"kv_transfer_params": {"lmcache.tag": "tenant"}}
    before = state(target)
    with pytest.raises(ValueError, match="per-request"):
        connector.snapshot_promotion_prefix(req, 8)
    assert state(target) == before and not connector.has_pending_push_work()
    req.sampling_params.extra_args = None
    snapshot = connector.snapshot_promotion_prefix(req, 8)
    control = connector.queue_promotion(snapshot, 4)
    assert control is not None and connector.has_pending_push_work()
    connector.cancel_promotion(control)
    assert not connector.has_pending_push_work() and source.ended == [control]


def test_worker_skips_cached_pages_and_waits_for_physical_conflict() -> None:
    """Native MP keys carry the full prefix; scatter must not overwrite APC pages."""
    done, conflict = Future(), Future()
    worker = Worker(done)
    command = PromotionCommand("control", tuple(range(8)), (1, 2, 3, 4), 1, 2, 4)
    worker.page_ownership.record(conflict, {0: (3,)}, writes=False)
    producer_event = object()
    with pytest.raises(TimeoutError):
        worker.submit_promotion(command, producer_event)
    worker.transfer_ctx.submit_retrieve.assert_not_called()
    conflict.ready = True
    worker.submit_promotion(command, producer_event)
    args = worker.transfer_ctx.submit_retrieve.call_args
    key = args.args[1]
    assert key.token_ids == tuple(range(8)) and key.start == 0 and key.end == 8
    assert key.request_id == "control" and args.args[4] == [[1, 2, 3, 4]]
    assert args.args[5] is producer_event and args.kwargs["skip_first_n_tokens"] == 2
    assert worker.retrieve_futures == {} and worker.error_block_ids == set()
    assert worker.get_promotion_receipts() == ()
    # Cached prefix is safe for other writers as far as this DMA is concerned;
    # the scheduler's independent prefix pin handles allocator ownership.
    worker.page_ownership.wait({}, {0: (1,)}, timeout=0)
    with pytest.raises(TimeoutError):
        worker.page_ownership.wait({0: (2,)}, {}, timeout=0)
    done.ready = True
    assert worker.get_promotion_receipts() == (PromotionReceipt("control", True),)
    worker.page_ownership.reap()
    assert worker.page_ownership.pending_count == 0


def test_worker_unhealthy_and_unknown_submission_do_not_release_as_success() -> None:
    """Never-submitted failure is explicit; an uncertain submit remains fail-closed."""
    command = PromotionCommand("control", tuple(range(4)), (1, 2), 0, 2, 4)
    worker = Worker(Future(), healthy=False)
    worker.submit_promotion(command, object())
    worker.transfer_ctx.submit_retrieve.assert_not_called()
    assert worker.get_promotion_receipts() == (PromotionReceipt("control", False),)
    worker = Worker(Future())
    worker.transfer_ctx.submit_retrieve.side_effect = TimeoutError("ack lost")
    with pytest.raises(TimeoutError):
        worker.submit_promotion(command, object())
    assert worker.get_promotion_receipts() == ()
    with pytest.raises(ValueError):
        worker.submit_promotion(command, object())


class RunnerConnector(ActiveKVConnector):
    """Exercise actual no_forward ordering with a CPU connector double."""

    def __init__(self, delegate: Any) -> None:
        self.kv_connector = delegate
        self.vllm_config = None
        self._pending_load_start = False
        self._disabled = False


@pytest.mark.parametrize("empty", [True, False])
def test_control_submission_in_empty_and_normal_worker_steps(empty: bool) -> None:
    """An empty step skips STORE but must submit controls before polling output."""
    delegate = MagicMock(spec=KVConnectorBase_V1)
    delegate.get_finished.return_value = (set(), set())
    delegate.get_block_ids_with_load_errors.return_value = set()
    delegate.get_kv_connector_stats.return_value = None
    delegate.get_kv_connector_kv_cache_events.return_value = None
    receipt = PromotionReceipt("control", True)
    delegate.build_connector_worker_meta.return_value = LMCacheMPWorkerMetadata(
        completed_store_requests={},
        promotion_receipts=(receipt,),
    )
    wrapper = RunnerConnector(delegate)
    output = SimpleNamespace(
        kv_connector_metadata=LMCacheMPConnectorMetadata(),
        has_sync_kv_loads=False,
        finished_req_ids=set(),
    )
    module = "vllm.v1.worker.gpu.kv_connector."
    with (
        patch(module + "is_forward_context_available", return_value=True),
        patch(module + "get_forward_context", return_value=None),
    ):
        if empty:
            result = wrapper.no_forward(output).kv_connector_output
        else:
            wrapper.pre_forward(output)
            result = wrapper.post_forward(set())
    calls = [call[0] for call in delegate.mock_calls]
    assert ("wait_for_save" in calls) == (not empty)
    if not empty:
        assert calls.index("wait_for_save") < calls.index("submit_control")
    assert (
        calls.index("start_load_kv")
        < calls.index("submit_control")
        < calls.index("get_finished")
    )
    assert calls.index("build_connector_worker_meta") < calls.index(
        "clear_connector_metadata"
    )
    assert result.finished_recving == set() and result.finished_sending == set()
    assert result.kv_connector_worker_meta.promotion_receipts == (receipt,)
