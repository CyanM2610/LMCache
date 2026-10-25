# SPDX-License-Identifier: Apache-2.0
"""Actual connector/tracker methods using the project's CPU transport fixtures.

The companion vLLM tests exercise allocation and whole scheduler trajectories.
These tests target the connector's distinction between physical compute progress,
known lower-cache coverage and the original lookup reference obligation.
"""

# Standard
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(os.environ["PREFILL_PROJECT_ROOT"])
RUNTIME = Path(os.environ["PREFILL_NATIVE_RUNTIME"])
sys.path.insert(0, str(ROOT / "scripts/kv_wait_probe/global_plan"))
import cpu_support as support  # noqa: E402


def fixture():
    """Load actual native connector methods, mocking only transport/CPU hardware."""
    value = support.fixture(
        runtime=RUNTIME, global_plan=False, control=False, store=False,
        prepare_load=False,
    )
    path = RUNTIME / "source/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py"
    support.ops.add_methods(
        path, "LMCacheMPConnector", type(value.connector),
        {"validate_recompute_path", "commit_recompute_path"},
        value.connector.build_connector_meta.__globals__,
    )
    return value


def complete_match(f, *, length=4096, hbm=1024, lower=3072):
    """Deliver ACK/STATUS through the real adapter, then commit a native match."""
    request = support.t.enqueue(f, "r", length)
    support.t.known_result(f, request, lower // 256)
    result = f.connector.match_prefix_tokens(
        request, hbm, allow_lookup_submit=False,
        allow_lookup_progress=False, commit_match=True,
    )
    return request, result


def adopt(f, request, *, hbm, recompute):
    """Exercise real connector adoption with a supplied native block table."""
    pages = (len(request.prompt_token_ids) + 15) // 16
    blocks = SimpleNamespace(
        get_block_ids=lambda: (list(range(1, pages + 1)),)
    )
    if recompute:
        f.connector.commit_recompute_path(request, hbm)
    lower = f.connector.request_trackers[request.request_id].num_lmcache_hit_tokens
    external = max(0, lower - hbm)
    f.connector.update_state_after_alloc(
        request, blocks, 0 if recompute else external,
    )
    return f.connector.request_trackers[request.request_id]


class RecomputeConnectorTests(unittest.TestCase):
    def tearDown(self):
        support.close_fixtures()

    def metadata(self, f, tracker):
        return f.connector._fixture_code.LMCacheMPRequestMetadata.PeekStoreMetadata(
            tracker, 256, [16],
        )

    def test_default_restore_accounting_is_unchanged(self):
        f = fixture()
        request, result = complete_match(f)
        self.assertEqual(result, (2048, True))
        tracker = adopt(f, request, hbm=1024, recompute=False)
        self.assertIsNone(tracker.local_compute_start_tokens)
        self.assertEqual(tracker.state.name, "WAITING_FOR_LOAD")
        cls = f.connector._fixture_code.LMCacheMPRequestMetadata
        load = cls.GetRetrieveMetadata(tracker, 256, [16])
        self.assertEqual((load.op.start, load.op.end), (1024, 3072))
        tracker.increase_num_scheduled_tokens(1024)
        saved = self.metadata(f, tracker)
        self.assertEqual((saved.op.start, saved.op.end), (3072, 4096))

    def test_recompute_keeps_physical_progress_separate_from_saved_prefix(self):
        f = fixture()
        request, _ = complete_match(f)
        tracker = adopt(f, request, hbm=1024, recompute=True)
        cls = f.connector._fixture_code.LMCacheMPRequestMetadata
        self.assertEqual(tracker.state.name, "READY")
        self.assertIsNone(cls.GetRetrieveMetadata(tracker, 256, [16]))
        self.assertEqual(tracker.num_stored_tokens, 3072)
        tracker.increase_num_scheduled_tokens(1024)
        self.assertIsNone(self.metadata(f, tracker))
        tracker.increase_num_scheduled_tokens(1024)
        self.assertIsNone(self.metadata(f, tracker))
        tracker.increase_num_scheduled_tokens(256)
        saved = self.metadata(f, tracker)
        self.assertEqual((saved.op.start, saved.op.end), (3072, 3328))
        cls.CommitStoreMetadata(tracker, saved, 256, [16])
        tracker.increase_num_scheduled_tokens(768)
        tail = self.metadata(f, tracker)
        self.assertEqual((tail.op.start, tail.op.end), (3328, 4096))
        self.assertEqual(tracker.recompute_lower_prefix_tokens, 3072)

    def test_zero_hbm_hit_is_not_replaced_by_lower_physical_progress(self):
        f = fixture()
        request, _ = complete_match(f, hbm=0)
        tracker = adopt(f, request, hbm=0, recompute=True)
        tracker.increase_num_scheduled_tokens(2048)
        self.assertIsNone(self.metadata(f, tracker))
        self.assertEqual(tracker.local_compute_start_tokens, 0)

    def test_hbm_hit_lower_miss_keeps_native_store_gap_repair(self):
        f = fixture()
        request, result = complete_match(f, lower=0)
        self.assertEqual(result, (0, False))
        tracker = adopt(f, request, hbm=1024, recompute=True)
        tracker.increase_num_scheduled_tokens(256)
        saved = self.metadata(f, tracker)
        self.assertEqual((saved.op.start, saved.op.end), (0, 1280))

    def test_partial_block_and_last_token_use_actual_local_start(self):
        for lower in (0, 256):
            with self.subTest(lower=lower):
                f = fixture()
                request, result = complete_match(f, length=256, hbm=255, lower=lower)
                self.assertEqual(result, (0, False))
                tracker = adopt(f, request, hbm=255, recompute=True)
                self.assertEqual(tracker.num_vllm_hit_tokens, 240)
                self.assertEqual(tracker.local_compute_start_tokens, 255)
                tracker.increase_num_scheduled_tokens(1)
                saved = self.metadata(f, tracker)
                if lower:
                    self.assertIsNone(saved)
                else:
                    self.assertEqual((saved.op.start, saved.op.end), (0, 256))

    def test_committed_recompute_cannot_reenter_failed_load_or_new_lookup(self):
        f = fixture()
        request, _ = complete_match(f, hbm=0)
        tracker = adopt(f, request, hbm=0, recompute=True)
        calls = list(f.rpc.calls)
        for _ in range(3):
            self.assertEqual(
                f.connector.get_num_new_matched_tokens(request, 0), (0, False)
            )
        self.assertEqual(f.rpc.calls, calls)
        self.assertEqual(tracker.state.name, "READY")
        self.assertEqual(tracker.num_stored_tokens, 3072)

    def test_pending_or_uncommitted_match_cannot_commit_path(self):
        f = fixture()
        request = support.t.enqueue(f, "r", 4096)
        self.assertEqual(
            f.connector.get_num_new_matched_tokens(request, 0), (None, True)
        )
        with self.assertRaisesRegex(ValueError, "complete valid lookup"):
            f.connector.commit_recompute_path(request, 0)
        support.ops.ack(f.rpc, "r")
        f.adapter.check_lookup_result("r")
        support.ops.status(f.rpc, 12, "r")
        f.adapter.check_lookup_result("r")
        with self.assertRaisesRegex(ValueError, "current committed match"):
            f.connector.commit_recompute_path(request, 0)
        self.assertIsNone(f.connector.request_trackers["r"].local_compute_start_tokens)

    def test_validation_reads_only_scalars_and_does_not_commit(self):
        f = fixture()
        request, _ = complete_match(f)
        before = support.t.signature(f, [])
        tracker = f.connector.request_trackers["r"]
        for _ in range(3):
            f.connector.validate_recompute_path(request, 1024)
        self.assertEqual(support.t.signature(f, []), before)
        self.assertIsNone(tracker.local_compute_start_tokens)
        self.assertIsNone(tracker.recompute_lower_prefix_tokens)

    def test_finished_request_rejects_commit_and_does_not_revive_query(self):
        f = fixture()
        request, _ = complete_match(f)
        adopt(f, request, hbm=1024, recompute=True)
        f.scheduler.finish_requests("r", f.rt.status.FINISHED_ABORTED)
        self.assertNotIn("r", f.connector.request_trackers)
        self.assertFalse(f.connector.lookup_operation_state("r")["submitted"])
        calls = list(f.rpc.calls)
        with self.assertRaisesRegex(ValueError, "live native request tracker"):
            f.connector.validate_recompute_path(request, 1024)
        with self.assertRaisesRegex(ValueError, "live native request tracker"):
            f.connector.commit_recompute_path(request, 1024)
        self.assertEqual(f.rpc.calls, calls)

    def test_commit_rejects_existing_adoption_and_conflicting_external_quota(self):
        f = fixture()
        request, _ = complete_match(f)
        f.connector.commit_recompute_path(request, 1024)
        blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2],))
        with self.assertRaisesRegex(ValueError, "cannot allocate external"):
            f.connector.update_state_after_alloc(request, blocks, 2048)
        self.assertEqual(f.connector.request_trackers["r"].allocated_block_ids, {})
        with self.assertRaisesRegex(ValueError, "first resource adoption"):
            f.connector.commit_recompute_path(request, 1024)

    def test_readmission_preserves_only_original_saved_copy_and_refreshes_hbm(self):
        f = fixture()
        request, _ = complete_match(f)
        old = adopt(f, request, hbm=1024, recompute=True)
        old.increase_num_scheduled_tokens(2304)
        cls = f.connector._fixture_code.LMCacheMPRequestMetadata
        saved = self.metadata(f, old)
        cls.CommitStoreMetadata(old, saved, 256, [16])
        self.assertEqual(old.num_stored_tokens, 3328)
        request.status = type(request.status).PREEMPTED
        calls = list(f.rpc.calls)
        self.assertEqual(f.connector.match_prefix_tokens(
            request, 512, allow_lookup_submit=False,
            allow_lookup_progress=False, commit_match=True,
        ), (0, False))
        tracker = f.connector.request_trackers["r"]
        self.assertIsNot(tracker, old)
        self.assertEqual(tracker.num_stored_tokens, 3072)
        self.assertEqual(tracker.num_scheduled_tokens, 0)
        self.assertEqual(tracker.local_compute_start_tokens, 512)
        self.assertEqual(f.rpc.calls, calls)
        f.connector.commit_recompute_path(request, 512)
        blocks = SimpleNamespace(get_block_ids=lambda: (list(range(1, 129)),))
        f.connector.update_state_after_alloc(request, blocks, 0)
        self.assertEqual(f.rpc.calls, calls)
        self.assertEqual(tracker.state.name, "READY")
        tracker.increase_num_scheduled_tokens(1536)
        self.assertIsNone(self.metadata(f, tracker))


if __name__ == "__main__":
    unittest.main()
