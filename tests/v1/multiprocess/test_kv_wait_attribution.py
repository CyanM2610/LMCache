# SPDX-License-Identifier: Apache-2.0
"""Diagnostic intervals must neither double-count TTFT nor invent history."""

import gzip
import json
import unittest
from io import StringIO
from unittest.mock import patch

from lmcache.integration.kv_wait_probe.analyze import (
    INGRESS_MILESTONES,
    Timeline,
    group_request_events,
    ingress_intervals,
    partition,
)
from lmcache.integration.kv_wait_probe.workload import actual_input


class WaitAttributionTests(unittest.TestCase):
    def test_crossing_cpu_observations_have_distinct_tracks_and_unchanged_times(self):
        stream = StringIO()
        host_args = {"request_id": "chat:a:0", "pid": 42, "tid": 7}
        with patch.object(gzip, "open", return_value=stream):
            timeline = Timeline("unused.json.gz", 0)
            timeline.add("model_forward", 1000, 4000, pid=42, tid=7, args=host_args)
            timeline.add(
                "model_forward",
                2000,
                5000,
                pid=42,
                tid=7,
                args={"request_id": "chat:a:0"},
                cat="nvtx",
            )
            timeline.add("next_step", 5000, 6000, pid=42, tid=7)
            payload = stream.getvalue() + '],"displayTimeUnit":"ms"}'
            timeline.close()
        events = json.loads(payload)["traceEvents"]
        host, nvtx, next_step = [r for r in events if r["ph"] == "X"]
        self.assertNotEqual(host["pid"], nvtx["pid"])
        self.assertNotEqual(host["tid"], nvtx["tid"])
        self.assertEqual(
            (host["pid"], host["tid"]), (next_step["pid"], next_step["tid"])
        )
        self.assertEqual((host["ts"], host["dur"]), (1.0, 3.0))
        self.assertEqual((nvtx["ts"], nvtx["dur"]), (2.0, 3.0))
        for row in (host, nvtx):
            self.assertIsInstance(row["pid"], int)
            self.assertIsInstance(row["tid"], int)
            self.assertEqual(row["args"]["request_id"], "chat:a:0")
            self.assertEqual(
                (row["args"]["real_pid"], row["args"]["real_tid"]), (42, 7)
            )
        self.assertNotIn("real_pid", host_args)
        self.assertEqual(len([r for r in events if r["name"] == "process_name"]), 2)

    def test_logical_api_and_pre_randomization_events_join_native_chain(self) -> None:
        events = [
            dict(event="engine_request_received", request_id="native-4", ts_ns=30),
            dict(event="api_request_received", request_id="chat:a:0", ts_ns=10),
            dict(
                event="engine_input_process_begin",
                request_id="preliminary-2",
                logical_id="chat:a:0",
                ts_ns=20,
            ),
            dict(
                event="input_known",
                request_id="native-4",
                logical_id="chat:a:0",
                ts_ns=40,
            ),
            dict(event="api_request_received", request_id="chat:b:0", ts_ns=15),
        ]
        aliases, grouped = group_request_events(events)
        self.assertEqual(aliases, {"native-4": "chat:a:0"})
        self.assertEqual([e["ts_ns"] for e in grouped["chat:a:0"]], [10, 20, 30, 40])
        self.assertEqual(len(grouped["chat:b:0"]), 1)
        self.assertNotIn("preliminary-2", grouped)

    def test_overlapping_ipc_completion_is_not_double_charged(self) -> None:
        milestones = {
            name: 10 + index * 5 for index, name in enumerate(INGRESS_MILESTONES)
        }
        milestones.update(
            dispatch=0,
            input_known=100,
            engine_ipc_submit_end=80,
            engine_socket_received=53,
        )
        intervals, detail = ingress_intervals(milestones)
        result = partition(0, 100, intervals)
        durations = {r["stage"]: r["end_ns"] - r["start_ns"] for r in result}
        self.assertEqual(sum(durations.values()), 100)
        self.assertEqual(durations["ipc_submission_and_delivery"], 8)
        self.assertEqual(durations["engine_message_decode"], 2)
        self.assertTrue(detail["client_ipc_overlaps_decoded_receipt"])
        self.assertEqual(detail["client_ipc_span_ms"], 35 / 1e6)
        self.assertEqual(detail["missing"], [])
        self.assertEqual(detail["order_violations"], [])
        self.assertNotIn("ingress_unexplained", durations)

    def test_missing_ingress_boundary_leaves_hole_without_fabricating_work(
        self,
    ) -> None:
        milestones = {
            name: 10 + index * 5 for index, name in enumerate(INGRESS_MILESTONES)
        }
        milestones.update(dispatch=0, input_known=100)
        del milestones["api_body_received"]
        intervals, detail = ingress_intervals(milestones)
        result = partition(0, 100, intervals)
        self.assertEqual(sum(r["end_ns"] - r["start_ns"] for r in result), 100)
        self.assertEqual(
            [
                (r["start_ns"], r["end_ns"])
                for r in result
                if r["stage"] == "ingress_unexplained"
            ],
            [(10, 20)],
        )
        self.assertEqual(detail["missing"], ["api_body_received"])
        self.assertNotIn("api_parse_validate_dispatch", [r["stage"] for r in result])

    def test_old_trace_remains_explicitly_coarse(self) -> None:
        intervals, detail = ingress_intervals(dict(dispatch=10, input_known=100))
        self.assertEqual(
            partition(10, 100, intervals),
            [dict(start_ns=10, end_ns=100, stage="ingress_and_preprocessing")],
        )
        self.assertEqual(detail["mode"], "legacy_coarse")
        self.assertEqual(len(detail["missing"]), len(INGRESS_MILESTONES))
        self.assertIsNone(detail["client_ipc_span_ms"])

    def test_overlapping_waits_conserve_time_and_keep_unknown_gap(self):
        result = partition(0, 100, [(10, 70, "lookup", 1), (40, 80, "capacity", 2)])
        self.assertEqual(
            [(r["stage"], r["end_ns"] - r["start_ns"]) for r in result],
            [
                ("unexplained", 10),
                ("lookup", 30),
                ("capacity", 40),
                ("unexplained", 20),
            ],
        )
        self.assertEqual(sum(r["end_ns"] - r["start_ns"] for r in result), 100)

    def test_child_uses_actual_returned_tokens_and_rejects_divergence(self):
        child = dict(parent="p", suffix_tokens=[9], input_tokens=[1, 2, 3, 9])
        result = dict(p=dict(status="ok", input_tokens=[1, 2], output_tokens=[3]))
        self.assertEqual(actual_input(child, result), [1, 2, 3, 9])
        result["p"]["output_tokens"] = [7]
        with self.assertRaisesRegex(ValueError, "history diverges"):
            actual_input(child, result)


if __name__ == "__main__":
    unittest.main()
