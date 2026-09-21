# SPDX-License-Identifier: Apache-2.0
"""Correlate observed host/model intervals without inventing GPU timestamps."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json


def main() -> None:
    """Write a fresh Perfetto-compatible Chrome timeline from one service run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    origin = report["origin_ns"]
    events: list[dict[str, Any]] = []

    def span(
        name: str,
        process: int,
        track: str,
        begin: int,
        end: int,
        fields: dict[str, Any],
        category: str = "observed_host",
    ) -> None:
        if end < begin:
            raise ValueError("negative observed interval")
        events.append(
            {
                "name": name,
                "cat": category,
                "ph": "X",
                "pid": process,
                "tid": track,
                "ts": (begin - origin) / 1000,
                "dur": (end - begin) / 1000,
                "args": fields,
            }
        )

    for row in report["requests"]:
        request = row["request_id"]
        span(
            "request",
            1,
            request,
            row["actual_arrival_ns"],
            row["completed_ns"],
            {"prompt": row["prompt"], "cached_tokens": row["num_cached_tokens"]},
        )
        span(
            "TTFT",
            2,
            request,
            row["actual_arrival_ns"],
            row["first_nonempty_text_ns"],
            {},
        )
        for chunk in row["outputs"]:
            events.append(
                {
                    "name": "stream output",
                    "ph": "i",
                    "s": "t",
                    "pid": 2,
                    "tid": request,
                    "ts": (chunk["observed_ns"] - origin) / 1000,
                    "args": {"token_ids": chunk["token_ids"], "text": chunk["text"]},
                }
            )
    for path in args.run.glob("scheduler-*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["event"] in (
                "core_enqueue",
                "schedule",
                "scheduler_output_processed",
            ):
                track = "native scheduler"
                span(row["event"], 3, track, row["begin_ns"], row["end_ns"], row)
                if row["event"] == "schedule":
                    for request in row["requests"]:
                        span(
                            "scheduled (not GPU)",
                            4,
                            request["request_id"],
                            row["begin_ns"],
                            row["end_ns"],
                            request,
                        )
    path = args.run / "events.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header = rows[0]
        offset = header["clock_anchor_monotonic_ns"] - header["clock_anchor_wall_ns"]
        for row in rows:
            if row.get("record") != "event":
                continue
            meta = row["metadata"]
            if row["event"] == "hybrid.access" and meta["phase"] == "complete":
                track = meta["transfer_key"] + "/chunk" + str(meta["chunk_index"])
                span(
                    meta["medium"] + " " + meta["direction"] + " ownership",
                    5,
                    track,
                    meta["demand_ns"],
                    meta["logical_observed_ns"],
                    meta,
                )
                if meta.get("modeled_service_ns") is not None:
                    end = meta["modeled_complete_ns"]
                    span(
                        "CXLMemSim service (not DMA)",
                        6,
                        track,
                        end - meta["modeled_service_ns"],
                        end,
                        meta,
                        "simulated_model",
                    )
            else:
                stamp = next(
                    (
                        meta[name]
                        for name in (
                            "cpu_submitted_ns",
                            "observed_ns",
                            "reply_ready_ns",
                        )
                        if name in meta
                    ),
                    None,
                )
                approximate = stamp is None
                if stamp is None:
                    stamp = round(row["timestamp_s"] * 1e9) + offset
                events.append(
                    {
                        "name": row["event"],
                        "ph": "i",
                        "s": "t",
                        "pid": 7,
                        "tid": row["session_id"] or "directory",
                        "ts": (stamp - origin) / 1000,
                        "args": meta | {"wall_anchor_conversion": approximate},
                    }
                )
    result = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source_run": str(args.run),
            "monotonic_origin_ns": origin,
            "gpu_intervals": None,
            "gpu_null_reason": "use separately captured raw native profiler GPU trace",
            "wall_conversion": "paired header anchor; wall events are approximate",
            "scheduler_adoption": "inside scheduler_output_processed host interval",
        },
    }
    with args.output.open("x") as output:
        output.write(json.dumps(result) + "\n")
    print(json.dumps({"events": len(events), "output": str(args.output)}))


if __name__ == "__main__":
    main()
