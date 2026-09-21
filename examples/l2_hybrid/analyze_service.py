# SPDX-License-Identifier: Apache-2.0
"""Validate cross-process outputs and summarize observed native service costs."""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import re
import statistics


def native_totals(report: dict[str, Any]) -> dict[str, int]:
    """Sum native metric increments after measured workload arrival, not warmup."""
    result: dict[str, int] = defaultdict(int)
    for row in report["native_metric_rows"]:
        if row["observed_ns"] < report["origin_ns"]:
            continue
        iteration = row.get("iteration", {})
        for key in ("total", "computed", "local_cache_hit", "external_kv_transfer"):
            result["prompt_" + key] += iteration.get("prompt_tokens", {}).get(key, 0)
        result["preemptions"] += iteration.get("preemptions", 0)
        for source in ("prefix_cache", "connector_prefix_cache"):
            cache = row.get("scheduler", {}).get(source) or {}
            for field in ("queries", "hits", "requests"):
                result[source + "_" + field] += cache.get(field, 0)
    return dict(result)


def inspect_trace(directory: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Verify trace loss, request correlation, and actual receive-before-use."""
    files = list(directory.glob("scheduler-*.jsonl"))
    if len(files) != 1:
        raise RuntimeError("expected exactly one native scheduler trace")
    observations = [json.loads(line) for line in files[0].read_text().splitlines()]
    footer = observations[-1]
    if footer["event"] != "footer" or footer["dropped"] or footer["error"]:
        raise RuntimeError("incomplete scheduler trace")
    states = [r for r in observations if r["event"] == "scheduler_state"]
    if states[-1]["protected_blocks"] or states[-1]["requests"]:
        raise RuntimeError("scheduler did not release all requests/holds")
    result: dict[str, Any] = {
        "boundary_peaks": {
            key: max(len(r["category_blocks"][key]) for r in states)
            for key in states[0]["category_blocks"]
        },
        "protected_block_peak": max(len(r["protected_blocks"]) for r in states),
        "final_scheduler_state": states[-1],
        "request_correlations": [],
    }
    transfers = []
    if (directory / "events.jsonl").exists():
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        footer = events[-1]
        if (
            footer.get("record") != "footer"
            or footer["bus_dropped_events"]
            or footer["subscriber_errors"]
        ):
            raise RuntimeError("incomplete LMCache event trace")
        transfers = [
            row["metadata"] | {"request_id": row["session_id"]}
            for row in events
            if row.get("event") == "hybrid.access"
            and row["metadata"]["phase"] == "complete"
        ]
        counters: dict[str, int] = defaultdict(int)
        read_reservations: dict[str, int] = defaultdict(int)
        read_peaks: dict[str, int] = defaultdict(int)
        # Single-rank workload: each native reservation call owns one read lock.
        # These are logical reader lifetimes; DMA intervals may still serialize.
        for event in events:
            name = event.get("event")
            if name in ("l1.read.reserved", "l1.read.finished"):
                change = 1 if name == "l1.read.reserved" else -1
                for key in event["metadata"]["keys"]:
                    identity = json.dumps(key, sort_keys=True)
                    read_reservations[identity] += change
                    if read_reservations[identity] < 0:
                        raise RuntimeError("unbalanced logical reader trace")
                    read_peaks[identity] = max(
                        read_peaks[identity], read_reservations[identity]
                    )
            elif name == "l1.keys.evicted":
                for key in event["metadata"]["keys"]:
                    if read_reservations[json.dumps(key, sort_keys=True)]:
                        raise RuntimeError("observed eviction during a reader lifetime")
        if any(read_reservations.values()):
            raise RuntimeError("logical reader trace did not drain")
        result["logical_reader_peak_per_object"] = max(read_peaks.values(), default=0)
        result["shared_objects_with_multiple_readers"] = sum(
            value > 1 for value in read_peaks.values()
        )
        for transfer in transfers:
            if transfer["state"] != 3:
                raise RuntimeError("failed transfer in success workload")
            name = transfer["medium"] + "_" + transfer["direction"]
            counters[name + "_objects"] += 1
            counters[name + "_actual_bytes"] += transfer["actual_host_dma_bytes"]
        result["transfer_totals"] = dict(counters)
    for request in report["requests"]:
        matches = [
            row
            for row in observations
            if row["event"] == "core_enqueue"
            and re.fullmatch(
                re.escape(request["request_id"]) + r"-[0-9a-f]{8}", row["request_id"]
            )
        ]
        if len(matches) != 1:
            raise RuntimeError("ambiguous core request mapping")
        enqueue = matches[0]
        internal = enqueue["request_id"]
        scheduled = [
            row
            for row in observations
            if row["event"] == "schedule"
            and any(r["request_id"] == internal for r in row["requests"])
        ]
        receive = [
            row
            for row in observations
            if row["event"] == "scheduler_output_processed"
            and internal in row["finished_receiving"]
        ]
        loads = [
            r
            for r in transfers
            if r["request_id"] == internal and r["direction"] == "load"
        ]
        if loads:
            if len(receive) != 1 or not scheduled:
                raise RuntimeError(
                    "restored request lacks unique adoption/use evidence"
                )
            if max(r["logical_observed_ns"] for r in loads) > receive[0]["end_ns"]:
                raise RuntimeError("scheduler adopted incomplete data")
            if receive[0]["end_ns"] > scheduled[0]["begin_ns"]:
                raise RuntimeError("request scheduled before receive was processed")
        result["request_correlations"].append(
            {
                "public_id": request["request_id"],
                "engine_id": internal,
                "core_enqueue_bounds_ns": [enqueue["begin_ns"], enqueue["end_ns"]],
                "first_schedule_bounds_ns": [
                    scheduled[0]["begin_ns"],
                    scheduled[0]["end_ns"],
                ]
                if scheduled
                else None,
                "receive_adoption_bounds_ns": [
                    [r["begin_ns"], r["end_ns"]] for r in receive
                ],
                "load_media": sorted({r["medium"] for r in loads}),
                "first_gpu_compute_ns": None,
                "null_reason": "light trace records scheduling, not GPU execution",
            }
        )
    shared = []
    loads = [r for r in transfers if r["direction"] == "load"]
    for index, first in enumerate(loads):
        for second in loads[index + 1 :]:
            if (
                first["object_key"] != second["object_key"]
                or first["request_id"] == second["request_id"]
            ):
                continue
            overlap = min(
                first["logical_observed_ns"], second["logical_observed_ns"]
            ) - max(first["demand_ns"], second["demand_ns"])
            if overlap > 0:
                shared.append(
                    {
                        "request_ids": [first["request_id"], second["request_id"]],
                        "object_key": first["object_key"],
                        "overlap_ns": overlap,
                    }
                )
    result["shared_object_load_overlap"] = shared
    return result


def main() -> None:
    """Require every declared report, compare outputs, and write a new summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directories = list(args.run)
    if args.controls:
        plan = json.loads((args.controls / "plan.json").read_text())
        directories.extend(args.controls / f"r{p['round']}-{p['mode']}" for p in plan)
    references: dict[str, dict[str, Any]] = {}
    rows = []
    traces = {}
    for directory in directories:
        report = json.loads((directory / "report.json").read_text())
        if not report["passed"]:
            raise RuntimeError(f"failed report: {directory}")
        max_delta = 0.0
        for request in report["requests"]:
            key = report["workload_sha256"] + request["request_id"]
            previous = references.setdefault(key, request)
            if (
                request["input_token_sha256"] != previous["input_token_sha256"]
                or request["output_token_ids"] != previous["output_token_ids"]
            ):
                raise RuntimeError(
                    f"cross-run input/output differs: {directory}, {key}"
                )
            max_delta = max(
                max_delta,
                max(
                    abs(a - b)
                    for a, b in zip(
                        request["selected_token_logprobs"],
                        previous["selected_token_logprobs"],
                        strict=True,
                    )
                ),
            )
        if max_delta > 0.02:
            raise RuntimeError("cross-run logprob tolerance exceeded")
        rows.append(
            {
                "directory": str(directory),
                "mode": report["mode"],
                "trace": report["trace"],
                **report["summary"],
                "max_cross_run_logprob_delta": max_delta,
                "native_totals": native_totals(report),
            }
        )
        if report["trace"] != "off":
            traces[str(directory)] = inspect_trace(directory, report)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"] + "/" + row["trace"]].append(row)
    aggregates: dict[str, dict[str, Any]] = {}
    for key, group in grouped.items():
        aggregates[key] = {"processes": len(group)}
        for metric in (
            "mean_ttft_s",
            "p95_ttft_s_nearest_rank",
            "mean_tpot_s",
            "throughput_requests_s_including_think",
        ):
            values = [r[metric] for r in group]
            aggregates[key][metric] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "raw": values,
            }
    result = {
        "schema_version": 1,
        "passed": True,
        "runs": rows,
        "groups": aggregates,
        "traces": traces,
        "claim": "bounded mechanism workload; no significance or superiority claim",
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "runs": len(rows), "groups": list(grouped)}))


if __name__ == "__main__":
    main()
