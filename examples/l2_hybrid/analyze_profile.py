# SPDX-License-Identifier: Apache-2.0
"""Validate native batch-to-GPU-range correlation in a heavy profiler run."""

# Standard
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import argparse
import gzip
import hashlib
import json
import statistics


def expected_annotation(requests: list[dict[str, Any]]) -> str:
    """Derive the native annotation shape from actual scheduled prompt work."""
    context = [r for r in requests if r["scheduled_prompt_tokens"] > 0]
    generation = [r for r in requests if r["scheduled_prompt_tokens"] == 0]
    return (
        f"execute_context_{len(context)}"
        f"({sum(r['scheduled_tokens'] for r in context)})"
        f"_generation_{len(generation)}"
        f"({sum(r['scheduled_tokens'] for r in generation)})"
    )


def main() -> None:
    """Require a shape-consistent bijection before attaching real GPU intervals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    (profile_path,) = (args.run / "profile").glob("service_rank0*.json.gz")
    with gzip.open(profile_path, "rt") as source:
        profile = json.load(source)
    (scheduler_path,) = args.run.glob("scheduler-*.jsonl")
    records = [json.loads(line) for line in scheduler_path.read_text().splitlines()]
    batches = sorted(
        (
            r
            for r in records
            if r["event"] == "schedule" and r["begin_ns"] >= report["origin_ns"]
        ),
        key=lambda r: r["begin_ns"],
    )
    ranges = sorted(
        (r for r in profile["traceEvents"] if r.get("cat") == "gpu_user_annotation"),
        key=lambda r: r["ts"],
    )
    if len(batches) != len(ranges):
        raise RuntimeError("native scheduler and GPU annotation counts differ")
    rows = []
    durations: dict[str, list[float]] = defaultdict(list)
    for batch, gpu in zip(batches, ranges, strict=True):
        expected = expected_annotation(batch["requests"])
        if gpu["name"] != expected:
            raise RuntimeError(
                f"native batch shape mismatch: {expected} != {gpu['name']}"
            )
        rows.append(
            {
                "batch_id": batch["batch_id"],
                "requests": batch["requests"],
                "native_shape_annotation": expected,
                "scheduler_bounds_monotonic_ns": [batch["begin_ns"], batch["end_ns"]],
                "gpu_profiler_ts_us": gpu["ts"],
                "gpu_profiler_duration_us": gpu["dur"],
                "profiler_external_id": gpu["args"]["External id"],
            }
        )
        durations[expected].append(gpu["dur"])
    summary = {
        key: {
            "samples": len(values),
            "mean_us": statistics.mean(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for key, values in durations.items()
    }
    result = {
        "schema_version": 1,
        "passed": True,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "profile_file": str(profile_path),
        "clock": "native profiler microseconds; original baseTimeNanoseconds preserved",
        "profiler_base_time_ns": profile.get("baseTimeNanoseconds"),
        "correlation": "ordered native batches; exact counts and per-batch shapes",
        "rows": rows,
        "shape_summary": summary,
        "raw_event_categories": dict(
            Counter(r.get("cat", "metadata") for r in profile["traceEvents"])
        ),
        "scope": "heavy full-model GPU ranges; not independent primitive estimates",
        "mp_payload_dma": None,
        "mp_dma_null_reason": "LMCache copies execute in a separate unprofiled process",
    }
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "batches": len(rows), "shapes": len(summary)}))


if __name__ == "__main__":
    main()
