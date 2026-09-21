# SPDX-License-Identifier: Apache-2.0
"""Audit independent copy samples, frozen-fit holdouts and real GPU overlap."""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import math
import statistics


def distribution(values: list[float]) -> dict[str, Any]:
    """Keep raw observations alongside descriptive statistics; no clipping."""
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(values)) - 1],
        "raw": values,
    }


def summarize(directory: Path) -> dict[str, Any]:
    """Require the frozen case counts, finite timings, bytes and final drain."""
    report = json.loads((directory / "report.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    if not report["passed"] or report["active_after_drain"]:
        raise RuntimeError("primitive run failed or did not drain")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in report["samples"]:
        if not row["warmup"]:
            grouped[row["case_id"]].append(row)
    if len(grouped) != len(plan["cases"]):
        raise RuntimeError("missing primitive case")
    cases = []
    for case_id, definition in enumerate(plan["cases"]):
        rows = grouped[case_id]
        if len(rows) != plan["measured_repetitions"]:
            raise RuntimeError("wrong repetition count")
        for row in rows:
            if row["operations"] != definition["operations"]:
                raise RuntimeError("operation differs from plan")
            if any(
                not math.isfinite(branch["gpu_event_elapsed_us"])
                or branch["gpu_event_elapsed_us"] <= 0
                for branch in row["branches"]
            ):
                raise RuntimeError("invalid CUDA duration")
        cases.append(
            {
                "case_id": case_id,
                **definition,
                "gpu_envelope_us": distribution([r["gpu_envelope_us"] for r in rows]),
                "cpu_submission_us": distribution(
                    [(r["cpu_submitted_ns"] - r["cpu_begin_ns"]) / 1000 for r in rows]
                ),
                "branches": [
                    {
                        "operation": operation,
                        "gpu_event_us": distribution(
                            [r["branches"][index]["gpu_event_elapsed_us"] for r in rows]
                        ),
                    }
                    for index, operation in enumerate(definition["operations"])
                ],
            }
        )
    for check in report["content_checks"]:
        case = plan["cases"][check["case_id"]]
        expected = sum(op != "compute" for op in case["operations"])
        if len(check["checks"]) != expected or not all(
            c["equal"] and c["bytes"] == case["bytes"] for c in check["checks"]
        ):
            raise RuntimeError("primitive byte checks incomplete")
    return {
        "directory": str(directory),
        "cases": cases,
        "cold_stage_ns": report["cold_stage_ns"],
        "gemm_shape_mnk": report["gemm_shape_mnk"],
        "provenance": report["provenance"],
    }


def interval_overlap(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> float:
    """Sum intersections of same-stream nonoverlapping activities across branches."""
    return sum(
        max(0.0, min(a["ts"] + a["dur"], b["ts"] + b["dur"]) - max(a["ts"], b["ts"]))
        for a in first
        for b in second
    )


def inspect_gpu_trace(directory: Path) -> dict[str, Any]:
    """Associate actual CUPTI memcpy/kernel activities with primitive ranges."""
    profile = json.loads((directory / "profile.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    samples = {(r["case_id"], r["repeat"]): r for r in report["samples"]}
    annotations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in profile["traceEvents"]:
        if event.get("cat") != "gpu_user_annotation" or not event["name"].startswith(
            "primitive_case_"
        ):
            continue
        fields = event["name"].split("_")
        annotations[int(fields[2]), int(fields[4])].append(event)
    solo_streams = {
        events[0]["tid"]
        for key, events in annotations.items()
        if len(samples[key]["operations"]) == 1
    }
    if len(solo_streams) != 1:
        raise RuntimeError("cannot identify the owned primary CUDA stream")
    (first_stream,) = solo_streams
    device_events = [
        e for e in profile["traceEvents"] if e.get("cat") in ("gpu_memcpy", "kernel")
    ]
    audited = []
    overlaps: dict[str, list[float]] = defaultdict(list)
    for key, sample in samples.items():
        if sample["warmup"]:
            continue
        ranges = annotations[key]
        if len(ranges) != len(sample["operations"]):
            raise RuntimeError("missing GPU branch annotation")
        branch_events = []
        branches = []
        for index, operation in enumerate(sample["operations"]):
            (window,) = [
                w for w in ranges if (w["tid"] == first_stream) == (index == 0)
            ]
            activities = [
                e
                for e in device_events
                if e["tid"] == window["tid"]
                and window["ts"] - 0.1 <= e["ts"]
                and e["ts"] + e["dur"] <= window["ts"] + window["dur"] + 0.1
            ]
            copies = [e for e in activities if e["cat"] == "gpu_memcpy"]
            kernels = [e for e in activities if e["cat"] == "kernel"]
            if operation == "compute":
                if not kernels or copies:
                    raise RuntimeError("GEMM branch lacks pure compute activities")
            elif sum(e["args"]["bytes"] for e in copies) != sample["bytes"] or kernels:
                raise RuntimeError("direct copy byte/operation attribution failed")
            branch_events.append(activities)
            branches.append(
                {
                    "operation": operation,
                    "activities": activities,
                    "gpu_active_us": sum(e["dur"] for e in activities),
                }
            )
        overlap = interval_overlap(*branch_events) if len(branch_events) == 2 else 0.0
        overlaps[" + ".join(sample["operations"])].append(overlap)
        audited.append(
            {
                "case_id": key[0],
                "repeat": key[1],
                "branches": branches,
                "actual_device_overlap_us": overlap,
            }
        )
    return {
        "profiler_base_time_ns": profile.get("baseTimeNanoseconds"),
        "samples": audited,
        "overlap_us": {key: distribution(values) for key, values in overlaps.items()},
        "clock": (
            "original CUPTI profiler microseconds; activities, not host event brackets"
        ),
    }


def main() -> None:
    """Validate frozen-fit provenance, holdout error and actual GPU overlap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text())
    held_raw = json.loads((args.holdout / "report.json").read_text())
    if (
        hashlib.sha256((args.training / "report.json").read_bytes()).hexdigest()
        != fit["training_sha256"]
    ):
        raise RuntimeError("training data changed after fit freeze")
    if (
        held_raw["frozen_fit_sha256"]
        != hashlib.sha256(args.fit.read_bytes()).hexdigest()
        or held_raw["started_wall_ns"] <= fit["frozen_wall_ns"]
    ):
        raise RuntimeError("holdout did not use the previously frozen fit")
    training, holdout, heavy = (
        summarize(path) for path in (args.training, args.holdout, args.heavy)
    )
    checks = []
    for case in holdout["cases"]:
        if case["kind"] != "solo_copy":
            continue
        (operation,) = case["operations"]
        coefficients = fit["fits"][operation]
        predicted = (
            coefficients["intercept_us"]
            + coefficients["slope_us_per_byte"] * case["bytes"]
        )
        observed = case["branches"][0]["gpu_event_us"]["median"]
        checks.append(
            {
                "operation": operation,
                "bytes": case["bytes"],
                "predicted_us": predicted,
                "observed_median_us": observed,
                "relative_error": (predicted - observed) / observed,
            }
        )
    result = {
        "schema_version": 1,
        "passed": True,
        "physical_cxl": False,
        "training": training,
        "holdout": holdout,
        "heavy": heavy,
        "heldout_checks": checks,
        "gpu_trace": inspect_gpu_trace(args.heavy),
        "fit_retrained": False,
        "accuracy_threshold": None,
        "scope": (
            "measurement/correlation checks pass; "
            "accuracy errors retained without a fit-quality claim"
        ),
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "heldout_checks": len(checks),
                "maximum_absolute_relative_error": max(
                    abs(r["relative_error"]) for r in checks
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
