# SPDX-License-Identifier: Apache-2.0
"""Audit independent native compute coverage using actual scheduler batches."""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import re
import statistics

# Third Party
from analyze_service import native_totals


def distribution(values: list[float]) -> dict[str, float | int]:
    """Return sample count and observed range/median in the input unit."""
    return {
        "samples": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def main() -> None:
    """Validate workload provenance, zero reuse, coverage, and profiler mapping.

    Raises:
        RuntimeError: If any input, request, or computed-token audit fails.
        FileExistsError: If the requested output already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    report = json.loads((args.run / "report.json").read_text())
    profile = json.loads((args.run / "profile-analysis.json").read_text())
    if (
        not report["passed"]
        or not profile["passed"]
        or report["mode"] != "hbm"
        or report["workload_sha256"] != hashlib.sha256(workload_bytes).hexdigest()
    ):
        raise RuntimeError("independent workload provenance failed")
    requests = {row["request_id"]: row for row in report["requests"]}
    expected = {row["request_id"]: row for row in workload["requests"]}
    if set(requests) != set(expected):
        raise RuntimeError("request coverage differs from frozen input")
    computed: dict[str, int] = defaultdict(int)
    mappings: dict[str, str] = {}
    groups: dict[str, list[float]] = defaultdict(list)
    batches = []
    for batch in profile["rows"]:
        members = []
        for member in batch["requests"]:
            internal = member["request_id"]
            matches = [
                key
                for key in requests
                if re.fullmatch(re.escape(key) + r"-[0-9a-f]{8}", internal)
            ]
            if len(matches) != 1:
                raise RuntimeError("ambiguous native compute request mapping")
            public = matches[0]
            if public in mappings and mappings[public] != internal:
                raise RuntimeError("multiple engine requests share one public ID")
            mappings[public] = internal
            computed[public] += member["scheduled_prompt_tokens"]
            record = requests[public]
            members.append(
                {
                    "public_id": public,
                    "context_tokens": member["computed_context_tokens"],
                    "prompt_tokens": member["prompt_tokens"],
                    "scheduled_prompt_tokens": member["scheduled_prompt_tokens"],
                    **{
                        key: record[key]
                        for key in (
                            "cost_context_tokens",
                            "requested_batch",
                            "cost_role",
                            "cost_repeat",
                            "temperature_state",
                        )
                    },
                }
            )
        signature = {
            "native_shape": batch["native_shape_annotation"],
            "actual_batch": len(members),
            "input_contexts": sorted({r["cost_context_tokens"] for r in members}),
            "roles": sorted({r["cost_role"] for r in members}),
            "temperature_states": sorted({r["temperature_state"] for r in members}),
        }
        groups[json.dumps(signature, sort_keys=True)].append(
            batch["gpu_profiler_duration_us"]
        )
        batches.append({**batch, "input_members": members})
    coverage = []
    for public, request in requests.items():
        source = expected[public]
        prompt = workload["prompts"][source["prompt"]]
        if (
            not request["finished"]
            or request["num_cached_tokens"] != 0
            or request["input_token_sha256"] != prompt["token_sha256"]
            or computed[public] != len(prompt["token_ids"])
            or len(request["output_token_ids"]) != source["max_output_tokens"]
        ):
            raise RuntimeError(f"compute/reuse audit failed for {public}")
        coverage.append(
            {
                "public_id": public,
                "engine_id": mappings[public],
                "computed_prompt_tokens": computed[public],
                "output_tokens": len(request["output_token_ids"]),
                "cost_role": source["cost_role"],
                "requested_batch": source["requested_batch"],
                "repeat": source["cost_repeat"],
            }
        )
    totals = native_totals(report)
    if totals["prompt_computed"] != sum(computed.values()):
        raise RuntimeError("native metric and actual schedule totals disagree")
    result: dict[str, Any] = {
        "schema_version": 1,
        "passed": True,
        "workload_sha256": report["workload_sha256"],
        "native_totals": totals,
        "request_coverage": coverage,
        "batch_rows": batches,
        "groups": [
            {**json.loads(key), "gpu_model_range_us": distribution(values)}
            for key, values in groups.items()
        ],
        "total_generated_tokens": sum(r["output_tokens"] for r in coverage),
        "fit": None,
        "fit_null_reason": (
            "compute contexts are measured independently; no model fitted to TTFT"
        ),
        "scope": (
            "heavy native model ranges include worker preparation and inference; "
            "not pure kernel service time or trace-off performance"
        ),
        "cold_scope": (
            "first occurrence of each requested shape after engine warmup; "
            "not a cold model or cold CUDA context"
        ),
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "requests": len(coverage),
                "batches": len(batches),
                "computed_prompt_tokens": sum(computed.values()),
                "groups": len(groups),
            }
        )
    )


if __name__ == "__main__":
    main()
