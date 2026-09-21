# SPDX-License-Identifier: Apache-2.0
"""Freeze a simple copy-size cost model before independent held-out execution."""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import statistics
import time


def main() -> None:
    """Fit only declared training medians; never read service or holdout results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.training / "report.json"
    report = json.loads(source.read_text())
    if not report["passed"] or report["holdout"] or report["heavy_profiler"]:
        raise ValueError("fit requires successful, unprofiled training samples")
    samples: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in report["samples"]:
        if row["warmup"] or row["kind"] != "solo_copy":
            continue
        (branch,) = row["branches"]
        samples[branch["operation"]][row["bytes"]].append(
            branch["gpu_event_elapsed_us"]
        )
    fits: dict[str, Any] = {}
    for operation, sizes in samples.items():
        if set(sizes) != {4096, 1 << 20, 14 << 20, 28 << 20}:
            raise RuntimeError("training sizes differ from frozen protocol")
        if any(len(values) != 7 for values in sizes.values()):
            raise RuntimeError("training repetition count differs from protocol")
        points = [
            (size, statistics.median(values)) for size, values in sorted(sizes.items())
        ]
        x_mean = statistics.mean(p[0] for p in points)
        y_mean = statistics.mean(p[1] for p in points)
        slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / sum(
            (x - x_mean) ** 2 for x, _ in points
        )
        intercept = y_mean - slope * x_mean
        fits[operation] = {
            "intercept_us": intercept,
            "slope_us_per_byte": slope,
            "training_points": [
                {"bytes": x, "median_us": y, "fitted_us": intercept + slope * x}
                for x, y in points
            ],
        }
    artifact = {
        "schema_version": 1,
        "frozen_wall_ns": time.time_ns(),
        "training_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "method": "ordinary least squares over four unweighted size medians",
        "target": "CUDA event copy bracket us, including within-stream/engine wait",
        "fits": fits,
        "heldout_bytes": [7 << 20, 21 << 20],
        "validation_protocol": "retain errors; no accuracy threshold or refit",
        "service_ttft_or_policy_ranking_used": False,
        "warning": "negative intercept is possible; no clipping or extrapolation",
    }
    with args.output.open("x") as output:
        output.write(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"frozen": True, "branches": len(fits)}))


if __name__ == "__main__":
    main()
