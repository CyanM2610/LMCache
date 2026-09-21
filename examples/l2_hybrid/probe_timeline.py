# SPDX-License-Identifier: Apache-2.0
"""Convert the diagnostic's observed/model boundaries to a Perfetto timeline."""

# Standard
from pathlib import Path
import argparse
import json


def main() -> None:
    """Write Chrome trace JSON without inventing absolute GPU timestamps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["schema_version"] != 1:
        raise ValueError("unsupported diagnostic schema")
    transfers = [
        sample[direction]
        for sample in report["samples"]
        for direction in ("load", "store")
    ]
    origin = min(t["demand_ns"] for t in transfers)
    events = []
    for transfer in transfers:
        identity = {
            "transfer_id": transfer["transfer_id"],
            "bytes": transfer["effective_bytes"],
            "direction": transfer["direction"],
            "api": transfer["api"],
            "cuda_event_elapsed_ns": transfer["cuda_event_elapsed_ns"],
        }
        for boundary in (
            "demand_ns",
            "cpu_submit_ns",
            "cpu_submitted_ns",
            "cuda_observed_ns",
            "logical_complete_observed_ns",
        ):
            events.append(
                {
                    "name": boundary,
                    "cat": "host_observed",
                    "ph": "i",
                    "s": "t",
                    "pid": 1,
                    "tid": transfer["transfer_id"],
                    "ts": (transfer[boundary] - origin) / 1000,
                    "args": identity,
                }
            )
        end = transfer["modeled_complete_ns"]
        service = transfer["modeled_service_ns"]
        events.append(
            {
                "name": "CXLMemSim modeled service",
                "cat": "model_not_dma",
                "ph": "X",
                "pid": 2,
                "tid": transfer["transfer_id"],
                "ts": (end - service - origin) / 1000,
                "dur": service / 1000,
                "args": identity | {"modeled_queue_ns": transfer["modeled_queue_ns"]},
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "ms",
                "metadata": {
                    "source": str(args.report),
                    "monotonic_origin_ns": origin,
                    "kind": "synchronizing_content_diagnostic",
                    "gpu_span_omitted_reason": "GPU/host clocks were not correlated",
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
