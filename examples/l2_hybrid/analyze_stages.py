# SPDX-License-Identifier: Apache-2.0
"""Measure inclusive service boundaries without relabeling waits as DMA."""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import statistics


def main() -> None:
    """Validate one light trace and retain per-request inclusive cost samples.

    Raises:
        RuntimeError: If event loss, missing boundaries, or negative time occurs.
        FileExistsError: If the output already exists.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.run / "events.jsonl").read_text().splitlines()
    ]
    footer = rows[-1]
    if (
        footer.get("record") != "footer"
        or footer["bus_dropped_events"]
        or footer["subscriber_errors"]
    ):
        raise RuntimeError("incomplete event trace")
    lookups: dict[str, dict[str, float]] = defaultdict(dict)
    jobs: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        name = row.get("event", "")
        request = row.get("session_id", "")
        if request.startswith("warmup-"):
            continue
        metadata = row.get("metadata", {})
        if name in ("mp.lookup_prefetch.start", "mp.lookup_prefetch.end"):
            lookups[request][name.rsplit(".", 1)[1]] = row["timestamp_s"]
        elif name == "hybrid.completion":
            jobs[metadata["transfer_key"]][metadata["phase"]] = metadata
        elif name == "hybrid.access" and metadata["phase"] == "complete":
            job = jobs[metadata["transfer_key"]]
            job.setdefault("objects", []).append(metadata)
            job["request_id"] = request
    samples: list[dict[str, Any]] = []
    for request, bounds in lookups.items():
        samples.append(
            {
                "request_id": request,
                "operation": "lookup_hash_directory_and_status_consumption",
                "duration_us": (bounds["end"] - bounds["start"]) * 1e6,
                "clock": "event Unix float64 seconds; same server process",
            }
        )
    transfers = []
    for key, job in jobs.items():
        objects = job["objects"]
        begin = min(obj["demand_ns"] for obj in objects)
        submit = job["submitted"]["cpu_submitted_ns"]
        observed = job["cuda_observed"]["observed_ns"]
        reply = job["reply_ready"]["reply_ready_ns"]
        intervals = {
            "native_submission_including_gather_scatter_enqueue": submit - begin,
            "submitted_to_cuda_observed_including_poll": observed - submit,
            ("cuda_observed_to_reply_including_model_wait_and_finalize"): reply
            - observed,
        }
        for operation, duration in intervals.items():
            samples.append(
                {
                    "request_id": job["request_id"],
                    "transfer_key": key,
                    "operation": operation,
                    "direction": objects[0]["direction"],
                    "duration_us": duration / 1000,
                    "clock": "host monotonic ns",
                }
            )
        transfers.append({"transfer_key": key, **job})
    groups: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample["duration_us"] < 0:
            raise RuntimeError("negative observed boundary interval")
        groups[sample["operation"]].append(sample["duration_us"])
    result = {
        "schema_version": 1,
        "passed": True,
        "samples": samples,
        "transfers": transfers,
        "summary_us": {
            key: {
                "count": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "max": max(values),
            }
            for key, values in groups.items()
        },
        "separately_unmeasured": {
            key: None
            for key in (
                "hash_only",
                "policy_only",
                "ipc_only",
                "gather_scatter_kernel_only",
                "service_payload_dma_exact_endpoints",
                "poll_only",
                "publish_only",
                "output_serialization_only",
            )
        },
        "scope": (
            "inclusive host stages; concurrent object/model times must not be "
            "summed; independent copy profiler supplies actual DMA activities"
        ),
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "samples": len(samples)}))


if __name__ == "__main__":
    main()
