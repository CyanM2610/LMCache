# SPDX-License-Identifier: Apache-2.0
"""Audit native block protection during a completed cancellation diagnostic."""

# Standard
from pathlib import Path
import argparse
import json


def main() -> None:
    """Require no block reuse before the cancelled transfer's terminal reply."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    directory = args.run / "mp"
    report = json.loads((directory / "report.json").read_text())
    if not report["passed"]:
        raise RuntimeError("cancellation probe did not pass")
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    (scheduler_path,) = directory.glob("scheduler-*.jsonl")
    scheduler = [json.loads(line) for line in scheduler_path.read_text().splitlines()]
    footer, scheduler_footer = events[-1], scheduler[-1]
    if (
        footer.get("record") != "footer"
        or footer["bus_dropped_events"]
        or footer["subscriber_errors"]
    ):
        raise RuntimeError("LMCache trace is incomplete")
    if (
        scheduler_footer["event"] != "footer"
        or scheduler_footer["dropped"]
        or scheduler_footer["error"]
    ):
        raise RuntimeError("scheduler trace is incomplete")
    cancelled = report["submission_event"]["session_id"]
    if report["direction"] == "lookup":
        owned = report["submission_event"]["metadata"]["logical_references"]
        released = report["cancelled_lookup_release"]
        if not owned or released["logical_references_released"] != owned:
            raise RuntimeError("cancelled lookup lost owned references")
        if any(
            row["event"] == "schedule"
            and any(request["request_id"] == cancelled for request in row["requests"])
            for row in scheduler
        ):
            raise RuntimeError("cancelled lookup reached model computation")
        final = [r for r in scheduler if r["event"] == "scheduler_state"][-1]
        if final["protected_blocks"] or final["requests"]:
            raise RuntimeError("native request state did not drain")
        result = {
            "schema_version": 1,
            "passed": True,
            "direction": "lookup",
            "cancelled_engine_id": cancelled,
            "logical_references_released": owned,
            "cancel_to_release_ns": released["observed_ns"]
            - report["cancel_called_ns"],
            "submitted_copies": 0,
            "model_computation_batches": 0,
            "retry_max_logprob_delta": report["retry_max_logprob_delta"],
        }
        with (directory / "cancellation-audit.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"passed": True, "direction": "lookup", "released": owned}))
        return
    chunks = [
        row["metadata"]
        for row in events
        if row.get("event") == "hybrid.access"
        and row["session_id"] == cancelled
        and row["metadata"]["phase"] == "complete"
    ]
    blocks = {
        block for chunk in chunks for group in chunk["block_ids"] for block in group
    }
    begin = report["cancel_called_ns"]
    end = report["cancelled_transfer_completion"]["reply_ready_ns"]
    held = []
    for row in scheduler:
        if row["event"] == "scheduler_state" and begin <= row["monotonic_ns"] < end:
            matching = [
                r
                for r in row["requests"]
                if r["request_id"] == cancelled and r["status"].startswith("FINISHED")
            ]
            if matching:
                if not blocks <= set(row["protected_blocks"]):
                    raise RuntimeError(
                        "cancelled transfer blocks released before terminal reply"
                    )
                held.append(row["monotonic_ns"])
        if row["event"] == "schedule" and begin <= row["begin_ns"] < end:
            for request in row["requests"]:
                if request["request_id"] == cancelled:
                    continue
                other = {block for group in request["block_ids"] for block in group}
                if blocks & other:
                    raise RuntimeError(
                        "competitor reused a cancelled transfer's blocks"
                    )
    if not held:
        raise RuntimeError("no observed native finished-but-held cancellation state")
    releases = [
        row
        for row in scheduler
        if row["event"] == "scheduler_output_processed"
        and cancelled in row["finished_receiving"] + row["finished_sending"]
    ]
    if len(releases) != 1 or releases[0]["begin_ns"] < end:
        raise RuntimeError("missing, duplicate, or premature scheduler release")
    checks = [r["metadata"] for r in events if r.get("event") == "mp.kv.content_check"]
    if not checks or not all(r["equal"] for r in checks):
        raise RuntimeError("missing or failing independent KV byte checks")
    final = [r for r in scheduler if r["event"] == "scheduler_state"][-1]
    if final["protected_blocks"] or final["requests"]:
        raise RuntimeError("native holds did not drain")
    result = {
        "schema_version": 1,
        "passed": True,
        "direction": report["direction"],
        "cancelled_engine_id": cancelled,
        "protected_physical_blocks": sorted(blocks),
        "observed_finished_held_ns": held,
        "cancel_to_reply_ready_ns": end - begin,
        "scheduler_release_bounds_ns": [releases[0]["begin_ns"], releases[0]["end_ns"]],
        "independent_byte_checks": len(checks),
        "retry_max_logprob_delta": report["retry_max_logprob_delta"],
        "scope": "submitted GPU work; model pending; active DMA at cancel unverified",
    }
    with (directory / "cancellation-audit.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "direction": report["direction"], "checks": len(checks)}
        )
    )


if __name__ == "__main__":
    main()
