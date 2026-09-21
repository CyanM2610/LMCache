# SPDX-License-Identifier: Apache-2.0
"""Audit final evidence against explicit bounded acceptance gates."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json


def require(condition: bool, description: str) -> None:
    """Raise RuntimeError when a required evidence condition is false."""
    if not condition:
        raise RuntimeError(description)


def audit_drain(status: dict[str, Any]) -> None:
    """Check explicit ownership drain and pool bounds, allowing retained KV.

    Args:
        status: Final MP server status, recorded after worker unregister.
    Raises:
        RuntimeError: If any reference, session, job, or quota invariant fails.
    """
    require(status["active_sessions"] == 0, "live sessions")
    require(status["active_prefetch_jobs"] == 0, "live lookup jobs")
    l1 = status["storage_manager"]["l1_manager"]
    for field in (
        "write_locked_count",
        "read_locked_count",
        "temporary_count",
        "transfer_reservation_count",
    ):
        require(l1[field] == 0, f"live {field}")
    for pool in (l1.get("pools") or {}).values():
        require(
            0 <= pool["used_bytes"] <= pool["capacity_bytes"] <= pool["backing_bytes"],
            "pool capacity/backing mismatch",
        )
    if l1.get("pools"):
        require(status.get("hybrid_completion") is not None, "missing completion state")
        require(status["hybrid_completion"]["pending"] == 0, "live completion jobs")


def main() -> None:
    """Require final reports, raw checks and source continuity before packaging.

    Raises:
        RuntimeError: If a declared gate or raw-data invariant is incomplete.
        FileExistsError: If output would overwrite an earlier audit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence
    controls = json.loads((evidence / "controls-002/runs.json").read_text())
    require(len(controls) == 12, "incomplete controls")
    for mode in ("hbm", "dram", "hybrid", "dram_equal"):
        require(sum(row["mode"] == mode for row in controls) == 3, "missing repeats")
    replay = json.loads((evidence / "revalidation-001/runs.json").read_text())
    require(len(replay) == 9, "incomplete affected-path replay")
    for row in controls + replay:
        require(row["exit_code"] == 0 and not row.get("timeout"), "failed child")
    normal = [evidence / "service-lease-light-001"]
    normal += [
        evidence / "controls-002" / f"r{row['round']}-{row['mode']}" for row in controls
    ]
    normal += [
        evidence / "revalidation-001" / name
        for name in ("light-002", "light-003", "heavy-001")
    ]
    checked = []
    for directory in normal:
        report = json.loads((directory / "report.json").read_text())
        require(report["passed"] and not report["diagnostic_hooks"], "service hooks")
        require(len(report["requests"]) == 12, "incomplete service")
        require(
            all(row["finished"] for row in report["requests"]), "unfinished request"
        )
        if "final_status" in report:
            audit_drain(report["final_status"])
        checked.append(str(directory.relative_to(evidence)))
    probes = [evidence / "hybrid-cancel-lookup-001"] + [
        evidence / "revalidation-001" / row["name"]
        for row in replay
        if row["kind"] == "probe"
    ]
    byte_checks = 0
    for directory in probes:
        outer = json.loads((directory / "report.json").read_text())
        report = json.loads((directory / "mp/report.json").read_text())
        require(outer["passed"] and report["passed"], "failed mechanism gate")
        require(outer["engine_exit_code"] == 0, "engine failed to stop")
        require(outer["cxl_server_exit_code"] == 0, "simulator failed to stop")
        audit_drain(report["final_status"])
        events = [
            json.loads(line)
            for line in (directory / "mp/events.jsonl").read_text().splitlines()
        ]
        footer = events[-1]
        require(
            footer.get("record") == "footer"
            and not footer["bus_dropped_events"]
            and not footer["subscriber_errors"]
            and not footer["bus_queue_depth"],
            "incomplete diagnostic trace",
        )
        content = [row for row in events if row.get("event") == "mp.kv.content_check"]
        require(bool(content), "missing independent KV checks")
        for row in content:
            metadata = row["metadata"]
            require(metadata["equal"], "KV bytes differ")
            require(
                metadata["source_sha256"] == metadata["destination_sha256"],
                "KV checksum mismatch",
            )
        byte_checks += len(content)
        if "cancel" in directory.name:
            cancel = json.loads((directory / "mp/cancellation-audit.json").read_text())
            require(cancel["passed"], "cancellation audit incomplete")
        if directory.name == "pressure":
            require(report["hbm_physical_blocks_reused"], "physical overwrite unproven")
            require(bool(report["pressure_evictions"]), "no pressure eviction")
        if "failure" in directory.name:
            require(report["partial_failure_recovered"], "partial failure unrecovered")
        checked.append(str(directory.relative_to(evidence)))
    analysis = json.loads(
        (evidence / "controls-002/analysis-observation.json").read_text()
    )
    require(analysis["passed"] and len(analysis["runs"]) == 16, "output comparison")
    for name in (
        "copy-analysis-001.json",
        "compute-independent-001/coverage.json",
        "revalidation-001/heavy-001/profile-analysis.json",
        "service-lease-light-001/stage-costs.json",
    ):
        require(json.loads((evidence / name).read_text())["passed"], name)
    source = json.loads((evidence / "bootstrap/runtime-stage4-source.json").read_text())
    for name, expected in source["files"].items():
        require(
            hashlib.sha256((args.workspace / name).read_bytes()).hexdigest()
            == expected,
            f"runtime changed after validation began: {name}",
        )
    result = {
        "schema_version": 1,
        "passed": True,
        "physical_cxl": False,
        "checked_service_and_mechanism_runs": checked,
        "normal_service_requests": 192,
        "independent_kv_byte_checks": byte_checks,
        "unchanged_runtime_files": len(source["files"]),
        "scope": "bounded amended acceptance; see ACCEPTANCE_REPORT.md for limits",
    }
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
