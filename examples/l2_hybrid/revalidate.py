# SPDX-License-Identifier: Apache-2.0
"""Rerun affected real-engine gates sequentially after an ownership change."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def main() -> None:
    """Freeze and execute a bounded plan, stopping on the first failed gate.

    Raises:
        FileExistsError: If the evidence directory already exists.
        RuntimeError: If a child fails, times out, or reports failed invariants.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    model = json.loads(args.workload.read_text())["model"]
    definitions = [
        ("light-002", "service", ["--trace", "light"]),
        ("light-003", "service", ["--trace", "light"]),
        ("heavy-001", "service", ["--trace", "heavy"]),
        ("pressure", "probe", ["--policy", "dram_first", "--pressure"]),
        (
            "store-failure",
            "probe",
            ["--policy", "force_cxl", "--failure", "store_partial"],
        ),
        (
            "load-failure-recompute",
            "probe",
            ["--policy", "force_cxl", "--failure", "load_partial"],
        ),
        (
            "load-failure-fail",
            "probe",
            [
                "--policy",
                "force_cxl",
                "--failure",
                "load_partial",
                "--load-failure-policy",
                "fail",
            ],
        ),
        ("cancel-load", "probe", ["--policy", "force_cxl", "--cancel", "load"]),
        ("cancel-store", "probe", ["--policy", "force_cxl", "--cancel", "store"]),
    ]
    plan: list[dict[str, Any]] = []
    for name, kind, options in definitions:
        script = "run_service.py" if kind == "service" else "probe_hybrid.py"
        command = [
            sys.executable,
            str(Path(__file__).with_name(script)),
            "--build",
            str(args.build.resolve()),
            "--out",
            str(args.out / name),
        ]
        if kind == "service":
            command += ["--workload", str(args.workload.resolve()), "--mode", "hybrid"]
        else:
            command += ["--model", model]
            if name.startswith("cancel-"):
                command += ["--reference", str(args.reference.resolve())]
        plan.append(
            {
                "name": name,
                "kind": kind,
                "command": command + options,
                "content_verifier": "0" if kind == "service" else "1",
                "timeout_s": 600,
            }
        )
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for item in plan:
        print(json.dumps({"starting": item["name"]}), flush=True)
        row = {**item, "started_wall_ns": time.time_ns()}
        environment = dict(os.environ)
        environment["LMCACHE_VERIFY_KV_CONTENT"] = item["content_verifier"]
        with (args.out / (item["name"] + ".log")).open("w") as log:
            child = subprocess.Popen(
                item["command"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                row["exit_code"] = child.wait(timeout=item["timeout_s"])
            except subprocess.TimeoutExpired:
                row["timeout"] = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                row["exit_code"] = child.returncode
        row["completed_wall_ns"] = time.time_ns()
        report_path = args.out / item["name"] / "report.json"
        row["passed"] = (
            row["exit_code"] == 0
            and report_path.exists()
            and json.loads(report_path.read_text())["passed"]
        )
        if row["passed"] and item["name"].startswith("cancel-"):
            row["cancel_audit_exit"] = subprocess.call(
                [
                    sys.executable,
                    str(Path(__file__).with_name("verify_cancellation.py")),
                    str(args.out / item["name"]),
                ]
            )
            row["passed"] = row["cancel_audit_exit"] == 0
        results.append(row)
        (args.out / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps({"finished": item["name"], "passed": row["passed"]}), flush=True
        )
        if not row["passed"]:
            raise RuntimeError(f"revalidation failed: {item['name']}")


if __name__ == "__main__":
    main()
