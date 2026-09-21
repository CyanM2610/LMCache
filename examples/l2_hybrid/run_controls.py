# SPDX-License-Identifier: Apache-2.0
"""Run the frozen rotated controls, one fresh process at a time."""

# Standard
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def main() -> None:
    """Persist the run order before execution and retain each terminal result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    modes = ["hbm", "dram", "hybrid", "dram_equal"]
    plan = [
        {"round": repeat, "mode": mode}
        for repeat in range(3)
        for mode in modes[repeat:] + modes[:repeat]
    ]
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    completed = []
    for item in plan:
        name = f"r{item['round']}-{item['mode']}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_service.py")),
            "--workload",
            str(args.workload.resolve()),
            "--build",
            str(args.build.resolve()),
            "--out",
            str(args.out / name),
            "--mode",
            str(item["mode"]),
            "--trace",
            "off",
        ]
        row = {**item, "command": command, "started_wall_ns": time.time_ns()}
        print(json.dumps({"event": "starting", "name": name}), flush=True)
        with (args.out / (name + ".log")).open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                row["exit_code"] = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                row["exit_code"] = child.returncode
                row["timeout"] = True
        row["completed_wall_ns"] = time.time_ns()
        completed.append(row)
        (args.out / "runs.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(
            json.dumps(
                {"event": "finished", "name": name, "exit_code": row["exit_code"]}
            ),
            flush=True,
        )
        if row["exit_code"] != 0:
            raise RuntimeError(f"control {name} failed; inspect its log and report")


if __name__ == "__main__":
    main()
