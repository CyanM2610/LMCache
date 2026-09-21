# SPDX-License-Identifier: Apache-2.0
"""Capture current source, model, hardware and interpreter identity as evidence."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys


def command(args: list[str]) -> dict:
    """Capture command, exit code and output, retaining failures explicitly."""
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {
            "argv": args,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": args, "error": repr(error)}


def digest(path: Path) -> str:
    """Hash the full file with bounded memory; return a hexadecimal SHA-256."""
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(8 << 20):
            hasher.update(data)
    return hasher.hexdigest()


def main() -> None:
    """Write an immutable inventory, including real model/tokenizer contents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh output path; prior inventories are immutable")
    result: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable,
        "platform": platform.platform(),
        "distributions": sorted(
            [
                {
                    "name": dist.metadata["Name"],
                    "version": dist.version,
                    "location": str(dist.locate_file("")),
                }
                for dist in importlib.metadata.distributions()
            ],
            key=lambda row: row["name"].lower(),
        ),
        "repositories": {},
        "commands": [],
        "model_files": [],
        "physical_cxl_observed": False,
        "sysfs_cxl": sorted(str(p) for p in Path("/sys/bus/cxl/devices").glob("*")),
        "sysfs_dax": sorted(str(p) for p in Path("/sys/bus/dax/devices").glob("*")),
    }
    for repo in ("vllm", "LMCache", "CXLMemSim"):
        result["repositories"][repo] = {
            name: command(["git", "-C", str(args.workspace / repo)] + argv)
            for name, argv in (
                ("sha", ["rev-parse", "HEAD"]),
                ("tag", ["describe", "--tags", "--always"]),
                ("changes", ["status", "--short"]),
            )
        }
    for argv in (
        ["nvidia-smi", "-q"],
        ["nvidia-smi", "topo", "-m"],
        ["lspci", "-nn"],
        ["numactl", "--hardware"],
        ["/usr/local/cuda/bin/nvcc", "--version"],
    ):
        result["commands"].append(command(argv))
    for path in sorted(args.model.iterdir()):
        if path.is_file():
            result["model_files"].append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
