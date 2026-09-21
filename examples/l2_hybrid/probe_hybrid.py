# SPDX-License-Identifier: Apache-2.0
"""Own a private CXLMemSim server for the same native MP correctness gate."""

# Standard
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import uuid

# First Party
from lmcache.v1.distributed.cxl_sim_client import read_payload_descriptor, shm_path


def main() -> None:
    """Run bounded real inference with peer quotas and retain all process logs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--failure", choices=["store_partial", "load_partial"])
    parser.add_argument("--cancel", choices=["lookup", "load", "store"])
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--load-failure-policy", choices=["recompute", "fail"], default="recompute"
    )
    parser.add_argument(
        "--policy",
        choices=["dram_first", "force_cxl", "force_dram"],
        default="dram_first",
    )
    args = parser.parse_args()
    if args.cancel and (
        args.policy != "force_cxl"
        or args.failure
        or args.pressure
        or not args.reference
    ):
        parser.error(
            "cancellation requires force_cxl and --reference without other hooks"
        )
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "/l2hybrid_" + uuid.uuid4().hex
    bulk, modeled, data = prefix + "_bulk", prefix + "_modeled", prefix + "_data"
    command = [
        str(args.build.resolve() / "cxlmemsim_server"),
        "--comm-mode=bulk-shm",
        "--capacity=64",
        "--bulk-shm-name=" + bulk,
        "--data-shm-name=" + data,
        "--modeled-access-shm-name=" + modeled,
        "--enable-gpu-direct-modeled-access=true",
        "--bulk-read-bandwidth=" + ("0.05" if args.cancel == "load" else "25"),
        "--bulk-write-bandwidth=" + ("0.05" if args.cancel == "store" else "25"),
        "--default_latency=100",
    ]
    # Actual Qwen2.5-7B BF16: one 256-token chunk is exactly 14 MiB.
    config = {
        "dram_bytes": (28 if args.policy == "force_dram" else 14) << 20,
        "cxl_bytes": 28 << 20,
        "bulk_control": bulk,
        "modeled_control": modeled,
        "client_library": str(args.build.resolve() / "libcxlmemsim_modeled_client.so"),
        "policy": args.policy,
    }
    config_path = args.out / "hybrid.json"
    if args.failure:
        config["diagnostic_failure"] = args.failure
    if args.cancel == "lookup":
        config["diagnostic_lookup_delay_s"] = 0.5
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "passed": False,
        "physical_cxl": False,
        "cxl_server_command": command,
        "config": config,
    }
    with (args.out / "cxl-server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                if server.poll() is not None:
                    raise RuntimeError("CXLMemSim server exited during startup")
                try:
                    desc = read_payload_descriptor(bulk)
                    if desc.name == data and shm_path(modeled).exists():
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("CXLMemSim startup timeout")
                time.sleep(0.02)
            with (args.out / "engine.log").open("w") as output:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(
                            Path(__file__).with_name(
                                "probe_cancel.py" if args.cancel else "probe_dram.py"
                            )
                        ),
                        "--model",
                        args.model,
                        "--out",
                        str(args.out / "mp"),
                        "--hybrid-config",
                        str(config_path),
                    ]
                    + (
                        [
                            "--direction",
                            args.cancel,
                            "--reference",
                            str(args.reference.resolve()),
                        ]
                        if args.cancel
                        else ["--load-failure-policy", args.load_failure_policy]
                    )
                    + (["--pressure"] if args.pressure else []),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            report["engine_exit_code"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(
                    "native MP gate failed; see engine.log and mp/server.log"
                )
            report["passed"] = json.loads((args.out / "mp/report.json").read_text())[
                "passed"
            ]
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            report["cxl_server_exit_code"] = server.returncode
            for name in (bulk, bulk + ".lock", modeled, data):
                shm_path(name).unlink(missing_ok=True)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "policy": args.policy}))


if __name__ == "__main__":
    main()
