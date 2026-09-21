# SPDX-License-Identifier: Apache-2.0
"""Independent real host/GPU copy costs and shared-link competition samples.

CUDA event brackets include stream/copy-engine waiting, not just bus occupancy.
CXLMemSim costs remain separate. This is an isolated primitive measurement,
not the native KV gather/scatter kernel or a service TTFT measurement.
"""

# Future
from __future__ import annotations

# Standard
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
import time
import uuid

# Third Party
from probe_transport import CudaDriver
import torch

# First Party
from lmcache.v1.distributed.cxl_sim_client import (
    ModeledClient,
    PayloadMapping,
    read_payload_descriptor,
    shm_path,
)


def address_location(address: int) -> dict[str, Any]:
    """Retain the actual process mapping and resident NUMA page distribution."""
    for line in Path("/proc/self/maps").read_text().splitlines():
        low, high = (int(value, 16) for value in line.split()[0].split("-"))
        if low <= address < high:
            numa = [
                row
                for row in Path("/proc/self/numa_maps").read_text().splitlines()
                if row.startswith(f"{low:x} ")
            ]
            return {"maps": line, "numa_maps": numa}
    raise RuntimeError("buffer is absent from process mappings")


def main() -> None:
    """Freeze cases before execution, keep every raw sample and failure report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--frozen-fit", type=Path)
    args = parser.parse_args()
    if args.holdout and not args.frozen_fit:
        parser.error("holdout requires a fit frozen before this process starts")
    fit = json.loads(args.frozen_fit.read_text()) if args.frozen_fit else None
    started_wall_ns = time.time_ns()
    if fit and fit["frozen_wall_ns"] >= started_wall_ns:
        raise ValueError("fit was not frozen before held-out execution")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    sizes = [7 << 20, 21 << 20] if args.holdout else [4096, 1 << 20, 14 << 20, 28 << 20]
    cases: list[dict[str, Any]] = [
        {"operations": [medium + ":" + direction], "bytes": size, "kind": "solo_copy"}
        for size in sizes
        for medium in ("dram", "cxl_sim")
        for direction in ("load", "store")
    ]
    size = (7 if args.holdout else 14) << 20
    competitors = [
        ["dram:load", "cxl_sim:load"],
        ["dram:store", "cxl_sim:store"],
        ["dram:load", "cxl_sim:store"],
        ["dram:store", "cxl_sim:load"],
        ["dram:load", "compute"],
        ["cxl_sim:load", "compute"],
        ["cxl_sim:store", "compute"],
        ["compute"],
    ]
    cases.extend(
        {"operations": ops, "bytes": size, "kind": "competition"} for ops in competitors
    )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "holdout": args.holdout,
        "warmup_repetitions": 2,
        "measured_repetitions": 7,
        "cases": cases,
        "timeout_s": 180,
        "fit_protocol": (
            "training solo-copy median OLS duration=a+b*bytes; "
            "freeze before separate holdout process"
        ),
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    prefix = "/l2primitive_" + uuid.uuid4().hex
    bulk, modeled, data = prefix + "_bulk", prefix + "_model", prefix + "_data"
    command = [
        str(args.build.resolve() / "cxlmemsim_server"),
        "--comm-mode=bulk-shm",
        "--capacity=128",
        "--bulk-shm-name=" + bulk,
        "--data-shm-name=" + data,
        "--modeled-access-shm-name=" + modeled,
        "--enable-gpu-direct-modeled-access=true",
        "--bulk-read-bandwidth=25",
        "--bulk-write-bandwidth=25",
        "--default_latency=100",
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "physical_cxl": False,
        "started_wall_ns": started_wall_ns,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "frozen_fit_sha256": (
            hashlib.sha256(args.frozen_fit.read_bytes()).hexdigest()
            if args.frozen_fit
            else None
        ),
        "holdout": args.holdout,
        "heavy_profiler": args.heavy,
        "command": command,
        "samples": [],
        "content_checks": [],
        "clock": "host monotonic ns; CUDA event offsets us",
        "scope": "flat copies; no KV gather/scatter; one BF16 projection x8",
    }
    bench = None
    profiler = None
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("CXLMemSim startup failed")
                try:
                    if (
                        read_payload_descriptor(bulk).name == data
                        and shm_path(modeled).exists()
                    ):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
            torch.manual_seed(74123)
            bench = CopyBench(args.build.resolve(), bulk, modeled, args.holdout)
            report.update(
                {
                    "cold_stage_ns": bench.cold,
                    "provenance": bench.provenance,
                    "gemm_shape_mnk": bench.gemm_shape,
                    "gemm_repetitions_per_sample": 8,
                }
            )
            deadline = time.monotonic() + plan["timeout_s"]
            if args.heavy:
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                )
                profiler.start()
            for index, case in enumerate(cases):
                bench.prepare(case["operations"], case["bytes"])
                for repeat in range(
                    -plan["warmup_repetitions"], plan["measured_repetitions"]
                ):
                    if time.monotonic() > deadline:
                        raise TimeoutError("primitive suite deadline")
                    context = (
                        torch.profiler.record_function(
                            f"primitive_case_{index}_rep_{repeat}"
                        )
                        if profiler
                        else nullcontext()
                    )
                    with context:
                        sample = bench.sample(case["operations"], case["bytes"])
                    report["samples"].append(
                        {
                            "case_id": index,
                            "repeat": repeat,
                            "warmup": repeat < 0,
                            **case,
                            **sample,
                        }
                    )
                report["content_checks"].append(
                    {
                        "case_id": index,
                        "checks": bench.verify(case["operations"], case["bytes"]),
                    }
                )
            report["active_after_drain"] = len(bench.client.active)
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            try:
                if profiler:
                    profiler.stop()
                    profiler.export_chrome_trace(str(args.out / "profile.json"))
                if bench:
                    bench.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
                for name in (bulk, bulk + ".lock", modeled, data):
                    shm_path(name).unlink(missing_ok=True)
                (args.out / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
    print(
        json.dumps(
            {"passed": True, "cases": len(cases), "measured_samples": len(cases) * 7}
        )
    )


class CopyBench:
    """Own buffers/streams while all submitted physical/model work drains."""

    def __init__(self, build: Path, bulk: str, modeled: str, holdout: bool) -> None:
        """Allocate and pin actual peer backing; retain separate cold costs."""
        self.cold: dict[str, int] = {}
        start = time.monotonic_ns()
        torch.cuda.init()
        self.cuda = CudaDriver()
        self.cold["torch_and_driver_context_ns"] = time.monotonic_ns() - start
        descriptor = read_payload_descriptor(bulk)
        start = time.monotonic_ns()
        self.mapping = PayloadMapping(descriptor)
        self.cold["cxl_mmap_ns"] = time.monotonic_ns() - start
        start = time.monotonic_ns()
        ct.memset(self.mapping.address, 0xA5, descriptor.capacity)
        self.cold["cxl_first_touch_ns"] = time.monotonic_ns() - start
        start = time.monotonic_ns()
        self.cuda.call(
            "cuMemHostRegister_v2", self.mapping.address, descriptor.capacity, 0
        )
        self.cold["cxl_register_ns"] = time.monotonic_ns() - start
        self.client = ModeledClient(
            str(build / "libcxlmemsim_modeled_client.so"), modeled, descriptor.capacity
        )
        self.maximum = 28 << 20
        start = time.monotonic_ns()
        self.dram = torch.empty(self.maximum, dtype=torch.uint8, pin_memory=True)
        self.cold["dram_allocate_and_pin_ns"] = time.monotonic_ns() - start
        start = time.monotonic_ns()
        self.dram.fill_(0x5A)
        self.cold["dram_first_touch_ns"] = time.monotonic_ns() - start
        self.host = {"dram": self.dram.data_ptr(), "cxl_sim": self.mapping.address}
        start = time.monotonic_ns()
        self.gpu = {
            medium: torch.empty(self.maximum, dtype=torch.uint8, device="cuda")
            for medium in self.host
        }
        self.cold["gpu_allocation_ns"] = time.monotonic_ns() - start
        # One representative Qwen hidden-size projection, not whole-model cost.
        self.gemm_shape = [384 if holdout else 512, 3584, 3584]
        m, n, k = self.gemm_shape
        self.matrix_a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self.matrix_b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        self.matrix_c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        self.compute_repetitions = 8
        wait = self.cuda.lib.cuStreamWaitEvent
        wait.argtypes, wait.restype = [ct.c_void_p, ct.c_void_p, ct.c_uint], ct.c_int
        self.streams: list[ct.c_void_p] = []
        self.events: list[ct.c_void_p] = []
        for _ in range(2):
            stream = ct.c_void_p()
            self.cuda.call("cuStreamCreate", ct.byref(stream), 1)
            self.streams.append(stream)
        self.torch_streams = [torch.cuda.ExternalStream(s.value) for s in self.streams]
        for _ in range(5):
            event = ct.c_void_p()
            self.cuda.call("cuEventCreate", ct.byref(event), 0)
            self.events.append(event)
        torch.cuda.current_stream().synchronize()
        self.provenance = {
            "cuda_device_name": self.cuda.name,
            "payload_descriptor": asdict(descriptor),
            "buffers": {
                medium: {
                    "host_ptr": pointer,
                    "gpu_ptr": self.gpu[medium].data_ptr(),
                    "location": address_location(pointer),
                }
                for medium, pointer in self.host.items()
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }

    def prepare(self, ops: list[str], size: int) -> None:
        """Poison destinations before warmup; content checks are outside timing."""
        for medium, pattern in (("dram", 0x5A), ("cxl_sim", 0xA5)):
            ct.memset(self.host[medium], pattern, self.maximum)
            self.gpu[medium].fill_(pattern)
        for op in ops:
            if op == "compute":
                continue
            medium, direction = op.rsplit(":", 1)
            if direction == "load":
                self.gpu[medium][:size].zero_()
            else:
                ct.memset(self.host[medium], 0, size)
        torch.cuda.current_stream().synchronize()

    def sample(self, ops: list[str], size: int) -> dict[str, Any]:
        """Submit independent streams; measure actual brackets and model RPCs."""
        access: dict[int, tuple[int, int]] = {}
        for index, op in enumerate(ops):
            if op.startswith("cxl_sim:"):
                start = time.monotonic_ns()
                begin = self.client.begin(op.split(":")[1], 0, size)
                access[index] = (begin.access_token, time.monotonic_ns() - start)
        start_ns = time.monotonic_ns()
        origin = self.events[0]
        self.cuda.call("cuEventRecord", origin, self.streams[0])
        branches: list[dict[str, Any]] = []
        for index, op in enumerate(ops):
            stream = self.streams[index]
            first, last = self.events[1 + index * 2 : 3 + index * 2]
            self.cuda.call("cuStreamWaitEvent", stream, origin, 0)
            submit = time.monotonic_ns()
            self.cuda.call("cuEventRecord", first, stream)
            if op == "compute":
                with torch.cuda.stream(self.torch_streams[index]):
                    for _ in range(self.compute_repetitions):
                        torch.mm(self.matrix_a, self.matrix_b, out=self.matrix_c)
                api = "torch.mm BF16 x8"
            else:
                medium, direction = op.rsplit(":", 1)
                host, gpu = self.host[medium], self.gpu[medium].data_ptr()
                api = (
                    "cuMemcpyHtoDAsync_v2"
                    if direction == "load"
                    else "cuMemcpyDtoHAsync_v2"
                )
                if direction == "load":
                    self.cuda.call(api, gpu, host, size, stream)
                else:
                    self.cuda.call(api, host, gpu, size, stream)
            self.cuda.call("cuEventRecord", last, stream)
            branches.append(
                {
                    "operation": op,
                    "api": api,
                    "cpu_submit_ns": submit,
                    "cpu_submitted_ns": time.monotonic_ns(),
                }
            )
        submitted = time.monotonic_ns()
        for index in range(len(ops)):
            self.cuda.call("cuEventSynchronize", self.events[2 + index * 2])
        observed = time.monotonic_ns()
        for index, row in enumerate(branches):
            first, last = self.events[1 + index * 2 : 3 + index * 2]
            for field, event in (
                ("gpu_bracket_start_us", first),
                ("gpu_bracket_end_us", last),
            ):
                elapsed = ct.c_float()
                self.cuda.call("cuEventElapsedTime", ct.byref(elapsed), origin, event)
                row[field] = elapsed.value * 1000
            row["gpu_event_elapsed_us"] = (
                row["gpu_bracket_end_us"] - row["gpu_bracket_start_us"]
            )
            row["cuda_observed_ns"] = observed
            if index in access:
                token, begin_ns = access[index]
                rpc_start = time.monotonic_ns()
                self.client.data_complete(token, True)
                data_ns = time.monotonic_ns() - rpc_start
                polls = 0
                deadline = time.monotonic() + 5
                while True:
                    result = self.client.poll(token)
                    polls += 1
                    if result.state in (3, 4, 5):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("primitive modeled completion timeout")
                    time.sleep(0.0001)
                if result.state != 3:
                    raise RuntimeError("primitive model did not finish successfully")
                row["model"] = {
                    "token": token,
                    "begin_rpc_ns": begin_ns,
                    "data_complete_rpc_ns": data_ns,
                    "poll_calls": polls,
                    "logical_observed_ns": time.monotonic_ns(),
                    "queue_ns": result.queue_ns,
                    "service_ns": result.service_ns,
                    "complete_ns": result.modeled_complete_ns,
                }
            else:
                row["model"] = None
        return {
            "cpu_begin_ns": start_ns,
            "cpu_submitted_ns": submitted,
            "cuda_observed_ns": observed,
            "branches": branches,
            "gpu_envelope_us": max(r["gpu_bracket_end_us"] for r in branches),
        }

    def verify(self, ops: list[str], size: int) -> list[dict[str, Any]]:
        """Verify destination bytes after the case, without timing the check."""
        checks = []
        for op in ops:
            if op == "compute":
                if not torch.isfinite(self.matrix_c).all().item():
                    raise RuntimeError("nonfinite independent GEMM result")
                continue
            medium, direction = op.rsplit(":", 1)
            expected = bytes([0x5A if medium == "dram" else 0xA5]) * size
            actual = (
                self.gpu[medium][:size].cpu().numpy().tobytes()
                if direction == "load"
                else ct.string_at(self.host[medium], size)
            )
            if actual != expected:
                raise RuntimeError("primitive destination differs: " + op)
            checks.append(
                {
                    "operation": op,
                    "bytes": size,
                    "equal": True,
                    "sha256": hashlib.sha256(actual).hexdigest(),
                }
            )
        return checks

    def close(self) -> None:
        """Drain physical operations before cancelling metadata and unmapping."""
        for stream in self.streams:
            self.cuda.call("cuStreamSynchronize", stream)
        for token in list(self.client.active):
            if token not in self.client.reported:
                self.client.data_complete(token, False)
            self.client.cancel(token, "primitive cleanup after actual stream drain")
            self.client.poll(token)
        self.client.close()
        for event in self.events:
            self.cuda.call("cuEventDestroy_v2", event)
        for stream in self.streams:
            self.cuda.call("cuStreamDestroy_v2", stream)
        self.cuda.call("cuMemHostUnregister", self.mapping.address)
        self.mapping.close()
        self.cuda.close()


if __name__ == "__main__":
    main()
