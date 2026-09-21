# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic: CXLMemSim payload <-> actual GPU, with byte checks.

This synchronizing content probe is not a service-performance benchmark.
Run from LMCache with .venv/bin/python examples/l2_hybrid/probe_transport.py.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import ctypes as ct
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
import uuid

# First Party
from lmcache.v1.distributed.cxl_sim_client import (
    ModeledClient,
    PayloadMapping,
    read_payload_descriptor,
    shm_path,
)


def transfer(
    cuda: CudaDriver,
    client: ModeledClient,
    mapping: PayloadMapping,
    device_ptr: int,
    size: int,
    direction: str,
) -> dict[str, Any]:
    """Copy the mapped bytes asynchronously then check both completion branches.

    Args:
        cuda: Actual CUDA driver, already bound to the selected GPU.
        client: Live CXLMemSim metadata client.
        mapping: Registered server payload mapping.
        device_ptr: Allocated GPU buffer address.
        size: Defined bytes to copy, excluding any padding.
        direction: ``load`` or ``store``.
    Returns:
        Observed host boundaries, CUDA event duration and modeled service time.
    Raises:
        RuntimeError: If CUDA/model fails or completion is published early.
        TimeoutError: If no model completion is observed in five seconds.
    """
    stream, start, end = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    cuda.call("cuStreamCreate", ct.byref(stream), 1)
    cuda.call("cuEventCreate", ct.byref(start), 0)
    cuda.call("cuEventCreate", ct.byref(end), 0)
    demand_ns = time.monotonic_ns()
    begin = client.begin(direction, 0, size)
    token = begin.access_token
    try:
        # Even if all model time elapsed, no payload success is yet reported.
        pre_dma = client.poll(token)
        if pre_dma.state not in (0, 1):
            raise RuntimeError("model published data before CUDA submission")
        submit_ns = time.monotonic_ns()
        cuda.call("cuEventRecord", start, stream)
        if direction == "load":
            api = "cuMemcpyHtoDAsync_v2"
            cuda.call(api, device_ptr, mapping.address, size, stream)
        else:
            api = "cuMemcpyDtoHAsync_v2"
            cuda.call(api, mapping.address, device_ptr, size, stream)
        cuda.call("cuEventRecord", end, stream)
        submitted_ns = time.monotonic_ns()
        cuda.call("cuEventSynchronize", end)
        observed_ns = time.monotonic_ns()
        duration = ct.c_float()
        cuda.call("cuEventElapsedTime", ct.byref(duration), start, end)
        client.data_complete(token, True)
        deadline = time.monotonic() + 5
        while True:
            result = client.poll(token)
            if result.state in (3, 4, 5):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("modeled completion timeout")
            time.sleep(0.0001)
        if result.state != 3:
            raise RuntimeError(f"modeled transfer failed: {result.state}")
        return {
            "transfer_id": token,
            "direction": direction,
            "api": api,
            "host_ptr": mapping.address,
            "device_ptr": device_ptr,
            "effective_bytes": size,
            "actual_bytes": size,
            "fragments": 1,
            "demand_ns": demand_ns,
            "cpu_submit_ns": submit_ns,
            "cpu_submitted_ns": submitted_ns,
            "cuda_observed_ns": observed_ns,
            "logical_complete_observed_ns": time.monotonic_ns(),
            "cuda_event_elapsed_ns": round(duration.value * 1e6),
            "dma_absolute_start_ns": None,
            "dma_absolute_end_ns": None,
            "null_reason": "CUDA events supply duration, not host-clock instants",
            "modeled_queue_ns": result.queue_ns,
            "modeled_service_ns": result.service_ns,
            "modeled_complete_ns": result.modeled_complete_ns,
            "pre_dma_state": pre_dma.state,
        }
    except BaseException:
        cuda.call("cuStreamSynchronize", stream)
        if token in client.active:
            if token not in client.reported:
                client.data_complete(token, False)
            client.cancel(token, "probe failed after CUDA drain")
            client.poll(token)
        raise
    finally:
        cuda.call("cuEventDestroy_v2", start)
        cuda.call("cuEventDestroy_v2", end)
        cuda.call("cuStreamDestroy_v2", stream)


def main() -> None:
    """Launch a private server, test defined bytes, save evidence, and stop it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4096, 1048576, 8388608]
    )
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) <= 0 or max(args.sizes) > 8 << 20:
        parser.error("sizes must be within 1..8388608 bytes")
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "/l2hybrid_" + uuid.uuid4().hex
    bulk, modeled, data = prefix + "_bulk", prefix + "_modeled", prefix + "_data"
    command = [
        str(args.build.resolve() / "cxlmemsim_server"),
        "--comm-mode=bulk-shm",
        "--capacity=16",
        "--bulk-shm-name=" + bulk,
        "--modeled-access-shm-name=" + modeled,
        "--data-shm-name=" + data,
        "--enable-gpu-direct-modeled-access=true",
        "--bulk-read-bandwidth=25",
        "--bulk-write-bandwidth=25",
        "--default_latency=100",
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "synchronizing_content_diagnostic",
        "cxl_kind": "CXLMemSim_posix_shm",
        "physical_cxl": False,
        "clock": "CLOCK_MONOTONIC_ns",
        "python": sys.executable,
        "client_source": str(Path(inspect.getfile(ModeledClient))),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(inspect.getfile(ModeledClient)),
                args.build / "cxlmemsim_server",
                args.build / "libcxlmemsim_modeled_client.so",
            )
        },
        "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": 74123,
        "samples": [],
        "passed": False,
    }
    client = mapping = cuda = None
    device, registered = ct.c_uint64(), False
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        "CXLMemSim exited during startup; see server.log"
                    )
                try:
                    descriptor = read_payload_descriptor(bulk)
                    if not shm_path(modeled).exists():
                        raise FileNotFoundError(modeled)
                    break
                except (FileNotFoundError, ValueError) as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "CXLMemSim did not publish compatible headers"
                        ) from error
                    time.sleep(0.02)
            if descriptor.name != data:
                raise RuntimeError("server did not publish the private payload name")
            report["payload"] = asdict(descriptor)
            mapping = PayloadMapping(descriptor)
            client = ModeledClient(
                str(args.build.resolve() / "libcxlmemsim_modeled_client.so"),
                modeled,
                descriptor.capacity,
            )
            report["modeled_generation"] = client.capabilities.server_generation
            cuda = CudaDriver()
            report["gpu"] = cuda.name
            register_start = time.monotonic_ns()
            cuda.call("cuMemHostRegister_v2", mapping.address, descriptor.capacity, 1)
            registered = True
            report["register_ns"] = time.monotonic_ns() - register_start
            cuda.call("cuMemAlloc_v2", ct.byref(device), max(args.sizes))
            rng = random.Random(report["seed"])
            for size in args.sizes:
                expected = rng.randbytes(size)
                mapping.mapping[:size] = expected
                load = transfer(cuda, client, mapping, device.value, size, "load")
                # Poison the only host payload copy, so an omitted D2H fails.
                ct.memset(mapping.address, 0, size)
                store = transfer(cuda, client, mapping, device.value, size, "store")
                actual = mapping.mapping[:size]
                sample = {
                    "size": size,
                    "load": load,
                    "store": store,
                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                    "actual_sha256": hashlib.sha256(actual).hexdigest(),
                    "equal": actual == expected,
                }
                report["samples"].append(sample)
                if actual != expected:
                    raise RuntimeError("round-trip content mismatch")
            # Exercise cancellation before data completion. It must not allow
            # client/mapping teardown before the local CUDA lifetime fence.
            token = client.begin("load", 0, 4096).access_token
            client.cancel(token, "diagnostic cancellation before CUDA submission")
            assert client.poll(token).state == 4
            try:
                client.close()
            except RuntimeError:
                report["cancel_retains_reservation"] = True
            else:
                raise RuntimeError("cancel prematurely released local reservation")
            client.data_complete(token, False)  # no CUDA submitted in this case
            assert client.poll(token).state == 4
            token = client.begin("store", 0, 4096).access_token
            client.data_complete(token, False)
            assert client.poll(token).state == 5
            report["copy_failure_not_published"] = True
            report["active_after_drain"] = len(client.active)
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            try:
                if cuda is not None:
                    if device.value:
                        cuda.call("cuMemFree_v2", device)
                    if registered and mapping is not None:
                        cuda.call("cuMemHostUnregister", mapping.address)
                    cuda.close()
                if client is not None:
                    client.close()
                if mapping is not None:
                    mapping.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
                report["server_exit_code"] = server.returncode
                for name in (bulk, bulk + ".lock", modeled, data):
                    shm_path(name).unlink(missing_ok=True)
                (args.out / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
    print(json.dumps({"passed": report["passed"], "report": str(args.out)}))


class CudaDriver:
    """Minimal real CUDA driver binding for independent content verification."""

    def __init__(self) -> None:
        """Initialize the visible GPU's primary context, without torch allocators."""
        self.lib = ct.CDLL("libcuda.so.1")
        ptr, u64, size = ct.c_void_p, ct.c_uint64, ct.c_size_t
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceGetName": [ct.c_char_p, ct.c_int, ct.c_int],
            "cuDevicePrimaryCtxRetain": [ct.POINTER(ptr), ct.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ct.c_int],
            "cuCtxSetCurrent": [ptr],
            "cuMemHostRegister_v2": [ptr, size, ct.c_uint],
            "cuMemHostUnregister": [ptr],
            "cuMemAlloc_v2": [ct.POINTER(u64), size],
            "cuMemFree_v2": [u64],
            "cuStreamCreate": [ct.POINTER(ptr), ct.c_uint],
            "cuStreamSynchronize": [ptr],
            "cuStreamDestroy_v2": [ptr],
            "cuMemcpyHtoDAsync_v2": [u64, ptr, size, ptr],
            "cuMemcpyDtoHAsync_v2": [ptr, u64, size, ptr],
            "cuEventCreate": [ct.POINTER(ptr), ct.c_uint],
            "cuEventRecord": [ptr, ptr],
            "cuEventSynchronize": [ptr],
            "cuEventElapsedTime": [ct.POINTER(ct.c_float), ptr, ptr],
            "cuEventDestroy_v2": [ptr],
        }
        for name, types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = types, ct.c_int
        self.call("cuInit", 0)
        self.device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(self.device), 0)
        device_name = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", device_name, len(device_name), self.device)
        self.name = device_name.value.decode()
        self.context = ptr()
        self.call("cuDevicePrimaryCtxRetain", ct.byref(self.context), self.device)
        self.call("cuCtxSetCurrent", self.context)

    def call(self, name: str, *args: Any) -> None:
        """Call a typed driver function and raise with its name/error code."""
        result = getattr(self.lib, name)(*args)
        if result:
            raise RuntimeError(f"{name} returned CUDA error {result}")

    def close(self) -> None:
        """Release this process's primary context reference."""
        self.call("cuDevicePrimaryCtxRelease_v2", self.device)


if __name__ == "__main__":
    main()
