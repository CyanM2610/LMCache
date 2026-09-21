# SPDX-License-Identifier: Apache-2.0
"""Fresh-process, bounded native service workload with optional observation."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar
import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid

# Third Party
from probe_dram import drain_status, free_port, get_json
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase

# First Party
from lmcache.v1.distributed.cxl_sim_client import read_payload_descriptor, shm_path


def stop_owned_process(process: subprocess.Popen | None) -> None:
    """Terminate only a process group created by this runner and reap its leader."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def interrupt_service(signum: int, frame: Any) -> None:
    """Turn launcher termination into Python cleanup of this runner's servers."""
    raise KeyboardInterrupt(f"service interrupted by signal {signum}")


async def exercise(
    args: argparse.Namespace,
    plan: dict[str, Any],
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Use actual predecessor completions for arrivals and stream real outputs."""
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**config), stat_loggers=[ServiceStats]
    )
    started_profile = False
    try:
        # A tiny separate warmup avoids measuring initial dispatcher/JIT startup.
        async for _ in engine.generate(
            {"prompt_token_ids": [1, 2, 3, 4]},
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
            "warmup",
        ):
            pass
        if args.trace == "heavy":
            await engine.start_profile("service")
            started_profile = True
        origin = time.monotonic_ns()
        report["origin_ns"] = origin
        completion_events = {
            row["request_id"]: asyncio.Event() for row in plan["requests"]
        }
        completed: dict[str, int] = {}

        async def request(spec: dict[str, Any]) -> None:
            request_id = spec["request_id"]
            for previous in spec["after_completion"]:
                await completion_events[previous].wait()
            planned_ns = (
                max(completed[p] for p in spec["after_completion"])
                if spec["after_completion"]
                else origin
            ) + round(spec["think_s"] * 1e9)
            await asyncio.sleep(max(0, (planned_ns - time.monotonic_ns()) / 1e9))
            prompt = plan["prompts"][spec["prompt"]]
            row: dict[str, Any] = {
                **spec,
                "planned_arrival_ns": planned_ns,
                "actual_arrival_ns": time.monotonic_ns(),
                "input_token_sha256": prompt["token_sha256"],
                "output_token_ids": [],
                "selected_token_logprobs": [],
                "outputs": [],
                "finished": False,
            }
            report["requests"].append(row)
            sampling = SamplingParams(
                **plan["inference"],
                max_tokens=spec["max_output_tokens"],
                output_kind=RequestOutputKind.DELTA,
            )
            try:
                async for result in engine.generate(
                    {"prompt_token_ids": prompt["token_ids"]}, sampling, request_id
                ):
                    observed = time.monotonic_ns()
                    item = result.outputs[0]
                    ids = list(item.token_ids)
                    row["outputs"].append(
                        {
                            "observed_ns": observed,
                            "token_ids": ids,
                            "text": item.text,
                        }
                    )
                    if ids:
                        row.setdefault("first_token_ids_ns", observed)
                        row["last_token_ids_ns"] = observed
                    if item.text:
                        row.setdefault("first_nonempty_text_ns", observed)
                    row["output_token_ids"].extend(ids)
                    row["selected_token_logprobs"].extend(
                        float(probs[token].logprob)
                        for token, probs in zip(ids, item.logprobs or [], strict=True)
                    )
                    row["num_cached_tokens"] = result.num_cached_tokens
                    row["finished"] = result.finished
                    row["finish_reason"] = item.finish_reason
                row["completed_ns"] = time.monotonic_ns()
                if (
                    not row["finished"]
                    or len(row["output_token_ids"]) != spec["max_output_tokens"]
                ):
                    raise RuntimeError("incomplete service request: " + request_id)
                row["ttft_s"] = (
                    row["first_nonempty_text_ns"] - row["actual_arrival_ns"]
                ) / 1e9
                row["tpot_s"] = (
                    (row["last_token_ids_ns"] - row["first_token_ids_ns"])
                    / 1e9
                    / (len(row["output_token_ids"]) - 1)
                )
                completed[request_id] = row["completed_ns"]
                completion_events[request_id].set()
            except BaseException as error:
                row["error"] = repr(error)
                row["failed_ns"] = time.monotonic_ns()
                raise

        async with asyncio.timeout(plan["timeout_s"]):
            async with asyncio.TaskGroup() as tasks:
                for spec in plan["requests"]:
                    tasks.create_task(request(spec))
        report["completed_ns"] = time.monotonic_ns()
        references: dict[str, dict[str, Any]] = {}
        for row in sorted(report["requests"], key=lambda r: r["request_id"]):
            previous = references.setdefault(row["prompt"], row)
            row["repeat_output_exact"] = (
                row["output_token_ids"] == previous["output_token_ids"]
            )
            row["repeat_max_logprob_delta"] = max(
                abs(a - b)
                for a, b in zip(
                    row["selected_token_logprobs"],
                    previous["selected_token_logprobs"],
                    strict=True,
                )
            )
            if not row["repeat_output_exact"] or row["repeat_max_logprob_delta"] > 0.02:
                raise RuntimeError("repeat output correctness failed")
        latencies = sorted(r["ttft_s"] for r in report["requests"])
        report["summary"] = {
            "requests": len(latencies),
            "failures": 0,
            "mean_ttft_s": statistics.mean(latencies),
            "p95_ttft_s_nearest_rank": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "mean_tpot_s": statistics.mean(r["tpot_s"] for r in report["requests"]),
            "throughput_requests_s_including_think": len(latencies)
            / ((report["completed_ns"] - origin) / 1e9),
        }
    finally:
        if started_profile:
            await engine.stop_profile()
        engine.shutdown(timeout=30)
        report["native_metric_rows"] = [
            row for logger in ServiceStats.instances for row in logger.rows
        ]


def main() -> None:
    """Start owned servers, execute one frozen workload, and retain failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["hbm", "dram", "hybrid", "dram_equal"], required=True
    )
    parser.add_argument("--trace", choices=["off", "light", "heavy"], default="off")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt_service)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if os.environ.get("LMCACHE_VERIFY_KV_CONTENT", "0") != "0":
        raise ValueError(
            "service measurements require content-check synchronization off"
        )
    plan = json.loads(args.workload.read_text())
    (args.out / "workload.json").write_text(args.workload.read_text())
    config: dict[str, Any] = {
        "model": plan["model"],
        "dtype": "bfloat16",
        "seed": plan["seed"],
        "tensor_parallel_size": 1,
        "max_model_len": 1024,
        "block_size": 16,
        "num_gpu_blocks_override": 96,
        "gpu_memory_utilization": 0.60,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "async_scheduling": True,
        "disable_log_stats": False,
    }
    if args.trace != "off":
        os.environ["L2_HYBRID_OBSERVATION_DIR"] = str(args.out)
        config["scheduler_cls"] = (
            "lmcache.integration.vllm.l2_hybrid_observer.ObservedAsyncScheduler"
        )
    if args.trace == "heavy":
        config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.out / "profile"),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "mode": args.mode,
        "trace": args.trace,
        "physical_cxl": False,
        "diagnostic_hooks": [],
        "config": config,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "requests": [],
        "timing_definition": {
            "ttft": "actual arrival to first stream output with nonempty text",
            "tpot": "(last token-ID observation - first)/(output token count - 1)",
            "itl": "raw coalesced output chunks; no fabricated per-token times",
            "clock": "CLOCK_MONOTONIC nanoseconds; one Linux host",
        },
    }
    cxl = mp = None
    logs = []
    names: list[str] = []
    try:
        hybrid = None
        if args.mode == "hybrid":
            prefix = "/l2service_" + uuid.uuid4().hex
            bulk, model, data = prefix + "_bulk", prefix + "_model", prefix + "_data"
            names = [bulk, bulk + ".lock", model, data]
            cxl_command = [
                str(args.build.resolve() / "cxlmemsim_server"),
                "--comm-mode=bulk-shm",
                "--capacity=128",
                "--bulk-shm-name=" + bulk,
                "--data-shm-name=" + data,
                "--modeled-access-shm-name=" + model,
                "--enable-gpu-direct-modeled-access=true",
                "--bulk-read-bandwidth=25",
                "--bulk-write-bandwidth=25",
                "--default_latency=100",
            ]
            report["cxl_command"] = cxl_command
            log = (args.out / "cxl.log").open("w")
            logs.append(log)
            cxl = subprocess.Popen(
                cxl_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 15
            while True:
                if cxl.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("CXLMemSim did not become ready")
                try:
                    if (
                        read_payload_descriptor(bulk).name == data
                        and shm_path(model).exists()
                    ):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
            hybrid = {
                **plan["capacity_extension"],
                "policy": "dram_first",
                "bulk_control": bulk,
                "modeled_control": model,
                "client_library": str(
                    args.build.resolve() / "libcxlmemsim_modeled_client.so"
                ),
            }
            (args.out / "hybrid.json").write_text(json.dumps(hybrid, indent=2) + "\n")
        if args.mode != "hbm":
            port, http_port = free_port(), free_port()
            while port == http_port:
                http_port = free_port()
            url = f"http://127.0.0.1:{http_port}"
            config["kv_transfer_config"] = {
                "kv_connector": "LMCacheMPConnector",
                "kv_connector_module_path": (
                    "lmcache.integration.vllm.lmcache_mp_connector"
                ),
                "kv_role": "kv_both",
                "kv_load_failure_policy": "recompute",
                "kv_connector_extra_config": {
                    "lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": port,
                    "lmcache.mp.require_terminal_completion": bool(hybrid),
                },
            }
            dram_bytes = (
                plan["equal_total_dram_bytes"]
                if args.mode == "dram_equal"
                else plan["capacity_extension"]["dram_bytes"]
            )
            command = [
                str(Path(sys.executable).with_name("lmcache")),
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--http-host",
                "127.0.0.1",
                "--http-port",
                str(http_port),
                "--l1-size-gb",
                str(dram_bytes / (1 << 30)),
                "--no-l1-use-lazy",
                "--shm-name",
                "",
                "--supported-transfer-mode",
                "lmcache_driven",
                "--chunk-size",
                "256",
                "--eviction-policy",
                "noop" if hybrid else "LRU",
                "--disable-metrics",
            ]
            if hybrid:
                command.extend(["--l1-hybrid-config", str(args.out / "hybrid.json")])
            if args.trace != "off":
                command.extend(["--event-jsonl", str(args.out / "events.jsonl")])
            report["mp_command"] = command
            log = (args.out / "mp.log").open("w")
            logs.append(log)
            mp = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + 60
            while True:
                if mp.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("LMCache did not become ready")
                try:
                    if get_json(url + "/healthcheck")["status"] == "healthy":
                        break
                except OSError:
                    pass
                time.sleep(0.05)
            report["initial_status"] = get_json(url + "/status")
        asyncio.run(exercise(args, plan, config, report))
        if mp:
            report["final_status"] = drain_status(url, mp)
            status = report["final_status"]
            if hybrid and (
                status["hybrid_completion"]["pending"]
                or status["storage_manager"]["l1_manager"]["transfer_reservation_count"]
            ):
                raise RuntimeError("hybrid did not drain")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        stop_owned_process(mp)
        stop_owned_process(cxl)
        for name in names:
            shm_path(name).unlink(missing_ok=True)
        for log in logs:
            log.close()
        (args.out / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    print(json.dumps(report["summary"]))


class ServiceStats(StatLoggerBase):
    """Retain native metric increments equally in all observation modes."""

    instances: ClassVar[list["ServiceStats"]] = []

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        """Collect native counters without network or per-step file I/O."""
        self.rows: list[dict[str, Any]] = []
        self.instances.append(self)

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        """Copy only public scalar/counter fields from native callbacks."""
        row: dict[str, Any] = {"observed_ns": time.monotonic_ns()}
        if scheduler_stats:
            row["scheduler"] = {
                "running": scheduler_stats.num_running_reqs,
                "waiting": scheduler_stats.num_waiting_reqs,
                "deferred": scheduler_stats.num_skipped_waiting_reqs,
                "kv_cache_usage": scheduler_stats.kv_cache_usage,
                "prefix_cache": asdict(scheduler_stats.prefix_cache_stats),
                "connector_prefix_cache": (
                    asdict(scheduler_stats.connector_prefix_cache_stats)
                    if scheduler_stats.connector_prefix_cache_stats
                    else None
                ),
            }
        if iteration_stats:
            row["iteration"] = {
                "prompt_tokens": asdict(iteration_stats.prompt_token_stats),
                "generation_tokens": iteration_stats.num_generation_tokens,
                "preemptions": iteration_stats.num_preempted_reqs,
                "finished_requests": [
                    asdict(r) for r in iteration_stats.finished_requests
                ],
            }
        self.rows.append(row)

    def log_engine_initialized(self) -> None:
        """No external logger is needed for this bounded artifact collector."""


if __name__ == "__main__":
    main()
