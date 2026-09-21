# SPDX-License-Identifier: Apache-2.0
"""Cancel real requests while a submitted hybrid transfer awaits completion.

The launcher slows only the CXLMemSim model for this diagnostic. It does not
claim that CUDA DMA is still running when cancellation is observed.
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

# Third Party
from probe_dram import drain_status, free_port, get_json
from run_service import stop_owned_process
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


async def exercise(
    args: argparse.Namespace, config: dict[str, Any], report: dict[str, Any]
) -> None:
    """Abort after actual server submission, then submit competing real work."""
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    workload = json.loads(Path(__file__).with_name("workload-001.json").read_text())
    prompt = workload["prompts"]["A0"]
    competitor = workload["prompts"]["E0"]
    report["input"] = prompt

    async def reset_hbm() -> None:
        # Frontend final output can precede native STORE/hold retirement.
        deadline = time.monotonic() + 30
        while not await engine.reset_prefix_cache(reset_connector=False):
            if time.monotonic() >= deadline:
                raise RuntimeError("diagnostic HBM reset did not become safe")
            await asyncio.sleep(0.02)

    async def generate(request_id: str, token_ids: list[int]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "request_id": request_id,
            "submitted_ns": time.monotonic_ns(),
        }
        report["requests"].append(row)
        async for result in engine.generate(
            {"prompt_token_ids": token_ids},
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=5),
            request_id,
        ):
            item = result.outputs[0]
            row.update(
                {
                    "output_token_ids": list(item.token_ids),
                    "finish_reason": item.finish_reason,
                    "finished": result.finished,
                    "cached_tokens": result.num_cached_tokens,
                    "selected_token_logprobs": [
                        float(probs[token].logprob)
                        for token, probs in zip(
                            item.token_ids, item.logprobs or [], strict=True
                        )
                    ],
                }
            )
        row["completed_ns"] = time.monotonic_ns()
        return row

    try:
        if args.direction in ("lookup", "load"):
            reference = await generate("seed", prompt["token_ids"])
            await reset_hbm()
        else:
            reference = None
        victim = asyncio.create_task(generate("cancelled", prompt["token_ids"]))
        event_path = args.out / "events.jsonl"
        deadline = time.monotonic() + 30
        submission = None
        while submission is None:
            if victim.done():
                await victim
                raise RuntimeError(
                    "victim completed before pending transfer was observed"
                )
            raw = event_path.read_text()
            lines = raw.splitlines()
            if raw and not raw.endswith("\n"):
                lines = lines[:-1]
            rows = [json.loads(line) for line in lines]
            for row in rows:
                if (
                    row.get("event")
                    == (
                        "hybrid.lookup_lease"
                        if args.direction == "lookup"
                        else "hybrid.completion"
                    )
                    and row["session_id"].startswith("cancelled-")
                    and row["metadata"]["phase"]
                    == ("reserved" if args.direction == "lookup" else "submitted")
                ):
                    submission = row
                    break
            if time.monotonic() > deadline:
                raise TimeoutError("no submitted transfer for cancellation")
            if submission is None:
                await asyncio.sleep(0.002)
        report["submission_event"] = submission
        report["cancel_called_ns"] = time.monotonic_ns()
        await engine.abort("cancelled")
        report["cancel_returned_ns"] = time.monotonic_ns()
        cancelled, competing = await asyncio.gather(
            victim, generate("competing", competitor["token_ids"])
        )
        if cancelled["finish_reason"] != "abort" or not competing["finished"]:
            raise RuntimeError("cancellation/competing request did not complete")
        await reset_hbm()
        retry = await generate("retry", prompt["token_ids"])
        if reference is None:
            control = json.loads(args.reference.read_text())
            reference = next(r for r in control["requests"] if r["prompt"] == "A0")
            if reference["input_token_sha256"] != prompt["token_sha256"]:
                raise RuntimeError("control input does not match cancellation prompt")
        report["retry_output_exact"] = (
            retry["output_token_ids"] == reference["output_token_ids"]
        )
        report["retry_max_logprob_delta"] = max(
            abs(a - b)
            for a, b in zip(
                retry["selected_token_logprobs"],
                reference["selected_token_logprobs"],
                strict=True,
            )
        )
        if not report["retry_output_exact"] or report["retry_max_logprob_delta"] > 0.02:
            raise RuntimeError("post-cancellation retry output changed")
    finally:
        engine.shutdown(timeout=30)


def main() -> None:
    """Own one MP server and engine, retaining cancellation and drain evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hybrid-config", type=Path, required=True)
    parser.add_argument(
        "--direction", choices=["lookup", "load", "store"], required=True
    )
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["L2_HYBRID_OBSERVATION_DIR"] = str(args.out)
    hybrid = json.loads(args.hybrid_config.read_text())
    port, http_port = free_port(), free_port()
    while port == http_port:
        http_port = free_port()
    url = f"http://127.0.0.1:{http_port}"
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
        str(hybrid["dram_bytes"] / (1 << 30)),
        "--no-l1-use-lazy",
        "--shm-name",
        "",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--chunk-size",
        "256",
        "--eviction-policy",
        "noop",
        "--disable-metrics",
        "--l1-hybrid-config",
        str(args.hybrid_config),
        "--event-jsonl",
        str(args.out / "events.jsonl"),
    ]
    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "seed": 74123,
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
        "scheduler_cls": (
            "lmcache.integration.vllm.l2_hybrid_observer.ObservedAsyncScheduler"
        ),
        "kv_transfer_config": {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": port,
                "lmcache.mp.require_terminal_completion": True,
            },
        },
    }
    report: dict[str, Any] = {
        "passed": False,
        "direction": args.direction,
        "config": config,
        "requests": [],
        "diagnostic_hooks": [
            "force_cxl",
            "reset_hbm",
            "slow_modeled_completion",
            "cancel",
        ],
        "claim": (
            "cancellation after GPU submission with combined completion pending; "
            "does not prove DMA remains active at cancel time"
        ),
    }
    if args.direction == "lookup":
        report["diagnostic_hooks"] = [
            "force_cxl",
            "reset_hbm",
            "lookup_ack_delay",
            "cancel",
        ]
        report["claim"] = (
            "cancel an owned hit lookup before allocation or LOAD submission"
        )
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("MP server startup failed")
                try:
                    if get_json(url + "/healthcheck")["status"] == "healthy":
                        break
                except OSError:
                    pass
                time.sleep(0.02)
            asyncio.run(exercise(args, config, report))
            report["final_status"] = drain_status(url, server)
            if (
                report["final_status"]["hybrid_completion"]["pending"]
                or report["final_status"]["storage_manager"]["l1_manager"][
                    "transfer_reservation_count"
                ]
            ):
                raise RuntimeError("cancellation did not drain physical reservations")
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            stop_owned_process(server)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        events = [
            json.loads(line)
            for line in (args.out / "events.jsonl").read_text().splitlines()
        ]
        if args.direction == "lookup":
            released = [
                r["metadata"]
                for r in events
                if r.get("event") == "hybrid.lookup_lease"
                and r["session_id"].startswith("cancelled-")
                and r["metadata"]["phase"] == "released"
            ]
            if (
                len(released) != 1
                or released[0]["observed_ns"] <= report["cancel_called_ns"]
            ):
                raise RuntimeError("lookup was not released after cancellation")
            if any(
                r.get("event") == "hybrid.access"
                and r["session_id"].startswith("cancelled-")
                for r in events
            ):
                raise RuntimeError("lookup cancellation unexpectedly submitted a copy")
            report["cancelled_lookup_release"] = released[0]
            report["passed"] = True
            return
        complete = [
            r["metadata"]
            for r in events
            if r.get("event") == "hybrid.completion"
            and r["session_id"].startswith("cancelled-")
            and r["metadata"]["phase"] == "reply_ready"
        ]
        if (
            len(complete) != 1
            or complete[0]["reply_ready_ns"] <= report["cancel_called_ns"]
        ):
            raise RuntimeError("cancellation did not occur before transfer completion")
        report["cancelled_transfer_completion"] = complete[0]
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
