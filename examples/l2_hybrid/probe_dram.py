# SPDX-License-Identifier: Apache-2.0
"""Native vLLM/LMCache MP miss, HBM reuse, and forced DRAM restore gate.

Resetting HBM is a diagnostic hook, not normal-policy performance evidence.
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Third Party
import vllm


def free_port() -> int:
    """Choose an unused local port; a bind race fails startup rather than retries."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url: str) -> dict[str, Any]:
    """Read a local diagnostic response with a bounded network wait."""
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def drain_status(url: str, server: subprocess.Popen) -> dict[str, Any]:
    """Wait at most 30 seconds for actual L1 locks and request sessions to drain."""
    deadline = time.monotonic() + 30
    while True:
        if server.poll() is not None:
            raise RuntimeError("LMCache server exited before drain")
        state = get_json(url + "/status")
        l1 = state["storage_manager"]["l1_manager"]
        if not (
            l1["write_locked_count"]
            or l1["read_locked_count"]
            or l1["temporary_count"]
            or state["active_sessions"]
        ):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "LMCache reservations/sessions did not drain: " + json.dumps(state)
            )
        time.sleep(0.02)


def generation(
    engine: vllm.LLM, text: str, stage: str, *, expect_load_error: bool = False
) -> dict[str, Any]:
    """Run one real greedy request, retaining chosen-token log probabilities."""
    print(json.dumps({"stage": stage, "event": "submit"}), flush=True)
    start = time.monotonic_ns()
    result = engine.generate(
        [text],
        vllm.SamplingParams(temperature=0, max_tokens=16, logprobs=5),
        use_tqdm=False,
    )[0]
    end = time.monotonic_ns()
    candidate = result.outputs[0]
    logprobs = [
        float(probs[token].logprob)
        for token, probs in zip(
            candidate.token_ids, candidate.logprobs or [], strict=True
        )
    ]
    expected_error = expect_load_error and candidate.finish_reason == "error"
    if expect_load_error and not expected_error:
        raise RuntimeError("injected LOAD did not fail under the native fail policy")
    if not result.finished or (not candidate.token_ids and not expected_error):
        raise RuntimeError("request did not produce completed real output")
    return {
        "stage": stage,
        "finish_reason": candidate.finish_reason,
        "expected_load_error": expected_error,
        "request_id": result.request_id,
        "input_text": text,
        "input_token_ids": result.prompt_token_ids,
        "output_text": candidate.text,
        "output_token_ids": list(candidate.token_ids),
        "selected_token_logprobs": logprobs,
        "num_cached_tokens": result.num_cached_tokens,
        "submitted_ns": start,
        "completed_ns": end,
    }


def main() -> None:
    """Start a private MP server and one engine; retain results including failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument(
        "--load-failure-policy", choices=["recompute", "fail"], default="recompute"
    )
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    hybrid = json.loads(args.hybrid_config.read_text()) if args.hybrid_config else None
    port, http_port = free_port(), free_port()
    while port == http_port:
        http_port = free_port()
    url = f"http://127.0.0.1:{http_port}"
    connector = {
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": args.load_failure_policy,
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": port,
            "lmcache.mp.require_terminal_completion": bool(hybrid),
        },
    }
    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "seed": 74123,
        "tensor_parallel_size": 1,
        "max_model_len": 1024 if args.pressure else 4096,
        "block_size": 16,
        "num_gpu_blocks_override": 96 if args.pressure else 1024,
        "gpu_memory_utilization": 0.60,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "disable_log_stats": False,
        "kv_transfer_config": connector,
    }
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
        str(hybrid["dram_bytes"] / (1 << 30)) if hybrid else "0.5",
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
        "--event-jsonl",
        str(args.out / "events.jsonl"),
    ]
    if args.hybrid_config:
        command.extend(["--l1-hybrid-config", str(args.hybrid_config.resolve())])
    report: dict[str, Any] = {
        "schema_version": 1,
        "hybrid_config": hybrid,
        "pressure": args.pressure,
        "passed": False,
        "diagnostic_hooks": ["reset_hbm"],
        "config": config,
        "server_command": command,
        "requests": [],
        "tolerance": {"selected_logprob_atol": 0.02, "output_token_ids": "exact"},
        "claim_scope": "model output and external restore; bytewise KV audit pending",
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "content_check": os.environ.get("LMCACHE_VERIFY_KV_CONTENT", "0"),
    }
    engine = None
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("LMCache server startup failed; see server.log")
                try:
                    if get_json(url + "/healthcheck")["status"] == "healthy":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("LMCache server readiness timeout")
                time.sleep(0.05)
            report["initial_status"] = get_json(url + "/status")
            engine = vllm.LLM(**config)
            text = (
                "In this deterministic test, a cache stores completed immutable "
                "prefixes. Readers reuse these prefixes when the exact tokens match. "
            ) * 32 + "Explain why cached prefixes need immutable data in one sentence."
            for stage in (
                "miss",
                "hbm_hit",
                "hybrid_restore" if hybrid else "dram_restore",
            ):
                if stage.endswith("_restore"):
                    report["hbm_reset_succeeded"] = engine.reset_prefix_cache(
                        reset_running_requests=False, reset_connector=False
                    )
                    if not report["hbm_reset_succeeded"]:
                        raise RuntimeError(
                            "HBM reset failed; cannot claim external path"
                        )
                record = generation(
                    engine,
                    text,
                    stage,
                    expect_load_error=bool(
                        hybrid
                        and hybrid.get("diagnostic_failure") == "load_partial"
                        and args.load_failure_policy == "fail"
                        and stage == "hybrid_restore"
                    ),
                )
                report["requests"].append(record)
                record["drained_server_status"] = drain_status(url, server)
                (args.out / "progress.json").write_text(json.dumps(report, indent=2))
            if args.pressure:
                competitor = (
                    "Zebra records form an independent prefix. Every entry describes "
                    "a different object and must not inherit unrelated cached data. "
                ) * 32 + "Summarize why unrelated cache keys must remain isolated."
                for stage, prompt in (
                    ("competitor", competitor),
                    ("revisit_after_eviction", text),
                ):
                    if not engine.reset_prefix_cache(reset_connector=False):
                        raise RuntimeError("pressure probe HBM reset failed")
                    row = generation(engine, prompt, stage)
                    report["requests"].append(row)
                    row["drained_server_status"] = drain_status(url, server)
            if hybrid and hybrid.get("diagnostic_failure"):
                if not engine.reset_prefix_cache(reset_connector=False):
                    raise RuntimeError("failure retry HBM reset failed")
                row = generation(engine, text, "retry_after_failure")
                report["requests"].append(row)
                row["drained_server_status"] = drain_status(url, server)
            references = {}
            for row in report["requests"]:
                if row["expected_load_error"]:
                    continue
                if row["input_text"] not in references:
                    references[row["input_text"]] = row
                    continue
                reference = references[row["input_text"]]
                row["output_exact"] = (
                    row["output_token_ids"] == reference["output_token_ids"]
                )
                if not row["output_exact"]:
                    raise RuntimeError(f"output differs after {row['stage']}")
                row["max_logprob_delta"] = max(
                    abs(a - b)
                    for a, b in zip(
                        row["selected_token_logprobs"],
                        reference["selected_token_logprobs"],
                        strict=True,
                    )
                )
                if row["max_logprob_delta"] > 0.02:
                    raise RuntimeError(f"logprob tolerance exceeded: {row['stage']}")
            engine.llm_engine.engine_core.shutdown()
            engine = None
            report["final_status"] = drain_status(url, server)
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            if engine is not None:
                engine.llm_engine.engine_core.shutdown()
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            report["server_exit_code"] = server.returncode
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        events = [
            json.loads(line)
            for line in (args.out / "events.jsonl").read_text().splitlines()
        ]
        footer = events[-1]
        if (
            footer.get("record") != "footer"
            or footer["bus_dropped_events"]
            or footer["subscriber_errors"]
        ):
            raise RuntimeError("event stream did not close without loss/errors")
        for request in report["requests"]:
            # The pinned vLLM InputProcessor appends an 8-hex nonce to external IDs.
            # Require a unique exact match and persist it; never equate the two IDs.
            pattern = re.escape(request["request_id"]) + r"-[0-9a-f]{8}"
            internal_ids = {
                event["session_id"]
                for event in events
                if event.get("record") == "event"
                and re.fullmatch(pattern, event.get("session_id", ""))
            }
            if len(internal_ids) != 1:
                raise RuntimeError("missing or ambiguous engine request ID mapping")
            request["engine_request_id"] = internal_ids.pop()
        restored_id = report["requests"][-1]["engine_request_id"]
        restored_tokens = sum(
            int(e["metadata"].get("num_tokens", 0))
            for e in events
            if e.get("event") == "mp.retrieve.end"
            and e.get("session_id") == restored_id
        )
        if restored_tokens <= 0:
            raise RuntimeError("no actual MP retrieval for the HBM-cold request")
        report["external_retrieved_tokens"] = restored_tokens
        if not hybrid:
            report["dram_retrieved_tokens"] = restored_tokens
        else:
            accesses = [
                e["metadata"]
                for e in events
                if e.get("event") == "hybrid.access"
                and e["metadata"]["phase"] == "complete"
            ]
            completed_loads = [
                a for a in accesses if a["direction"] == "load" and a["state"] == 3
            ]
            expected_media = (
                {"dram", "cxl_sim"}
                if hybrid["policy"] == "dram_first"
                else {"dram" if hybrid["policy"] == "force_dram" else "cxl_sim"}
            )
            if {a["medium"] for a in completed_loads} != expected_media:
                raise RuntimeError(
                    "completed LOAD media do not match the required path"
                )
            report["hybrid_accesses"] = accesses
            if hybrid.get("diagnostic_failure"):
                if report["final_status"]["hybrid_completion"]["failed"] != 1:
                    raise RuntimeError("expected one fully drained failed transfer")
                if hybrid["diagnostic_failure"] == "store_partial":
                    first_state = report["requests"][0]["drained_server_status"]
                    if (
                        first_state["storage_manager"]["l1_manager"][
                            "total_object_count"
                        ]
                        != 0
                    ):
                        raise RuntimeError(
                            "failed partial STORE published or leaked an object"
                        )
                report["partial_failure_recovered"] = True
            for request in report["requests"]:
                status = request["drained_server_status"]
                if status["storage_manager"]["l1_manager"][
                    "transfer_reservation_count"
                ]:
                    raise RuntimeError("physical transfer pins remain after drain")
                if status["hybrid_completion"]["pending"]:
                    raise RuntimeError("deferred replies remain after drain")
            if args.pressure:
                evictions = [
                    e
                    for e in events
                    if e.get("event") == "l1.keys.evicted" and e["metadata"].get("keys")
                ]
                if not evictions:
                    raise RuntimeError("retention pressure did not produce an eviction")
                if any(
                    meta["backend"] != "cxl_sim"
                    for e in evictions
                    for meta in e["metadata"]["meta"]
                ):
                    raise RuntimeError("capacity policy evicted a DRAM resident")
                if restored_tokens != 256:
                    raise RuntimeError(
                        "expected retained DRAM prefix and recomputed evicted suffix"
                    )
                report["pressure_evictions"] = evictions
        if report["content_check"] == "1":
            checks = [
                event["metadata"]
                for event in events
                if event.get("event") == "mp.kv.content_check"
            ]
            if not checks or not all(check["equal"] for check in checks):
                raise RuntimeError("missing or failed bytewise KV content checks")
            if {check["direction"] for check in checks} != {"load", "store"}:
                raise RuntimeError("both transfer directions require content evidence")
            report["content_checks"] = checks
            report["claim_scope"] = (
                "output/logprob and bytewise external round-trip diagnostics"
            )
            if args.pressure:
                initial = [
                    c
                    for c in checks
                    if c["direction"] == "store"
                    and c["transfer_key"].startswith(
                        report["requests"][0]["engine_request_id"] + "#"
                    )
                ]
                competing = [
                    c
                    for c in checks
                    if c["direction"] == "store"
                    and c["transfer_key"].startswith(
                        report["requests"][-2]["engine_request_id"] + "#"
                    )
                ]
                reused = any(
                    set(a["block_ids"]) & set(b["block_ids"])
                    and a["source_sha256"] != b["source_sha256"]
                    for a in initial
                    for b in competing
                )
                if not reused:
                    raise RuntimeError(
                        "competing content did not overwrite an original HBM block"
                    )
                report["hbm_physical_blocks_reused"] = True
        report["passed"] = True
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"passed": True, "external_retrieved_tokens": restored_tokens})
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
