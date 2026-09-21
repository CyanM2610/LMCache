# SPDX-License-Identifier: Apache-2.0
"""Fresh-process Qwen2.5-7B pure-vLLM gate, before attaching external retention."""

# Standard
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import time

# Third Party
import torch
import vllm


def main() -> None:
    """Generate actual output and save input IDs, runtime identity and failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "seed": 74123,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "block_size": 16,
        "num_gpu_blocks_override": 1024,
        "gpu_memory_utilization": 0.60,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "disable_log_stats": False,
    }
    report = {
        "schema_version": 1,
        "stage": "pure_vllm_gpu_smoke",
        "passed": False,
        "python": sys.executable,
        "vllm_source": vllm.__file__,
        "vllm_distribution_version": importlib.metadata.version("vllm"),
        "torch_source": torch.__file__,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "config": config,
        "sampling": {"temperature": 0.0, "max_tokens": 16, "logprobs": 5},
    }
    started = time.monotonic_ns()
    try:
        engine = vllm.LLM(**config)
        report["engine_ready_ns"] = time.monotonic_ns()
        text = (
            "You are reading a small deterministic systems test. "
            "A cache keeps completed immutable data for later reuse. "
        ) * 16 + "Explain the purpose of a cache in one short sentence."
        params = vllm.SamplingParams(**report["sampling"])
        request_start = time.monotonic_ns()
        outputs = engine.generate([text], params, use_tqdm=False)
        request_end = time.monotonic_ns()
        result = outputs[0]
        candidate = result.outputs[0]
        report["request"] = {
            "text": text,
            "input_token_ids": result.prompt_token_ids,
            "request_id": result.request_id,
            "start_ns": request_start,
            "complete_ns": request_end,
            "output_text": candidate.text,
            "output_token_ids": list(candidate.token_ids),
            "finish_reason": candidate.finish_reason,
        }
        if not candidate.token_ids or not result.finished:
            raise RuntimeError("vLLM did not finish a nonempty generation")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["started_ns"] = started
        report["ended_ns"] = time.monotonic_ns()
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
