# SPDX-License-Identifier: Apache-2.0
"""Freeze independent native-model compute shapes without retention reuse."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json

# Third Party
from transformers import AutoTokenizer


def main() -> None:
    """Write cold/repeated and held-out contexts before any model cost samples."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    previous: list[str] = []
    stage = 0
    for length in (128, 512, 640, 384):
        for batch in (1, 2):
            for repeat in range(3):
                current = []
                for _ in range(batch):
                    request_id = f"c{len(requests):03d}"
                    text = (
                        f"Probe {len(requests):04d}: "
                        + (
                            "Independent records require accurate computation "
                            "and stable ordering. "
                        )
                        * 200
                    )
                    ids = tokenizer.encode(text, add_special_tokens=False)[:length]
                    text = tokenizer.decode(ids)
                    if tokenizer.encode(text, add_special_tokens=False) != ids:
                        raise RuntimeError("compute prompt does not round-trip")
                    prompts[request_id] = {
                        "text": text,
                        "token_ids": ids,
                        "token_sha256": hashlib.sha256(
                            json.dumps(ids, separators=(",", ":")).encode()
                        ).hexdigest(),
                    }
                    requests.append(
                        {
                            "request_id": request_id,
                            "prompt": request_id,
                            "stage": stage,
                            "category": "independent_compute",
                            "after_completion": previous,
                            "think_s": 0.02,
                            "root_arrival_s": 0.0 if not previous else None,
                            "max_output_tokens": 16,
                            "cost_context_tokens": length,
                            "requested_batch": batch,
                            "cost_repeat": repeat,
                            "cost_role": "heldout" if length == 384 else "training",
                            "temperature_state": "first_shape"
                            if repeat == 0
                            else "repeated_shape",
                        }
                    )
                    current.append(request_id)
                previous = current
                stage += 1
    first_blocks = [tuple(p["token_ids"][:16]) for p in prompts.values()]
    if len(first_blocks) != len(set(first_blocks)):
        raise RuntimeError("independent prompts share a complete first HBM block")
    artifact = {
        "schema_version": 1,
        "seed": 74123,
        "model": args.model,
        "prompts": prompts,
        "requests": requests,
        "maximum_concurrency": 2,
        "timeout_s": 180,
        "diagnostic_hooks": [],
        "inference": {"temperature": 0, "ignore_eos": True, "logprobs": 5},
        "capacity_extension": {"dram_bytes": 28 << 20, "cxl_bytes": 84 << 20},
        "equal_total_dram_bytes": 112 << 20,
        "arrival_rule": (
            "native predecessor completion + 20 ms; "
            "simultaneous pair for requested batch 2"
        ),
        "scope": (
            "native HBM-only model; actual batch shapes taken from scheduler, "
            "not assumed"
        ),
        "cost_protocol": (
            "128/512/640 training contexts; 384 held out; "
            "first occurrence separated from two repeats"
        ),
    }
    with args.out.open("x") as output:
        output.write(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "requests": len(requests),
                "stages": stage,
                "unique_first_blocks": len(first_blocks),
            }
        )
    )


if __name__ == "__main__":
    main()
