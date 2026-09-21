# SPDX-License-Identifier: Apache-2.0
"""Freeze a bounded mechanism workload before any measured service runs."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json

# Third Party
from transformers import AutoTokenizer


def main() -> None:
    """Write exact text/tokens and completion-dependent arrivals; never overwrite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    topics = {
        "A": "Amber archives store immutable records and reuse matching prefixes.",
        "B": "Blue journals track independent objects and protect concurrent readers.",
        "C": "Coral ledgers preserve complete entries and reject unfinished copies.",
        "D": "Delta notebooks retain useful data and evict only unreferenced objects.",
        "E": "Emerald indexes identify distinct contents and never confuse versions.",
        "F": "Frost catalogs describe separate entries and account for storage limits.",
    }
    suffixes = [
        " Explain the main principle in one sentence.",
        " State the central rule in a short sentence.",
        " Give one reason this rule matters in practice.",
    ]
    prompts = {}
    for topic, phrase in topics.items():
        # Keep exactly 640 shared tokens; suffixes stay well below 1024 total.
        prefix_ids = tokenizer.encode((phrase + " ") * 100, add_special_tokens=False)[
            :640
        ]
        prefix = tokenizer.decode(prefix_ids)
        for variant, suffix in enumerate(suffixes):
            text = prefix + suffix
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompts[f"{topic}{variant}"] = {
                "text": text,
                "token_ids": ids,
                "token_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
    stages = [
        (["A0"], 0.0, "initial_miss"),
        (["B0"], 0.03, "retention_fill"),
        (["C0"], 0.03, "retention_fill"),
        (["D0"], 0.03, "retention_pressure"),
        (["B0"], 0.03, "revisit"),
        (["A0"], 0.03, "revisit"),
        (["A1"], 0.03, "shared_prefix"),
        (["E0", "F0"], 0.03, "unrelated_burst"),
        (["B1", "B2"], 0.03, "shared_prefix_burst"),
        (["A0"], 0.2, "idle_revisit"),
    ]
    requests: list[dict[str, Any]] = []
    predecessor: list[str] = []
    for stage, (names, think, label) in enumerate(stages):
        current = []
        for name in names:
            request_id = f"r{len(requests):02d}"
            requests.append(
                {
                    "request_id": request_id,
                    "prompt": name,
                    "stage": stage,
                    "category": label,
                    "after_completion": predecessor,
                    "think_s": think,
                    "root_arrival_s": 0.0 if not predecessor else None,
                    "max_output_tokens": 16,
                }
            )
            current.append(request_id)
        predecessor = current
    artifact = {
        "schema_version": 1,
        "seed": 74123,
        "model": args.model,
        "prompts": prompts,
        "requests": requests,
        "timeout_s": 180,
        "maximum_concurrency": 2,
        "arrival_rule": "max(actual predecessor completions) + frozen think_s",
        "hbm_budget_note": "96 blocks x 16 tokens, including one null block",
        "capacity_extension": {"dram_bytes": 28 << 20, "cxl_bytes": 84 << 20},
        "equal_total_dram_bytes": 112 << 20,
        "inference": {"temperature": 0, "ignore_eos": True, "logprobs": 5},
        "diagnostic_hooks": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "requests": len(requests),
                "max_prompt_tokens": max(len(p["token_ids"]) for p in prompts.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
