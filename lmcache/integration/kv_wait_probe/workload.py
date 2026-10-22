# SPDX-License-Identifier: Apache-2.0
"""One frozen input file; complete conversations and returned-token history."""

# Standard
from pathlib import Path
import hashlib
import json
import math


def digest(value: object) -> str:
    """Return the canonical identity of a JSON value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load(path: str | Path) -> dict:
    """Reject incomplete or inconsistent trajectories before opening a server."""
    data = json.loads(Path(path).read_text())
    if data["schema"] != "native-kv-wait-v1":
        raise ValueError("unknown workload schema")
    rows = data["requests"]
    by_id = {r["id"]: r for r in rows}
    if not rows or len(rows) != len(by_id):
        raise ValueError("empty or duplicate workload")
    seen = set()
    for row in rows:
        rid, parent = row["id"], row["parent"]
        tokens = row["answer_tokens"]
        if (not tokens or tokens[-1] != 151645 or 151645 in tokens[:-1]
                or len(tokens) > 2048 or any(type(t) is not int or t < 0
                                          for t in tokens + row["input_tokens"])):
            raise ValueError("invalid token reference or public budget")
        if parent:
            if parent not in seen:
                raise ValueError("requests must be in dependency order")
            predecessor = by_id[parent]
            expected = (predecessor["input_tokens"] + predecessor["answer_tokens"]
                        + row["suffix_tokens"])
            if expected != row["input_tokens"]:
                raise ValueError("history is not exact cumulative token history")
        elif row["root_arrival_s"] is None:
            raise ValueError("missing root arrival")
        for value in (row["think_s"], row["root_arrival_s"] or 0):
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid arrival or think time")
        seen.add(rid)
    if data["promotion"] != "off":
        raise ValueError("this package observes the native OFF baseline only")
    return data


def actual_input(row: dict, completed: dict) -> list[int]:
    """Build each child from its parent's actual returned tokens."""
    if row["parent"]:
        parent = completed[row["parent"]]
        if parent["status"] != "ok":
            raise ValueError("predecessor failed")
        prompt = (parent["input_tokens"] + parent["output_tokens"]
                  + row["suffix_tokens"])
    else:
        prompt = list(row["input_tokens"])
    if prompt != row["input_tokens"]:
        raise ValueError("returned history diverges from frozen workload")
    return prompt
