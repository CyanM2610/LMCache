# SPDX-License-Identifier: Apache-2.0
"""Public timing adapter contracts without starting a model or GPU."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import l2_hybrid_timing as timing


def test_timing_preserves_outputs_and_correlates_repeated_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repeated signatures retain occurrences, exact results and drain accounting."""
    ready = [False]

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def record(self, stream: Any) -> None:
            pass

        def query(self) -> bool:
            return ready[0]

        def elapsed_time(self, end: Any) -> float:
            return 1.25

    execute_result, sample_result = object(), object()
    monkeypatch.setenv("L2_HYBRID_OBSERVATION_DIR", str(tmp_path))
    monkeypatch.setattr(timing.Worker, "__init__", lambda self: None)
    monkeypatch.setattr(timing.Worker, "shutdown", lambda self: None)
    monkeypatch.setattr(
        timing.Worker, "execute_model", lambda self, arg: execute_result
    )
    monkeypatch.setattr(timing.Worker, "sample_tokens", lambda self, arg: sample_result)
    monkeypatch.setattr(
        timing,
        "torch_dev",
        SimpleNamespace(
            Event=Event, current_stream=lambda: SimpleNamespace(cuda_stream=7)
        ),
    )
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 1},
        total_num_scheduled_tokens=2,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a", "b"], num_computed_tokens=[128, 128]
        ),
    )
    worker = timing.TimedWorker()
    for _ in range(2):
        assert worker.execute_model(output) is execute_result
        assert worker.sample_tokens(None) is sample_result
    ready[0] = True
    worker.shutdown()
    rows = [
        json.loads(s)
        for s in next(tmp_path.glob("worker-*.jsonl")).read_text().splitlines()
    ]
    cuda = [r for r in rows if r["event"] == "worker_cuda"]
    assert [(r["worker_step_id"], r["phase"]) for r in cuda] == [
        (1, "execute"),
        (1, "sample"),
        (2, "execute"),
        (2, "sample"),
    ]
    assert [r["signature_occurrence"] for r in cuda] == [1, 1, 2, 2]
    assert all(r["cuda_elapsed_ms"] == 1.25 for r in cuda)
    assert rows[-1]["dropped"] == 0 and rows[-1]["error"] is None
    assert next(r for r in rows if r["event"] == "timing_drain")["pending"] == 0
