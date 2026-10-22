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


def test_native_copy_fence_and_future_consumption_are_correlated(
    tmp_path: Path,
) -> None:
    """Ready copy events do not imply native Futures were already consumed."""
    # Third Party
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture

    ready = [False]
    calls = []
    expected = object()

    def get_output() -> Any:
        calls.append("native")
        ready[0] = True
        return expected

    native: Any = SimpleNamespace(
        copy_event=SimpleNamespace(cuda_event=123, query=lambda: ready[0]),
        get_output=get_output,
    )
    sink = timing.TraceSink(tmp_path / "output.jsonl")
    output = timing.ObservedAsyncOutput(native, sink, {"worker_step_id": 7}, 456)
    future = AsyncOutputFuture(output, single_value=True)
    assert not future.done()
    assert future.result() is expected
    assert future.result() is expected
    assert calls == ["native"]
    sink.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "output.jsonl").read_text().splitlines()
    ]
    handoff = next(r for r in records if r["event"] == "async_output_handoff")
    consumed = next(r for r in records if r["event"] == "async_output_consumed")
    assert handoff["copy_event_id"] == consumed["copy_event_id"] == 123
    assert consumed["worker_step_id"] == 7 and consumed["copy_stream_id"] == 456
    assert consumed["copy_fence_ready_at_entry"] is False
    assert consumed["copy_fence_ready_at_return"] is True and consumed["error"] is None
    assert handoff["output_copy_cuda_ms"] is None


@pytest.mark.parametrize("native_fails", [False, True])
def test_copy_observation_errors_preserve_native_result_or_exception(
    tmp_path: Path, native_fails: bool
) -> None:
    """Handoff/entry/return diagnostic failures never replace native consumption."""
    calls = []
    expected = object()
    native_error = ValueError("native output failure")

    def broken_query() -> bool:
        raise RuntimeError("diagnostic CUDA query failure")

    def get_output() -> Any:
        calls.append("native")
        if native_fails:
            raise native_error
        return expected

    native: Any = SimpleNamespace(
        copy_event=SimpleNamespace(cuda_event=123, query=broken_query),
        get_output=get_output,
    )
    sink = timing.TraceSink(tmp_path / "failed-query.jsonl")
    output = timing.ObservedAsyncOutput(native, sink, {"worker_step_id": 9}, 456)
    try:
        if native_fails:
            with pytest.raises(ValueError) as raised:
                output.get_output()
            assert raised.value is native_error
        else:
            assert output.get_output() is expected
    finally:
        sink.close()
    assert calls == ["native"]
    records = [
        json.loads(line)
        for line in (tmp_path / "failed-query.jsonl").read_text().splitlines()
    ]
    consumed = next(r for r in records if r["event"] == "async_output_consumed")
    assert consumed["copy_fence_ready_at_entry"] is None
    assert consumed["copy_fence_ready_at_return"] is None
    assert "diagnostic CUDA query failure" in consumed["copy_fence_entry_query_error"]
    assert "diagnostic CUDA query failure" in consumed["copy_fence_return_query_error"]
