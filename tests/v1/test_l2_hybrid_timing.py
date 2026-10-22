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


def test_phase_stream_intervals_keep_producing_identity_without_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Copy/main fork observations survive later steps and only query CUDA events."""
    # First Party
    from lmcache.integration.vllm import l2_hybrid_phases as phases

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            self.stream = None

        def record(self, stream: Any) -> None:
            self.stream = stream.cuda_stream

        def query(self) -> bool:
            return True

        def elapsed_time(self, end: Any) -> float:
            assert self.stream == end.stream
            return 0.5

    monkeypatch.setattr(phases, "torch_dev", SimpleNamespace(Event=Event))
    sink = timing.TraceSink(tmp_path / "phases.jsonl")
    identity = dict(worker_step_id=1)
    recorder = phases.PhaseRecorder(sink, lambda: identity)
    main, copy = SimpleNamespace(cuda_stream=1), SimpleNamespace(cuda_stream=2)
    begin = Event()
    begin.record(main)
    recorder.sample_begin(identity, begin)
    mark = recorder.capture()
    mark("sample_before_output_fork", stream=main)
    mark("output_copy_start", stream=copy)
    mark("output_copy_end", stream=copy)
    mark("sample_after_output_fork", stream=main)
    mark("sample_tail_end", stream=main)
    identity["worker_step_id"] = 2
    mark("output_wait_begin")
    recorder.close()
    sink.close()
    rows = [json.loads(s) for s in (tmp_path / "phases.jsonl").read_text().splitlines()]
    intervals = [r for r in rows if r["event"] == "phase_cuda"]
    assert {r["phase"] for r in intervals} == {
        "sample_pre_fork",
        "fork_enqueue_gap",
        "sample_main_tail",
        "output_copy",
    }
    assert all(r["worker_step_id"] == 1 for r in intervals)
    assert (
        next(r for r in rows if r.get("name") == "output_wait_begin")["worker_step_id"]
        == 1
    )
    assert next(r for r in rows if r["event"] == "phase_drain")["unpaired"] == 0


@pytest.mark.parametrize("fails", [False, True])
def test_phase_callback_failure_cannot_replace_native_result(fails: bool) -> None:
    """Observer faults are audited while native values and exceptions survive."""
    # Third Party
    from vllm.v1.worker.gpu.phase_observation import (
        observed_host_call,
        phase_observation_errors,
        set_phase_observer,
    )

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("observer failed")

    expected = object()
    error = ValueError("native failed")

    @observed_host_call("native")
    def native() -> Any:
        if fails:
            raise error
        return expected

    set_phase_observer(lambda: broken)
    try:
        if fails:
            with pytest.raises(ValueError) as caught:
                native()
            assert caught.value is error
        else:
            assert native() is expected
        assert len(phase_observation_errors()) == 2
    finally:
        # Reset the process-local diagnostic state for other tests.
        set_phase_observer(lambda: broken)
        set_phase_observer(None)


def test_native_output_keeps_one_copy_wait_and_conversion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional markers bracket the existing wait, leaving IDs unchanged."""
    # Third Party
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu import async_utils
    import numpy as np

    calls = []

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def record(self, stream: Any) -> None:
            calls.append("fence_record")

        def synchronize(self) -> None:
            calls.append("native_wait")

    monkeypatch.setattr(async_utils.torch.cuda, "Event", Event)
    monkeypatch.setattr(async_utils.torch.cuda, "set_stream", lambda s: None)
    monkeypatch.setattr(async_utils, "async_copy_to_np", lambda x: x)
    stream = SimpleNamespace(wait_stream=lambda s: calls.append("native_dependency"))
    result = ModelRunnerOutput(
        req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[]
    )
    sampler = SimpleNamespace(
        sampled_token_ids=np.array([[7, 8]]),
        logprobs_tensors=None,
        num_nans=None,
        sampling_mask_tensors=None,
    )
    output = async_utils.AsyncOutput(
        result,
        sampler,
        np.array([1]),
        stream,
        stream,
        phase_observer=lambda name, **kw: calls.append(name),
    )
    assert output.get_output() is result
    assert result.sampled_token_ids == [[7]]
    assert calls == [
        "native_dependency",
        "output_copy_start",
        "output_copy_end",
        "fence_record",
        "output_wait_begin",
        "native_wait",
        "output_wait_end",
        "output_conversion_end",
    ]


@pytest.mark.parametrize("failure", ["query", "elapsed_time"])
def test_failed_phase_drain_is_audited_without_interrupting_native_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    """Diagnostic collection is non-throwing; terminal validation rejects it."""
    # First Party
    from lmcache.integration.vllm import l2_hybrid_phases as phases

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def record(self, stream: Any) -> None:
            pass

        def query(self) -> bool:
            if failure == "query":
                raise RuntimeError("query failed")
            return True

        def elapsed_time(self, end: Any) -> float:
            raise RuntimeError("elapsed failed")

    monkeypatch.setattr(phases, "torch_dev", SimpleNamespace(Event=Event))
    sink = timing.TraceSink(tmp_path / "failed-phase.jsonl")
    recorder = phases.PhaseRecorder(sink, lambda: {"worker_step_id": 1})
    mark = recorder.capture()
    stream = SimpleNamespace(cuda_stream=2)
    mark("output_copy_start", stream=stream)
    mark("output_copy_end", stream=stream)
    recorder.collect()  # Native caller may continue; error is retained for audit.
    assert recorder.errors
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.close()
    sink.close()
    rows = [
        json.loads(s)
        for s in (tmp_path / "failed-phase.jsonl").read_text().splitlines()
    ]
    assert next(r for r in rows if r["event"] == "phase_drain")["errors"]


def test_dispatch_description_reports_stored_callable_without_redispatch() -> None:
    """Changing the surrounding config cannot change a constructed-op observation."""
    # Third Party
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import CustomOp

    class Probe(CustomOp):
        name = "native005_dispatch_probe"

        def forward_native(self) -> str:
            return "forward_native"

        def forward_cuda(self) -> str:
            return "forward_cuda"

        def forward_cpu(self) -> str:
            return "forward_cpu"

    enabled = VllmConfig(
        compilation_config=CompilationConfig(mode=0, custom_ops=["all"])
    )
    disabled = VllmConfig(
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"])
    )
    with set_current_vllm_config(enabled):
        op = Probe()
        expected = op()
        description = op.get_dispatch_info()
        assert description["method_qualname"].endswith(expected)
    with set_current_vllm_config(disabled):
        assert op.get_dispatch_info() == description
        assert op() == expected
        assert Probe().get_dispatch_info()["method_qualname"].endswith("forward_native")


# Contract: public nn.Module hooks must preserve identity, exceptions and call
# counts; a CPU dummy is sufficient and must require no GPU setup.
def test_model_dispatch_hooks_preserve_results_and_failures() -> None:
    """Optional explanatory hooks preserve native calls and always pair exits."""
    # Third Party
    import torch

    # First Party
    from lmcache.integration.vllm.model_dispatch_observer import ModelDispatchObserver

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, value: object) -> object:
            self.calls += 1
            if isinstance(value, Exception):
                raise value
            return value

    model = Model()
    observation = ModelDispatchObserver(model)
    value = object()
    assert model(value) is value
    error = ValueError("native forward failed")
    with pytest.raises(ValueError) as caught:
        model(error)
    assert caught.value is error
    assert model.calls == 2
    assert observation.status() == {"pending": 0, "errors": []}
    observation.close()
    assert model(value) is value
    assert model.calls == 3


def test_retirement_observation_failure_preserves_native_call() -> None:
    """A failed witness callback cannot consume an iterator or change outcomes."""
    # Third Party
    from vllm.v1.core.retirement_observation import (
        retirement_call,
        retirement_errors,
        set_retirement_observer,
    )

    calls = []

    def broken_observer(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("observer failure")

    @retirement_call("test")
    def native(owner: Any, value: Any) -> Any:
        calls.append(owner)
        if isinstance(value, Exception):
            raise value
        return list(value)

    set_retirement_observer(broken_observer)
    try:
        assert native("first", iter([1, 2])) == [1, 2]
        failure = ValueError("native failure")
        with pytest.raises(ValueError) as caught:
            native("second", failure)
        assert caught.value is failure
        assert calls == ["first", "second"]
        assert len(retirement_errors()) == 4
    finally:
        set_retirement_observer(None)


def test_retirement_close_exports_snapshot_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Final read failure still disables callbacks and writes terminal evidence."""
    # First Party
    from lmcache.integration.vllm import retirement_witness

    monkeypatch.setattr(retirement_witness, "scheduler_snapshot", lambda _: {})
    recorder = retirement_witness.RetirementRecorder(object(), tmp_path)

    def fail(_: Any) -> Any:
        raise ValueError("snapshot failed")

    monkeypatch.setattr(retirement_witness, "scheduler_snapshot", fail)
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.close()
    recorded = [
        json.loads(line)
        for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    ]
    assert recorded[-1]["event"] == "footer"
    assert "snapshot failed" in recorded[-2]["errors"][0]


def test_scheduler_shutdown_closes_both_observers_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native shutdown failure and state-read failure cannot skip either writer."""
    # First Party
    from lmcache.integration.vllm.l2_hybrid_observer import (
        AsyncScheduler,
        ObservedAsyncScheduler,
    )

    def fail_native(_: Any) -> None:
        raise ValueError("native shutdown failed")

    def fail_snapshot(_: Any) -> None:
        raise RuntimeError("state failed")

    monkeypatch.setattr(AsyncScheduler, "shutdown", fail_native)
    # Inject failure at the private read boundary to test public shutdown cleanup.
    monkeypatch.setattr(ObservedAsyncScheduler, "_observe_state", fail_snapshot)
    scheduler: Any = ObservedAsyncScheduler.__new__(ObservedAsyncScheduler)
    closed = []
    scheduler.observation = SimpleNamespace(close=lambda: closed.append("scheduler"))
    scheduler.retirement_recorder = SimpleNamespace(
        close=lambda: closed.append("retirement")
    )
    with pytest.raises(ValueError, match="native shutdown failed"):
        scheduler.shutdown()
    assert closed == ["scheduler", "retirement"]
