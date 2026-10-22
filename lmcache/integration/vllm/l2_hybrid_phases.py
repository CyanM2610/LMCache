# SPDX-License-Identifier: Apache-2.0
"""Optional native phase recorder: event recording/query only, no stream waits."""

# Standard
from collections import deque
from typing import Any, Callable
import threading
import time

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.l2_hybrid_observer import TraceSink


def cpu_clock() -> dict[str, int]:
    """Return wall, current-thread CPU and process CPU clocks, all in nanoseconds."""
    return dict(
        wall_ns=time.monotonic_ns(),
        thread_cpu_ns=time.thread_time_ns(),
        process_cpu_ns=time.process_time_ns(),
        thread_id=threading.get_ident(),
    )


class PhaseRecorder:
    """Capture immutable step identities and asynchronously drain CUDA intervals."""

    def __init__(self, sink: TraceSink, identity: Callable[[], dict[str, Any]]) -> None:
        """Bind the sink and producing-step snapshot callback.

        Args:
            sink: Loss-auditable writer owned by the calling worker.
            identity: Returns a snapshot containing integer worker_step_id and
                the producing schedule signature. Captures outlive this step.
        """
        self.sink = sink
        self.identity = identity
        self.starts: dict[tuple[int, str], Any] = {}
        self.pending: deque[tuple[Any, Any, dict[str, Any]]] = deque()
        self.errors: list[str] = []

    def capture(self) -> Callable[..., None]:
        """Return a marker bound to the current step, including later consumption."""
        identity = dict(self.identity())

        def mark(name: str, **fields: Any) -> None:
            before = cpu_clock()
            stream = fields.pop("stream", None)
            try:
                self.sink.emit(
                    "native_phase", **{**identity, **fields}, name=name, **before
                )
                if stream is not None:
                    event = torch_dev.Event(enable_timing=True)
                    event.record(stream)
                    key = (identity.get("worker_step_id", 0), name)
                    self.starts[key] = event
                    intervals = {
                        "sample_before_output_fork": (
                            "sample_begin",
                            "sample_pre_fork",
                        ),
                        "sample_after_output_fork": (
                            "sample_before_output_fork",
                            "fork_enqueue_gap",
                        ),
                        "sample_tail_end": (
                            "sample_after_output_fork",
                            "sample_main_tail",
                        ),
                        "output_copy_end": ("output_copy_start", "output_copy"),
                    }
                    if name in intervals:
                        start_name, phase = intervals[name]
                        start = self.starts.pop((key[0], start_name))
                        self.pending.append(
                            (
                                start,
                                event,
                                dict(
                                    **identity,
                                    phase=phase,
                                    stream_id=int(stream.cuda_stream),
                                ),
                            )
                        )
                    if name in ("sample_tail_end", "output_copy_end"):
                        self.starts.pop(key)
                self.sink.emit(
                    "phase_observer_cost",
                    name=name,
                    worker_step_id=identity.get("worker_step_id"),
                    begin=before,
                    end=cpu_clock(),
                )
            except Exception as error:
                self.errors.append(repr(error))

        return mark

    def sample_begin(self, identity: dict[str, Any], event: Any) -> None:
        """Reuse the existing outer sample start marker, without another wait.

        Args:
            identity: Producing identity containing integer worker_step_id.
            event: Timing-enabled event already recorded on the main stream;
                paired with that step's sample_before_output_fork marker.
        """
        self.starts[(identity["worker_step_id"], "sample_begin")] = event

    def collect(self) -> None:
        """Query completed intervals without synchronizing a stream/device.

        Diagnostic failures retain interval identities in errors. They do not
        interrupt native work; close() rejects the incomplete evidence later.
        """
        remaining: deque[tuple[Any, Any, dict[str, Any]]] = deque()
        while self.pending:
            start, end, row = self.pending.popleft()
            try:
                if end.query():
                    self.sink.emit(
                        "phase_cuda",
                        **row,
                        elapsed_ms=start.elapsed_time(end),
                        observed_ns=time.monotonic_ns(),
                        absolute_gpu_start_ns=None,
                        absolute_gpu_end_ns=None,
                    )
                else:
                    remaining.append((start, end, row))
            except Exception as error:
                self.errors.append(f"interval {row!r}: {error!r}")
        self.pending = remaining

    def close(self) -> None:
        """Bounded query drain; raise RuntimeError if any section/error remains."""
        deadline = time.monotonic() + 20
        while self.pending and time.monotonic() < deadline:
            self.collect()
            if self.pending:
                time.sleep(0.001)
        # Third Party
        from vllm.v1.worker.gpu.phase_observation import phase_observation_errors

        self.errors.extend(phase_observation_errors())
        self.sink.emit(
            "phase_drain",
            pending=len(self.pending),
            unpaired=len(self.starts),
            errors=self.errors,
        )
        if self.pending or self.starts or self.errors:
            raise RuntimeError("native phase trace incomplete")
