# SPDX-License-Identifier: Apache-2.0
"""Opt-in current-stream envelopes around the pinned native worker calls.

These event intervals include launch gaps and stream dependencies. They do not
cover uncorrelated async output-copy streams or measure kernel-active time.
"""

# Standard
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable
import os
import time

# Third Party
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu_worker import Worker

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.l2_hybrid_observer import TraceSink


def query_copy_fence(event: Any) -> dict[str, Any]:
    """Observe a native event without changing the caller's output/error path.

    Args:
        event: Already-recorded native output-copy event.
    Returns:
        Event handle/readiness, or null readiness plus diagnostic error. Failures
        invalidate timing evidence but never replace native get_output behavior.
    """
    identity = None
    try:
        identity = int(event.cuda_event)
        return dict(event_id=identity, ready=event.query(), error=None)
    except Exception as error:
        return dict(event_id=identity, ready=None, error=repr(error))


class ObservedAsyncOutput(AsyncModelRunnerOutput):
    """Delegate native output consumption and correlate its existing copy fence."""

    def __init__(
        self,
        output: AsyncOutput,
        sink: TraceSink,
        identity: dict[str, Any],
        copy_stream_id: int,
    ) -> None:
        """Retain the native result and query its already-recorded copy event.

        Args:
            output: Unmodified native async result with a blocking copy event.
            sink: Process-local loss-auditable writer.
            identity: Exact producing worker step/signature/occurrence.
            copy_stream_id: Native output-copy stream identity.
        Notes:
            Adds no wait or new stream dependency; the native event has timing
            disabled, so these observations cannot provide a DMA duration.
        """
        self.output = output
        self.sink = sink
        self.identity = dict(identity)
        self.copy_stream_id = copy_stream_id
        observation = query_copy_fence(output.copy_event)
        self.sink.emit(
            "async_output_handoff",
            **self.identity,
            copy_stream_id=copy_stream_id,
            copy_event_id=observation["event_id"],
            copy_fence_ready=observation["ready"],
            copy_fence_query_error=observation["error"],
            output_copy_cuda_ms=None,
            null_reason="native completion event has no timing/start marker",
        )

    def get_output(self) -> ModelRunnerOutput:
        """Return the identical native output after native future consumption.

        Returns:
            Native ModelRunnerOutput without copying or changing its contents.
        Raises:
            Exception: Propagates native consumption errors after recording them.
        Notes:
            The envelope includes any native copy-fence wait and CPU conversion.
            It is not CPU-active time and overlaps prior GPU/output-copy work.
        """
        before = time.monotonic_ns()
        entry = query_copy_fence(self.output.copy_event)
        error = None
        try:
            return self.output.get_output()
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            after = time.monotonic_ns()
            returned = query_copy_fence(self.output.copy_event)
            self.sink.emit(
                "async_output_consumed",
                **self.identity,
                copy_stream_id=self.copy_stream_id,
                copy_event_id=returned["event_id"],
                begin_ns=before,
                end_ns=after,
                copy_fence_ready_at_entry=entry["ready"],
                copy_fence_ready_at_return=returned["ready"],
                copy_fence_entry_query_error=entry["error"],
                copy_fence_return_query_error=returned["error"],
                error=error,
            )


class TimedWorker(Worker):
    """Observe inherited execute/sample calls without changing their results."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct the native worker and a fresh, loss-auditable trace."""
        super().__init__(*args, **kwargs)
        path = Path(os.environ["L2_HYBRID_OBSERVATION_DIR"])
        self.timing_sink = TraceSink(path / f"worker-{os.getpid()}.jsonl")
        self.timing_pending: deque[tuple[Any, Any, dict[str, Any]]] = deque()
        self.timing_occurrences: Counter[tuple[tuple[str, int], ...]] = Counter()
        self.timing_step = 0
        self.timing_identity: dict[str, Any] = {}
        self.timing_sink.emit(
            "timing_scope",
            role="instrument-validation/training",
            cuda_scope="current compute stream event envelopes, not kernel active",
            output_copy_cuda_ms=None,
            output_copy_null_reason=(
                "native copy fence correlated separately; DMA duration unavailable"
            ),
            synchronization="query during execution; bounded query drain at shutdown",
        )

    def execute_model(self, scheduler_output: Any) -> Any:
        """Return native output unchanged and record its exact schedule signature."""
        self.timing_step += 1
        signature = tuple(scheduler_output.num_scheduled_tokens.items())
        self.timing_occurrences[signature] += 1
        contexts = {
            item.req_id: item.num_computed_tokens
            for item in scheduler_output.scheduled_new_reqs
        }
        contexts.update(
            zip(
                scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                strict=True,
            )
        )
        self.timing_identity = {
            "worker_step_id": self.timing_step,
            "signature": signature,
            "signature_occurrence": self.timing_occurrences[signature],
            "contexts": contexts,
            "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
        }
        return self._observe("execute", super().execute_model, scheduler_output)

    def sample_tokens(self, grammar_output: Any) -> Any:
        """Observe current-stream work and delegate native async consumption."""
        output = self._observe("sample", super().sample_tokens, grammar_output)
        if isinstance(output, AsyncOutput):
            return ObservedAsyncOutput(
                output,
                self.timing_sink,
                self.timing_identity,
                int(self.model_runner.output_copy_stream.cuda_stream),
            )
        return output

    def shutdown(self) -> None:
        """Drain timing events before native shutdown and persist the trace footer.

        Raises:
            RuntimeError: If the bounded event drain or trace export fails.
                Native shutdown still runs; a timing_drain record with pending
                events invalidates this trace even when the writer is clean.
        """
        try:
            deadline = time.monotonic() + 20
            while self.timing_pending and time.monotonic() < deadline:
                self._collect()
                if self.timing_pending:
                    time.sleep(0.001)
            self.timing_sink.emit("timing_drain", pending=len(self.timing_pending))
            if self.timing_pending:
                raise RuntimeError("worker timing events did not drain")
        finally:
            try:
                super().shutdown()
            finally:
                self.timing_sink.close()

    def _observe(self, phase: str, call: Callable[..., Any], argument: Any) -> Any:
        self._collect()
        stream = torch_dev.current_stream()
        start = torch_dev.Event(enable_timing=True)
        end = torch_dev.Event(enable_timing=True)
        start.record(stream)
        before = time.monotonic_ns()
        try:
            return call(argument)
        finally:
            after = time.monotonic_ns()
            end.record(stream)
            row = {
                **self.timing_identity,
                "phase": phase,
                "stream_id": int(stream.cuda_stream),
                "host_begin_ns": before,
                "host_return_ns": after,
                "cuda_event_enqueued_ns": time.monotonic_ns(),
            }
            self.timing_sink.emit("worker_host", **row)
            self.timing_pending.append((start, end, row))

    def _collect(self) -> None:
        while self.timing_pending and self.timing_pending[0][1].query():
            start, end, row = self.timing_pending.popleft()
            self.timing_sink.emit(
                "worker_cuda",
                **row,
                cuda_elapsed_ms=start.elapsed_time(end),
                completion_observed_ns=time.monotonic_ns(),
                cuda_absolute_begin_ns=None,
                cuda_absolute_end_ns=None,
            )
