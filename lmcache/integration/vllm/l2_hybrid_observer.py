# SPDX-License-Identifier: Apache-2.0
"""Opt-in observation of the pinned native AsyncScheduler, without decisions.

The inherited scheduler owns all allocation, ordering, and completion changes.
Timestamps bracket native calls: a schedule event is not a GPU execution time,
and a receive-processing interval is not an exact internal adoption timestamp.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import queue
import threading
import time

# Third Party
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


class TraceSink:
    """Bounded asynchronous JSONL export with occurrence times and loss counts."""

    def __init__(self, path: Path) -> None:
        """Open a new trace file; an existing path raises FileExistsError."""
        self.output = path.open("x")
        self.pending: queue.Queue[dict[str, Any] | None] = queue.Queue(65536)
        self.dropped = 0
        self.written = 0
        self.error: str | None = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.emit(
            "header",
            schema_version=1,
            pid=os.getpid(),
            clock="CLOCK_MONOTONIC_ns",
            wall_ns=time.time_ns(),
            scope="host observations; GPU bounds require profiler",
        )

    def emit(self, event: str, **fields: Any) -> None:
        """Enqueue a JSON-compatible event without waiting for file I/O."""
        row = {"event": event, "monotonic_ns": time.monotonic_ns(), **fields}
        try:
            self.pending.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        """Drain and append a loss/error footer, failing on an unclean close."""
        if self.closed:
            return
        self.closed = True
        self.pending.put(None, timeout=10)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("observation writer did not drain")
        self.output.write(
            json.dumps(
                {
                    "event": "footer",
                    "written": self.written,
                    "dropped": self.dropped,
                    "error": self.error,
                }
            )
            + "\n"
        )
        self.output.close()
        if self.error or self.dropped:
            raise RuntimeError("observation trace is incomplete")

    def _run(self) -> None:
        try:
            while (row := self.pending.get()) is not None:
                self.output.write(json.dumps(row) + "\n")
                self.written += 1
        except Exception as error:
            self.error = repr(error)


class ObservedAsyncScheduler(AsyncScheduler):
    """Read and export native scheduler decisions after the native call returns."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct the unmodified native scheduler and a process-local trace."""
        super().__init__(*args, **kwargs)
        directory = Path(os.environ["L2_HYBRID_OBSERVATION_DIR"])
        self.observation = TraceSink(directory / f"scheduler-{os.getpid()}.jsonl")
        self.observation_step = 0
        self.observation_batches: dict[int, int] = {}
        self.observation_last_state: dict[str, Any] = {}
        self.observation.emit(
            "effective_config",
            async_scheduling=self.scheduler_config.async_scheduling,
            max_concurrent_batches=self.vllm_config.max_concurrent_batches,
            max_in_flight_tokens=self.vllm_config.max_in_flight_tokens,
            use_v2_model_runner=self.vllm_config.use_v2_model_runner,
            pipeline_parallel_size=self.parallel_config.pipeline_parallel_size,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            effective_max_scheduled_tokens=self.max_num_scheduled_tokens,
            max_model_len=self.max_model_len,
            num_gpu_blocks=self.kv_cache_manager.block_pool.num_gpu_blocks,
            block_size=self.cache_config.block_size,
            enforce_eager=self.vllm_config.model_config.enforce_eager,
        )

    def add_request(self, request: Request) -> None:
        """Record actual core enqueue around the inherited enqueue operation."""
        before = time.monotonic_ns()
        super().add_request(request)
        self.observation.emit(
            "core_enqueue",
            request_id=request.request_id,
            begin_ns=before,
            end_ns=time.monotonic_ns(),
            prompt_tokens=request.num_prompt_tokens,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Export computed context and physical blocks selected by native logic."""
        before = time.monotonic_ns()
        output = super().schedule(throttle_prefills)
        after = time.monotonic_ns()
        self.observation_step += 1
        self.observation_batches[id(output)] = self.observation_step
        contexts = {
            item.req_id: item.num_computed_tokens for item in output.scheduled_new_reqs
        }
        contexts.update(
            zip(
                output.scheduled_cached_reqs.req_ids,
                output.scheduled_cached_reqs.num_computed_tokens,
                strict=True,
            )
        )
        if output.num_scheduled_tokens or output.preempted_req_ids:
            requests = []
            for request_id, count in output.num_scheduled_tokens.items():
                request = self.requests[request_id]
                context = contexts[request_id]
                requests.append(
                    {
                        "request_id": request_id,
                        "scheduled_tokens": count,
                        "computed_context_tokens": context,
                        "prompt_tokens": request.num_prompt_tokens,
                        "scheduled_prompt_tokens": max(
                            0, min(count, request.num_prompt_tokens - context)
                        ),
                        "block_ids": self.kv_cache_manager.get_block_ids(request_id),
                    }
                )
            self.observation.emit(
                "schedule",
                batch_id=self.observation_step,
                begin_ns=before,
                end_ns=after,
                requests=requests,
                preempted_request_ids=sorted(output.preempted_req_ids or ()),
                gpu_begin_ns=None,
                gpu_end_ns=None,
                gpu_null_reason="native scheduling precedes device execution",
            )
        self._observe_state()
        return output

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        """Bracket native receive adoption and export its actual worker IDs."""
        before = time.monotonic_ns()
        transfer = model_runner_output.kv_connector_output
        receiving = sorted(transfer.finished_recving or ()) if transfer else []
        sending = sorted(transfer.finished_sending or ()) if transfer else []
        output = super().update_from_output(scheduler_output, model_runner_output)
        after = time.monotonic_ns()
        batch_id = self.observation_batches.pop(id(scheduler_output), None)
        if receiving or sending or scheduler_output.total_num_scheduled_tokens:
            self.observation.emit(
                "scheduler_output_processed",
                batch_id=batch_id,
                begin_ns=before,
                end_ns=after,
                finished_receiving=receiving,
                finished_sending=sending,
                adoption_ns=None,
                adoption_bounds_reason="inside native update_from_output interval",
            )
        self._observe_state()
        return output

    def shutdown(self) -> None:
        """Drain native connectors, then persist the observation footer."""
        try:
            super().shutdown()
        finally:
            self._observe_state()
            self.observation.close()

    def _observe_state(self) -> None:
        pool = self.kv_cache_manager.block_pool
        categories: dict[str, set[int]] = {
            "active": set(),
            "remote_reserved": set(),
            "finished_held": set(),
            "other_request": set(),
        }
        requests = []
        for request_id, request in self.requests.items():
            status = request.status.name
            category = (
                "finished_held"
                if request.is_finished()
                else "remote_reserved"
                if status == "WAITING_FOR_REMOTE_KVS"
                else "active"
                if status == "RUNNING"
                else "other_request"
            )
            blocks = self.kv_cache_manager.get_block_ids(request_id)
            categories[category].update(b for group in blocks for b in group if b)
            requests.append({"request_id": request_id, "status": status})
        state = {
            "requests": requests,
            "category_blocks": {
                key: sorted(value) for key, value in categories.items()
            },
            "protected_blocks": [b.block_id for b in pool.blocks if b.ref_cnt > 0],
            "retained_unreferenced_blocks": sum(
                b.ref_cnt == 0 and b.block_hash is not None for b in pool.blocks
            ),
            "free_blocks": pool.get_num_free_blocks(),
            "total_blocks": pool.num_gpu_blocks,
            "null_block_id": pool.null_block.block_id,
        }
        if state != self.observation_last_state:
            self.observation.emit("scheduler_state", **state)
            self.observation_last_state = state
