# SPDX-License-Identifier: Apache-2.0
"""Read-only scheduler/allocator snapshots for the bounded native006 witness."""

# Standard
from pathlib import Path
from typing import Any
import os

# Third Party
from vllm.v1.core.retirement_observation import (
    retirement_errors,
    set_retirement_observer,
)

# First Party
from lmcache.integration.vllm.l2_hybrid_observer import TraceSink
from lmcache.integration.vllm.l2_hybrid_phases import cpu_clock


def scheduler_snapshot(scheduler: Any) -> dict[str, Any]:
    """Copy current public allocation and fence state without changing ownership.

    Args:
        scheduler: Actual initialized native scheduler.
    Returns:
        Physical IDs/refcounts/free order, request generations and global fences.
    """
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    return dict(
        requires_kv_delivery=scheduler.requires_kv_delivery,
        defer_block_free=scheduler.defer_block_free,
        max_concurrent_batches=scheduler.vllm_config.max_concurrent_batches,
        sched_step_seq=scheduler.sched_step_seq,
        processed_step_seq=scheduler.processed_step_seq,
        deferred_frees=[
            dict(fence=fence, block_ids=[b.block_id for b in blocks])
            for fence, blocks in scheduler.deferred_frees
        ],
        requests={
            key: dict(
                generation=req.num_preemptions,
                status=req.status.name,
                last_sched_seq=req.last_sched_seq,
                in_flight=req.num_in_flight_tokens,
                stale=req.num_stale_output_tokens,
                placeholders=req.num_output_placeholders,
                drop_stale_output=req.drop_stale_output,
                computed=req.num_computed_tokens,
                output_tokens=list(req.output_token_ids),
                block_ids=manager.get_block_ids(key),
            )
            for key, req in scheduler.requests.items()
        },
        pool=[
            dict(
                block_id=b.block_id,
                ref_cnt=b.ref_cnt,
                is_null=b.is_null,
                block_hash=b.block_hash.hex() if b.block_hash is not None else None,
            )
            for b in pool.blocks
        ],
        free_queue=[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
    )


def worker_snapshot(worker: Any) -> dict[str, Any]:
    """Read existing MP future/event state without importing replies or waiting.

    Args:
        worker: Native LMCache worker adapter.
    Returns:
        Context identity, pending STORE/LOAD and existing producer event queries.
    """

    def future_state(future: Any) -> dict[str, Any]:
        return dict(identity=id(future), **future.observation_state())

    return dict(
        healthy=worker.is_healthy,
        transfer_context=type(worker.transfer_ctx).__qualname__,
        pending_store={
            key: future_state(value) for key, value in worker.store_futures.items()
        },
        pending_load={
            key: dict(**future_state(value[0]), block_ids=list(value[1]))
            for key, value in worker.retrieve_futures.items()
        },
        store_producer_events={
            key: dict(identity=id(value), ready=value.query())
            for key, value in worker.store_events.items()
        },
        load_producer_events={
            key: dict(identity=id(value), ready=value.query())
            for key, value in worker.retrieve_events.items()
        },
    )


class RetirementRecorder:
    """Bind immutable batch generations to exact native retirement/free calls."""

    def __init__(self, scheduler: Any, directory: Path) -> None:
        """Install this opt-in recorder after native scheduler construction.

        Args:
            scheduler: Initialized native scheduler with public allocation state.
            directory: Existing directory for a fresh process-specific trace.
        Raises:
            OSError: The trace cannot be created, including an existing filename.
        """
        self.scheduler = scheduler
        initial_state = scheduler_snapshot(scheduler)
        self.sink = TraceSink(directory / f"retirement-{os.getpid()}.jsonl")
        self.batches: dict[int, dict[str, Any]] = {}
        self.sink.emit("retirement_initial", state=initial_state)
        set_retirement_observer(self.observe)

    def observe(self, name: str, owner: Any, **fields: Any) -> None:
        """Snapshot a native boundary, copying identities before mutable reuse.

        Args:
            name: Named native method or stale-output decision.
            owner: Original method receiver; never mutated by this callback.
            fields: Call phase/arguments/result or explicit decision fields.
        Notes:
            Wrapper catches observation errors, preserving native execution.
            No result(), wait(), synchronize(), new event or device copy is used.
        """
        begin = cpu_clock()
        phase = fields.get("phase", "point")
        args, kwargs = fields.get("args", ()), fields.get("kwargs", {})
        result = fields.get("result")
        state = scheduler_snapshot(self.scheduler)
        row: dict[str, Any] = dict(
            name=name,
            phase=phase,
            call_id=fields.get("call_id"),
            clock=begin,
            error=fields.get("error"),
            state=state,
        )
        if name in ("preempt", "free_request", "allocate_slots"):
            request = args[0] if args else kwargs["request"]
            row["request_id"] = request.request_id
        if name == "preempt":
            row["drop_stale_argument"] = kwargs.get(
                "drop_stale_output", args[2] if len(args) > 2 else False
            )
        if name == "pool_allocate" and phase == "end" and result is not None:
            row["allocated_ids"] = [b.block_id for b in result]
        if name == "allocate_slots" and phase == "end":
            row["allocation_succeeded"] = result is not None
        if name == "schedule" and phase == "end" and result is not None:
            batch = dict(
                schedule_call_id=fields["call_id"],
                sched_step_seq=state["sched_step_seq"],
                signature=list(result.num_scheduled_tokens.items()),
                requests={
                    key: state["requests"][key] for key in result.num_scheduled_tokens
                },
                preempted_request_ids=sorted(result.preempted_req_ids or ()),
                need_flush_before_forward=getattr(
                    result.kv_connector_metadata, "need_flush_before_forward", None
                ),
            )
            if result.total_num_scheduled_tokens:
                self.batches[id(result)] = batch
            row["submitted_batch"] = batch
        if name == "consume":
            row["submitted_batch"] = self.batches.get(id(args[0]))
            row["sampled_token_ids"] = [list(v) for v in args[1].sampled_token_ids]
            row["sampled_request_ids"] = list(args[1].req_ids)
            if phase == "end":
                self.batches.pop(id(args[0]), None)
        if name == "stale_drop":
            row["request_id"] = fields["request_id"]
            row["scheduled_tokens"] = fields["scheduled_tokens"]
            output = fields["model_runner_output"]
            index = output.req_id_to_index.get(fields["request_id"])
            row["sampled_token_ids"] = (
                list(output.sampled_token_ids[index])
                if index is not None and output.sampled_token_ids
                else None
            )
            row["missing_stale_output_index"] = index is None
        if name.startswith("worker_"):
            row["worker"] = worker_snapshot(owner)
            if args and isinstance(args[0], str):
                row["request_id"] = args[0]
            if len(args) > 1 and hasattr(args[1], "block_ids"):
                row["operation"] = dict(
                    start=args[1].start,
                    end=args[1].end,
                    block_ids=[list(v) for v in args[1].block_ids],
                    skip_first_n_tokens=args[1].skip_first_n_tokens,
                )
            if name == "worker_preempt":
                row["need_flush_before_forward"] = args[0]
        self.sink.emit("retirement_observation", **row)
        self.sink.emit(
            "retirement_observer_cost",
            name=name,
            call_id=fields.get("call_id"),
            begin=begin,
            end=cpu_clock(),
        )

    def close(self) -> None:
        """Disable the callback and drain evidence even if the final snapshot fails.

        Raises:
            RuntimeError: Pending batches, observation errors or incomplete drain.
            OSError: The trace writer cannot finish exporting its file.
        """
        errors = retirement_errors()
        set_retirement_observer(None)
        state = None
        try:
            state = scheduler_snapshot(self.scheduler)
        except Exception as error:
            errors.append("final snapshot: " + repr(error))
        try:
            self.sink.emit(
                "retirement_drain",
                pending_batches=list(self.batches.values()),
                errors=errors,
                state=state,
            )
        finally:
            self.sink.close()
        if errors or self.batches:
            raise RuntimeError("retirement witness incomplete")
