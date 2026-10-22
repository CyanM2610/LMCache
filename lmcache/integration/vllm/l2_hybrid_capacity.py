# SPDX-License-Identifier: Apache-2.0
"""Report native allocator peaks at shutdown without changing scheduler or streams."""

# Standard
from pathlib import Path
import json
import os

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.l2_hybrid_observer import ObservedAsyncScheduler


class CapacityWorker(Worker):
    """Native worker with terminal allocator counters for the capacity canary."""

    def shutdown(self) -> None:
        """Export peak/current device allocation, always performing native shutdown.

        Raises:
            OSError: If the new evidence file cannot be written. Missing evidence
                invalidates the audit even if native shutdown suppresses this error.
        """
        try:
            row = dict(
                pid=os.getpid(),
                peak_allocated_bytes=torch_dev.max_memory_allocated(),
                peak_reserved_bytes=torch_dev.max_memory_reserved(),
                current_allocated_bytes=torch_dev.memory_allocated(),
                current_reserved_bytes=torch_dev.memory_reserved(),
            )
            directory = Path(os.environ["L2_HYBRID_OBSERVATION_DIR"])
            with (directory / f"memory-{os.getpid()}.json").open("x") as stream:
                json.dump(row, stream, indent=2)
        finally:
            super().shutdown()


class CapacityScheduler(ObservedAsyncScheduler):
    """Record each live request's actual allocated blocks at native boundaries."""

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Return the original schedule and report its physical allocation state."""
        result = super().schedule(throttle_prefills)
        self.observe_capacity("after_schedule")
        return result

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        """Return the original adoption result and report remaining live allocations."""
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.observe_capacity("after_adoption")
        return result

    def observe_capacity(self, boundary: str) -> None:
        """Export physical blocks and explicitly optimistic scheduler token counts.

        Args:
            boundary: Native schedule/adoption boundary label.
        Notes:
            Counts may include in-flight scheduled tokens. Offline full-prefill
            residency additionally requires the corresponding adoption records.
        """
        self.observation.emit(
            "capacity_state",
            boundary=boundary,
            requests=[
                dict(
                    request_id=identity,
                    status=request.status.name,
                    prompt_tokens=request.num_prompt_tokens,
                    scheduler_computed_tokens=request.num_computed_tokens,
                    output_placeholders=request.num_output_placeholders,
                    block_ids=self.kv_cache_manager.get_block_ids(identity),
                )
                for identity, request in self.requests.items()
            ],
        )
