# SPDX-License-Identifier: Apache-2.0
"""Deferred MP replies after actual CUDA and CXLMemSim completion.

The affinity worker only submits ordered native copies. This separate observer
polls per-operation events; it never synchronizes the device or blocks the next
submission behind a modeled delay. A pending reply retains vLLM's block hold.
"""

# Standard
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable
import threading
import time
from vllm.v1.kv_wait_trace import emit as wait_emit

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import L1BackendType, ObjectKey
from lmcache.v1.distributed.memory_manager.hybrid_l1_memory_manager import (
    HybridL1MemoryManager,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.host_phase import observed_mp_call


@dataclass
class HybridAccess:
    """One actual host-buffer access, plus optional modeled-operation identity."""

    metadata: dict[str, Any]
    token: int | None = None
    reported: bool = False
    done: bool = False


@dataclass
class PendingTransfer:
    """Strong references and response ownership until all completion gates pass."""

    event: Any
    context: Any
    accesses: list[HybridAccess]
    success: bool
    finalize: Callable[[bool], tuple[bytes, bool]]
    transfer_key: str
    deadline: float
    future: Future = field(default_factory=Future)
    cuda_observed_ns: int | None = None
    error: str | None = None


class HybridCompletionLoop:
    """One completion observer per MP server, independent of affinity submission."""

    def __init__(self, memory: HybridL1MemoryManager) -> None:
        self.memory = memory
        self.condition = threading.Condition()
        self.pending: list[PendingTransfer] = []
        self.closing = False
        self.protocol_failure: str | None = None
        self.completed = 0
        self.failed = 0
        self.peak_pending = 0
        self.failure_injected = False
        self.thread = threading.Thread(
            target=self._run, name="hybrid-completion", daemon=True
        )
        self.thread.start()

    @observed_mp_call("model_begin_objects")
    def begin_objects(
        self,
        accesses: list[HybridAccess],
        objects: list[MemoryObj | None],
        keys: list[ObjectKey],
        direction: str,
        transfer_key: str,
        block_ids: list[list[int]],
        chunk_tokens: int,
        skip_tokens: int = 0,
    ) -> None:
        """Start model branches before native DMA and retain partial submissions.

        Args:
            accesses: Caller-owned list extended after each successful begin.
            objects: Actual reserved buffers in logical chunk order.
            keys: Corresponding immutable version identities.
            direction: ``store`` or ``load``.
            transfer_key: MP operation identity.
            block_ids: Actual engine block lists for correlation.
            chunk_tokens: Complete tokens per host object.
            skip_tokens: HBM prefix excluded from LOAD destinations.
        Raises:
            RuntimeError: If the modeled backend failed or rejects a begin.
        """
        if self.protocol_failure:
            raise RuntimeError(
                "hybrid modeled backend failed: " + self.protocol_failure
            )
        for index, (obj, key) in enumerate(zip(objects, keys, strict=True)):
            if obj is None or (
                direction == "load" and (index + 1) * chunk_tokens <= skip_tokens
            ):
                continue
            medium = self.memory.get_backend_type(obj)
            offset = (
                obj.data_ptr - self.memory.mapping.address
                if medium in (L1BackendType.CXL_SIM, L1BackendType.CXL_DAX)
                else obj.meta.address
            )
            skipped = (
                max(0, min(chunk_tokens, skip_tokens - index * chunk_tokens))
                if direction == "load"
                else 0
            )
            metadata = {
                "transfer_key": transfer_key,
                "object_key": key,
                "chunk_index": index,
                "direction": direction,
                "medium": medium.value,
                "host_ptr": obj.data_ptr,
                "slot_offset": offset,
                "physical_bytes": obj.meta.phy_size,
                "actual_host_dma_bytes": obj.get_size(),
                "effective_destination_bytes": obj.get_size()
                * (chunk_tokens - skipped)
                // chunk_tokens,
                "block_ids": block_ids,
                "demand_ns": time.monotonic_ns(),
                "source": "hbm" if direction == "store" else medium.value,
                "destination": medium.value if direction == "store" else "hbm",
                "payload_staging_in_dram": False,
                "dma_absolute_start_ns": None,
                "dma_absolute_end_ns": None,
                "dma_null_reason": "event query observes completion only",
            }
            access = HybridAccess(metadata)
            if medium == L1BackendType.CXL_SIM:
                begin = self.memory.client.begin(direction, offset, obj.get_size())
                access.token = begin.access_token
                metadata["modeled_token"] = begin.access_token
                metadata["model_begin_observed_ns"] = time.monotonic_ns()
            if medium == L1BackendType.CXL_DAX:
                descriptor = self.memory.mapping.descriptor
                metadata.update(
                    dax_device=descriptor.device,
                    dax_device_offset=descriptor.offset_bytes + offset,
                    dax_target_node=descriptor.target_node,
                    cuda_host_registered=self.memory.registered,
                )
            accesses.append(access)
            self._emit(
                EventType.HYBRID_ACCESS, transfer_key, metadata | {"phase": "begin"}
            )

    def claim_failure(self, direction: str) -> bool:
        """Consume the explicitly configured one-shot partial-copy test hook."""
        if self.memory.config.diagnostic_failure != direction + "_partial":
            return False
        with self.condition:
            if self.failure_injected:
                return False
            self.failure_injected = True
            return True

    def defer(
        self,
        event: Any,
        context: Any,
        accesses: list[HybridAccess],
        success: bool,
        finalize: Callable[[bool], tuple[bytes, bool]],
        transfer_key: str,
    ) -> Future[tuple[bytes, bool]]:
        """Retain submitted resources until the CUDA and modeled gates finish.

        Args:
            event: Queryable device fence recorded after all submitted work.
            context: Engine/device owner retained until finalization.
            accesses: Model accesses begun for these actual host buffers.
            success: False if submission partly failed; its fence must still drain.
            finalize: Called once after terminal completion, with success status.
                It publishes or aborts writes, releases owned reservations, and
                returns the wire event handle and status. It must not wait for
                additional GPU work. Exceptions fail the reply and retire backend.
            transfer_key: Stable operation identity for emitted events.
        Returns:
            A running Future that cannot cancel already-submitted DMA ownership.
        Raises:
            RuntimeError: If shutdown already began. The rejected caller still
                owns the submitted resources and must drain them before release.
        """
        job = PendingTransfer(
            event,
            context,
            accesses,
            success,
            finalize,
            transfer_key,
            time.monotonic() + self.memory.config.completion_timeout_s,
        )
        # Cancellation of demand cannot cancel the already-submitted DMA fence.
        job.future.set_running_or_notify_cancel()
        with self.condition:
            if self.closing:
                raise RuntimeError("cannot submit after hybrid drain began")
            self.pending.append(job)
            self.peak_pending = max(self.peak_pending, len(self.pending))
            self.condition.notify_all()
        self._emit(
            EventType.HYBRID_COMPLETION,
            transfer_key,
            {"phase": "submitted", "cpu_submitted_ns": time.monotonic_ns()},
        )
        return job.future

    def report_status(self) -> dict[str, Any]:
        """Return pending replies separately from retained-cache occupation."""
        with self.condition:
            return {
                "pending": len(self.pending),
                "peak_pending": self.peak_pending,
                "completed": self.completed,
                "failed": self.failed,
                "protocol_failure": self.protocol_failure,
                "failure_injected": self.failure_injected,
            }

    def close(self) -> None:
        """Drain before allocator/context teardown; retain resources on timeout.

        Raises:
            RuntimeError: If the completion deadline expires with live transfers.
                The caller must retain the allocator and IPC contexts.
        """
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=self.memory.config.completion_timeout_s + 3)
        if self.thread.is_alive():
            raise RuntimeError("hybrid DMA did not drain; protected resources retained")

    def drain_context(self, context: Any) -> None:
        """Fence one engine's outstanding copies before closing its IPC mappings.

        Args:
            context: Exact engine context object whose submitted jobs must finish.
        Raises:
            RuntimeError: If the configured drain deadline expires. The caller
                must retain the context and protected buffers in that case.
        """
        deadline = time.monotonic() + self.memory.config.completion_timeout_s + 3
        with self.condition:
            while any(job.context is context for job in self.pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "engine IPC mappings still have active transfers"
                    )
                self.condition.wait(timeout=remaining)

    def _emit(self, event_type: EventType, key: str, metadata: dict[str, Any]) -> None:
        get_event_bus().publish(
            Event(
                event_type=event_type,
                session_id=key.rsplit("#", 1)[0],
                metadata={"transfer_key": key, **metadata},
            )
        )

    def _advance(self, job: PendingTransfer) -> bool:
        if job.cuda_observed_ns is None:
            with torch_dev.device(job.context.device):
                try:
                    ready = job.event.query()
                except RuntimeError as error:
                    # Query failure is not evidence that DMA stopped touching
                    # memory. Retain protections until a real terminal query.
                    job.success = False
                    job.error = repr(error)
                    self.protocol_failure = "CUDA context failed: " + repr(error)
                    return False
            if not ready:
                if time.monotonic() > job.deadline:
                    job.success = False
                    job.error = "CUDA timeout; waiting for its actual terminal state"
                return False
            job.cuda_observed_ns = time.monotonic_ns()
            self._emit(
                EventType.HYBRID_COMPLETION,
                job.transfer_key,
                {"phase": "cuda_observed", "observed_ns": job.cuda_observed_ns},
            )
        for access in job.accesses:
            if access.done:
                continue
            if access.token is None:
                access.done = True
                self._emit(
                    EventType.HYBRID_ACCESS,
                    job.transfer_key,
                    access.metadata
                    | {
                        "phase": "complete",
                        "cuda_observed_ns": job.cuda_observed_ns,
                        "logical_observed_ns": time.monotonic_ns(),
                        "modeled_queue_ns": None,
                        "modeled_service_ns": None,
                        "modeled_complete_ns": None,
                        "state": 3 if job.success else 5,
                    },
                )
                continue
            try:
                if not access.reported:
                    access.reported = True
                    self.memory.client.data_complete(access.token, job.success)
                result = self.memory.client.poll(access.token)
                if result.state not in (3, 4, 5):
                    if time.monotonic() > job.deadline:
                        self.memory.client.cancel(
                            access.token, "model completion timeout after CUDA drain"
                        )
                    continue
                access.done = True
                if result.state != 3:
                    job.success = False
                    job.error = f"model terminal state {result.state}"
                self._emit(
                    EventType.HYBRID_ACCESS,
                    job.transfer_key,
                    access.metadata
                    | {
                        "phase": "complete",
                        "cuda_observed_ns": job.cuda_observed_ns,
                        "modeled_queue_ns": result.queue_ns,
                        "modeled_service_ns": result.service_ns,
                        "modeled_complete_ns": result.modeled_complete_ns,
                        "logical_observed_ns": time.monotonic_ns(),
                        "state": result.state,
                    },
                )
            except Exception as error:
                # Real CUDA is terminal, but no usable model completion exists.
                # Fail the reply and retire this backend instead of publishing it.
                job.success = False
                job.error = repr(error)
                self.protocol_failure = repr(error)
                self.memory.client.abandon_drained(access.token)
                access.done = True
        return all(access.done for access in job.accesses)

    def _run(self) -> None:
        while True:
            with self.condition:
                if not self.pending:
                    if self.closing:
                        return
                    self.condition.wait()
                jobs = deque(self.pending)
            while jobs:
                job = jobs.popleft()
                try:
                    if not self._advance(job):
                        continue
                    response = job.finalize(job.success)
                    self._emit(
                        EventType.HYBRID_COMPLETION,
                        job.transfer_key,
                        {
                            "phase": "reply_ready",
                            "success": job.success,
                            "error": job.error,
                            "reply_ready_ns": time.monotonic_ns(),
                        },
                    )
                    if any(a.metadata.get("direction") == "load" for a in job.accesses):
                        wait_emit(
                            "restore_terminal_ready",
                            job.transfer_key.rsplit("#", 1)[0],
                            transfer_key=job.transfer_key,
                            success=job.success,
                        )
                    job.future.set_result(response)
                except Exception as error:
                    # Finalizer bugs must remain visible; never report success.
                    self.protocol_failure = repr(error)
                    job.future.set_exception(error)
                with self.condition:
                    self.pending.remove(job)
                    self.completed += 1
                    self.failed += int(
                        not job.success or job.future.exception() is not None
                    )
                    # Do not retain the last completed context while this
                    # thread sleeps. Release snapshot/local ownership before
                    # drain_context can observe an empty pending set and ACK
                    # UNREGISTER; otherwise imported IPC tensors outlive it.
                    del job
                    self.condition.notify_all()
            with self.condition:
                if self.pending:
                    self.condition.wait(timeout=self.memory.config.poll_interval_s)
