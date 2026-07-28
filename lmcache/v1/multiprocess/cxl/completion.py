# SPDX-License-Identifier: Apache-2.0
"""Composite completion for concurrent CUDA and modeled access branches."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Literal
import threading
import time

# Local
from .contracts import CompositeCompletion, DataCompletion
from .bounded import BoundedSet
from .model_client import (
    CXLModelClient,
    ModelCompletion,
    ModeledAccessRequest,
    RegisteredModelRegion,
)


class NoopModelClient:
    """Mark modeled completion as unnecessary for Gates A-C."""

    def join(self, data_completion: DataCompletion) -> CompositeCompletion:
        """Project a CUDA completion onto the composite public contract.

        Args:
            data_completion: Physical CUDA completion to expose.

        Returns:
            A composite completion with no modeled queue or service delay.
        """
        success = data_completion.status == "ok"
        return CompositeCompletion(
            op_id=data_completion.op_id,
            cuda_status=data_completion.status,
            modeled_status="not_required",
            cuda_elapsed_ns=data_completion.elapsed_ns,
            modeled_queue_ns=None,
            modeled_service_ns=None,
            cuda_complete_ns=data_completion.complete_ns,
            modeled_complete_ns=None,
            effective_complete_ns=(data_completion.complete_ns if success else None),
            effective_elapsed_ns=data_completion.elapsed_ns if success else None,
            error=data_completion.error,
        )


def compose_completion(
    start_ns: int,
    data_completion: DataCompletion,
    model_completion: ModelCompletion,
) -> CompositeCompletion:
    """Compose branch results without summing their durations.

    Args:
        start_ns: Shared logical operation start.
        data_completion: Local CUDA terminal evidence.
        model_completion: Unmodified modeled-service terminal evidence.

    Returns:
        A terminal composite whose successful completion is the later branch.
    """
    if start_ns <= 0:
        raise ValueError("start_ns must be positive")
    if data_completion.op_id != model_completion.op_id:
        raise ValueError("completion op IDs do not match")
    success = data_completion.status == "ok" and model_completion.status == "ok"
    effective_complete_ns = None
    effective_elapsed_ns = None
    if success:
        if data_completion.complete_ns is None:
            raise ValueError("successful CUDA completion has no timestamp")
        effective_complete_ns = max(
            data_completion.complete_ns, model_completion.modeled_complete_ns
        )
        if effective_complete_ns < start_ns:
            raise ValueError("completion predates logical operation start")
        effective_elapsed_ns = effective_complete_ns - start_ns
    modeled_status: Literal["pending", "ok", "error", "cancelled"]
    if model_completion.status in ("pending", "modeled_complete", "data_complete"):
        modeled_status = "pending"
    elif model_completion.status == "ok":
        modeled_status = "ok"
    elif model_completion.status == "error":
        modeled_status = "error"
    else:
        modeled_status = "cancelled"
    error = data_completion.error or model_completion.error
    if modeled_status == "cancelled" and error is None:
        error = "modeled access was cancelled"
    return CompositeCompletion(
        op_id=data_completion.op_id,
        cuda_status=data_completion.status,
        modeled_status=modeled_status,
        cuda_elapsed_ns=data_completion.elapsed_ns,
        modeled_queue_ns=model_completion.queue_ns,
        modeled_service_ns=model_completion.service_ns,
        cuda_complete_ns=data_completion.complete_ns,
        modeled_complete_ns=model_completion.modeled_complete_ns,
        effective_complete_ns=effective_complete_ns,
        effective_elapsed_ns=effective_elapsed_ns,
        error=error,
    )


class ModeledCompletionCoordinator:
    """Reserve model service, launch CUDA, then await both branches."""

    def __init__(
        self,
        client: CXLModelClient,
        region: RegisteredModelRegion,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._client = client
        self._region = region
        self._clock_ns = clock_ns
        self._cancelled: BoundedSet[str] = BoundedSet()
        self._states: dict[str, Literal["pending", "active"]] = {}
        self._lock = threading.RLock()

    def run(
        self,
        *,
        op_id: str,
        instance_id: int,
        direction: Literal["store", "retrieve"],
        offset: int,
        bytes: int,
        launch: Callable[[], DataCompletion],
    ) -> CompositeCompletion:
        """Run one logical operation with a shared start timestamp.

        Args:
            op_id: Unique operation identity.
            instance_id: Engine client identity.
            direction: CXL data direction.
            offset: Region-relative byte offset.
            bytes: Positive transfer size.
            launch: Immediate CUDA launch callback.

        Returns:
            Terminal composite completion.
        """
        with self._lock:
            if op_id in self._cancelled:
                raise RuntimeError(f"modeled operation {op_id} was cancelled")
            if op_id in self._states:
                raise RuntimeError(f"modeled operation {op_id} is already active")
            self._states[op_id] = "pending"
        start_ns = self._clock_ns()
        try:
            self._client.begin_access(
                ModeledAccessRequest(
                    op_id=op_id,
                    client_id=instance_id,
                    direction=direction,
                    server_region_token=self._region.server_region_token,
                    offset=offset,
                    bytes=bytes,
                    start_ns=start_ns,
                )
            )
        except BaseException:
            with self._lock:
                self._states.pop(op_id, None)
            raise
        with self._lock:
            cancelled = op_id in self._cancelled
            if not cancelled:
                self._states[op_id] = "active"
        if cancelled:
            self._client.cancel(op_id, "cancelled while modeled access registered")
            with self._lock:
                self._states.pop(op_id, None)
            raise RuntimeError(f"modeled operation {op_id} was cancelled")
        try:
            try:
                data_completion = launch()
            except BaseException:
                self._client.cancel(op_id, "CUDA launch raised before terminal state")
                raise
            complete_ns = data_completion.complete_ns or self._clock_ns()
            cuda_status: Literal["ok", "error"] = (
                "ok" if data_completion.status == "ok" else "error"
            )
            self._client.data_complete(op_id, cuda_status, complete_ns)
            try:
                model_completion = self._client.await_completion(op_id)
            except BaseException:
                self._client.cancel(op_id, "modeled completion failed")
                raise
            return compose_completion(start_ns, data_completion, model_completion)
        finally:
            with self._lock:
                self._states.pop(op_id, None)

    def cancel(self, op_id: str, reason: str) -> None:
        """Cancel before launch or forward cancellation to an active model op."""
        if not op_id or not reason:
            raise ValueError("cancellation identity and reason are required")
        with self._lock:
            self._cancelled.add(op_id)
            active = self._states.get(op_id) == "active"
        if active:
            self._client.cancel(op_id, reason)

    def close(self) -> None:
        """Close the underlying modeled client."""
        self._client.close()
