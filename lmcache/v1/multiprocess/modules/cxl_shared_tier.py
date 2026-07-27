# SPDX-License-Identifier: Apache-2.0
"""Gate B orchestration for immutable GPU-visible CXL proxy objects."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Literal, Protocol, Sequence
import json
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import EncodedObjectKey, ObjectKey
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.completion import (
    ModeledCompletionCoordinator,
    NoopModelClient,
)
from lmcache.v1.multiprocess.cxl.contracts import (
    CompositeCompletion,
    ExtentDescriptor,
    TransferPlan,
)
from lmcache.v1.multiprocess.cxl.cuda_executor import (
    LMCacheIPCTransferExecutor,
    RegisteredRegionView,
)
from lmcache.v1.multiprocess.cxl.data_plane_adapter import (
    VLLMDataPlaneAdapter,
    VLLMTransferRequest,
)
from lmcache.v1.multiprocess.cxl.layout import layout_fingerprint
from lmcache.v1.multiprocess.cxl.model_client import CXLMemSimModelClient
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import (
    CXLMemSimShmRegionProvider,
    PosixShmRegionProvider,
    RegionHandle,
    RegionProvider,
)
from lmcache.v1.multiprocess.cxl.residency import (
    ReadLease,
    Residency,
    SingleResidencyDirectory,
)


logger = init_logger(__name__)


CXLOperationState = Literal[
    "ready",
    "lease_acquired",
    "ok",
    "miss",
    "error",
    "cancelled",
]


@dataclass(frozen=True)
class CXLOperationEvent:
    """Pointer-free observable state transition for one CXL chunk operation."""

    timestamp_ns: int
    op_id: str
    instance_id: int
    object_key: EncodedObjectKey
    direction: Literal["store", "retrieve"]
    path: Literal["cxl_direct"]
    extent: ExtentDescriptor | None
    generation: int | None
    lease_id: str | None
    layout_fingerprint: str | None
    payload_checksum: str | None
    cuda_elapsed_ns: int | None
    modeled_queue_ns: int | None
    modeled_service_ns: int | None
    effective_elapsed_ns: int | None
    state: CXLOperationState
    terminal: bool

    def to_primitive(self) -> dict[str, Any]:
        """Return a JSON-safe record without process-local data.

        Returns:
            Primitive operation fields suitable for JSONL telemetry.
        """
        return {
            "timestamp_ns": self.timestamp_ns,
            "op_id": self.op_id,
            "instance_id": self.instance_id,
            "object_key": asdict(self.object_key),
            "direction": self.direction,
            "path": self.path,
            "extent": None if self.extent is None else asdict(self.extent),
            "generation": self.generation,
            "lease_id": self.lease_id,
            "layout_fingerprint": self.layout_fingerprint,
            "payload_checksum": self.payload_checksum,
            "cuda_elapsed_ns": self.cuda_elapsed_ns,
            "modeled_queue_ns": self.modeled_queue_ns,
            "modeled_service_ns": self.modeled_service_ns,
            "effective_elapsed_ns": self.effective_elapsed_ns,
            "state": self.state,
            "terminal": self.terminal,
        }


class CXLOperationSink(Protocol):
    """Boundary for durable or in-memory CXL operation telemetry."""

    def record(self, event: CXLOperationEvent) -> None:
        """Record one immutable operation event.

        Args:
            event: Pointer-free event to persist or expose.
        """
        ...


class LoggingCXLOperationSink:
    """Emit CXL operation events as structured JSON through LMCache logging."""

    def record(self, event: CXLOperationEvent) -> None:
        """Log one operation event.

        Args:
            event: Pointer-free event to serialize.
        """
        logger.info(
            "cxl_operation=%s", json.dumps(event.to_primitive(), sort_keys=True)
        )


class InMemoryCXLOperationSink:
    """Retain a bounded snapshot of operation events for proof and testing."""

    def __init__(self, capacity: int = 4096) -> None:
        """Create a bounded thread-safe sink.

        Args:
            capacity: Maximum number of newest events to retain.

        Raises:
            ValueError: If capacity is not positive.
        """
        if capacity <= 0:
            raise ValueError("operation sink capacity must be positive")
        self._events: deque[CXLOperationEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, event: CXLOperationEvent) -> None:
        """Append one event, dropping the oldest record at capacity.

        Args:
            event: Immutable operation event to retain.
        """
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[CXLOperationEvent, ...]:
        """Return retained events in recording order.

        Returns:
            Immutable point-in-time operation event sequence.
        """
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True)
class CXLTransferResult:
    """Observable terminal result of one shared-tier request."""

    status: Literal["ok", "miss", "error", "cancelled"]
    path: Literal["cxl_direct"]
    completions: tuple[CompositeCompletion, ...]
    payload_bytes: int
    dram_allocated_bytes_delta: int
    error: str | None

    def __post_init__(self) -> None:
        """Validate terminal result fields.

        Raises:
            ValueError: If status, path, DRAM accounting, or error fields are
                inconsistent.
        """
        if self.status not in ("ok", "miss", "error", "cancelled"):
            raise ValueError("invalid CXL transfer status")
        if self.path != "cxl_direct":
            raise ValueError("shared-tier path must be cxl_direct")
        if self.dram_allocated_bytes_delta != 0:
            raise ValueError("CXL direct transfer cannot allocate DRAM staging")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.status == "error" and not self.error:
            raise ValueError("error result requires a diagnostic")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error result cannot contain a diagnostic")


@dataclass(frozen=True)
class _PendingStore:
    residency_id: str
    primary_key: ObjectKey
    aliases: tuple[ObjectKey, ...]


@dataclass
class _ActiveOperation:
    instance_id: int
    cancelled: bool = False
    modeled_op_ids: set[str] = field(default_factory=set)


class _OperationCancelled(RuntimeError):
    pass


class CXLSharedTierModule:
    """Coordinate immutable CXL STORE and RETRIEVE for one LMCache server."""

    _LEASE_TTL_NS = 30_000_000_000

    def __init__(
        self,
        config: CXLSharedTierConfig,
        provider: RegionProvider,
        handle: RegionHandle,
        *,
        native_ops: Any | None,
        runtime: Any | None,
        clock_ns: Callable[[], int],
        operation_sink: CXLOperationSink,
        modeled_coordinator: ModeledCompletionCoordinator | None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._handle = handle
        self._native_ops = native_ops
        self._runtime = runtime
        self._clock_ns = clock_ns
        self._operation_sink = operation_sink
        self._region_manager: CXLRegionManager | None = None
        self._region_view: RegisteredRegionView | None = None
        self._executor: LMCacheIPCTransferExecutor | None = None
        self._directory: SingleResidencyDirectory | None = None
        self._layout: Any | None = None
        self._adapters: dict[int, VLLMDataPlaneAdapter] = {}
        self._primary_by_alias: dict[ObjectKey, ObjectKey] = {}
        self._aliases_by_primary: dict[ObjectKey, tuple[ObjectKey, ...]] = {}
        self._pending_aliases: dict[ObjectKey, str] = {}
        self._model_client = NoopModelClient()
        self._modeled_coordinator = modeled_coordinator
        self._active_operations: dict[str, _ActiveOperation] = {}
        self._closed = False
        self._closing = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    @classmethod
    def open(
        cls,
        config: CXLSharedTierConfig,
        *,
        native_ops: Any | None = None,
        runtime: Any | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        operation_sink: CXLOperationSink | None = None,
    ) -> CXLSharedTierModule:
        """Open and validate the configured proxy region without fallback.

        Args:
            config: Complete enabled shared-tier configuration.
            native_ops: Optional replacement for the CUDA/native boundary.
            runtime: Optional replacement for CUDA timing and execution.
            clock_ns: Monotonic clock used for completions and leases.
            operation_sink: Optional pointer-free operation telemetry sink.

        Returns:
            An open shared-tier module awaiting engine registration.

        Raises:
            ValueError: If the feature is disabled or provider is unsupported.
            OSError: If the configured POSIX SHM object cannot be opened.
            RuntimeError: If the region header or capacity is incompatible.
        """
        if not config.enabled:
            raise ValueError("CXL shared tier is disabled")
        if config.shm_name is None or config.capacity_bytes is None:
            raise ValueError("enabled CXL shared tier configuration is incomplete")
        provider: RegionProvider
        if config.provider == "posix_shm":
            provider = PosixShmRegionProvider(
                region_id=f"cxl-posix:{config.shm_name}",
                shm_name=config.shm_name,
                expected_capacity=config.capacity_bytes,
            )
        elif config.provider == "cxlmemsim_shm":
            provider = CXLMemSimShmRegionProvider(
                region_id=f"cxlmemsim:{config.shm_name}",
                shm_name=config.shm_name,
                expected_capacity=config.capacity_bytes,
            )
        else:
            raise ValueError("unsupported CXL shared-tier provider")
        model_client: CXLMemSimModelClient | None = None
        try:
            handle = provider.provision()
            coordinator = None
            if config.model_mode == "cxlmemsim":
                assert config.model_client_library is not None
                model_client = CXLMemSimModelClient.open(
                    library_path=config.model_client_library,
                    control_name=config.model_control_name,
                    timeout_ms=config.model_timeout_ms,
                    clock_ns=clock_ns,
                )
                modeled_region = model_client.register_region(handle)
                coordinator = ModeledCompletionCoordinator(
                    model_client, modeled_region, clock_ns=clock_ns
                )
            return cls(
                config,
                provider,
                handle,
                native_ops=native_ops,
                runtime=runtime,
                clock_ns=clock_ns,
                operation_sink=(
                    LoggingCXLOperationSink()
                    if operation_sink is None
                    else operation_sink
                ),
                modeled_coordinator=coordinator,
            )
        except BaseException:
            if model_client is not None:
                model_client.close()
            provider.close()
            raise

    def register_engine(
        self, instance_id: int, cache_context: Any, model_name: str
    ) -> None:
        """Register an imported engine context against the shared layout.

        Args:
            instance_id: Non-negative engine instance identifier.
            cache_context: Existing LMCache CUDA cache context.
            model_name: Model identity included in the packed fingerprint.

        Raises:
            RuntimeError: If the module is closed or registration fails.
            ValueError: If the context layout is incompatible.
        """
        with self._lock:
            self._require_open()
            adapter = VLLMDataPlaneAdapter(cache_context)
            layout = adapter.describe_layout(
                model_name, cache_context.lmcache_tokens_per_chunk
            )
            self._validate_attention_layout(cache_context)
            if layout.layout_id != self._config.layout_id:
                raise ValueError("engine layout ID does not match CXL config")
            if self._executor is None:
                fingerprint = layout_fingerprint(layout)
                manager = CXLRegionManager(
                    self._handle,
                    layout_id=layout.layout_id,
                    layout_fingerprint=fingerprint,
                )
                view = RegisteredRegionView.open(
                    self._handle,
                    descriptor_validator=manager.validate_descriptor,
                    native_ops=self._native_ops,
                    expected_layout_fingerprint=fingerprint,
                )
                self._region_manager = manager
                self._region_view = view
                self._directory = SingleResidencyDirectory(manager)
                self._executor = LMCacheIPCTransferExecutor(
                    view,
                    native_ops=self._native_ops,
                    runtime=self._runtime,
                    clock_ns=self._clock_ns,
                )
                self._layout = layout
            else:
                adapter.validate_context(cache_context, self._require_layout())
            self._require_executor().register_engine(instance_id, cache_context)
            self._adapters[instance_id] = adapter

    def unregister_engine(self, instance_id: int) -> None:
        """Drop one imported engine context without deleting READY objects.

        Args:
            instance_id: Previously registered engine identifier.
        """
        with self._condition:
            for operation in self._active_operations.values():
                if operation.instance_id == instance_id:
                    operation.cancelled = True
            self._condition.wait_for(
                lambda: all(
                    operation.instance_id != instance_id
                    for operation in self._active_operations.values()
                )
            )
            self._adapters.pop(instance_id, None)
            if self._executor is not None:
                self._executor.unregister_engine(instance_id)

    def cancel(self, op_id: str) -> bool:
        """Cancel an active request without releasing its context early.

        Args:
            op_id: Parent operation identifier supplied in the transfer request.

        Returns:
            True when an active operation was marked cancelled, otherwise False.
        """
        with self._condition:
            operation = self._active_operations.get(op_id)
            if operation is None:
                return False
            operation.cancelled = True
            if self._modeled_coordinator is not None:
                for modeled_op_id in tuple(operation.modeled_op_ids):
                    self._modeled_coordinator.cancel(
                        modeled_op_id, "parent operation was cancelled"
                    )
            return True

    def store(self, request: VLLMTransferRequest) -> CXLTransferResult:
        """Store independently allocated chunks and publish after CUDA success.

        Args:
            request: Complete engine transfer metadata for one STORE request.

        Returns:
            A terminal direct-path result. CUDA errors abort every unpublished
            residency and never fall back to DRAM staging.
        """
        payload_bytes = 0
        pending: list[_PendingStore] = []
        pending_aliases: list[ObjectKey] = []
        plans: list[TransferPlan] = []
        completions: list[CompositeCompletion] = []
        existing: list[tuple[VLLMTransferRequest, Residency, ObjectKey]] = []
        operation_started = False
        terminal_state: Literal["error", "cancelled"] | None = None
        try:
            with self._lock:
                adapter = self._adapter_for(request.instance_id)
                chunks = adapter.split_chunks(request)
                payload_bytes = sum(
                    adapter.packed_size_bytes(chunk) for chunk in chunks
                )
                pending_chunks: list[VLLMTransferRequest] = []
                for chunk in chunks:
                    existing_primary = self._existing_primary(chunk.object_keys)
                    if existing_primary is None:
                        pending_chunks.append(chunk)
                        continue
                    residency = self._require_directory().lookup_ready(existing_primary)
                    if residency is None or residency.descriptor is None:
                        raise RuntimeError("existing CXL residency is not READY")
                    existing.append((chunk, residency, existing_primary))
                new_aliases = [
                    alias for chunk in pending_chunks for alias in chunk.object_keys
                ]
                if len(new_aliases) != len(set(new_aliases)):
                    raise RuntimeError(
                        "ObjectKey alias appears in multiple new CXL chunks"
                    )
                if any(alias in self._pending_aliases for alias in new_aliases):
                    raise RuntimeError("ObjectKey alias has an active CXL STORE")
                for alias in new_aliases:
                    self._pending_aliases[alias] = request.op_id
                    pending_aliases.append(alias)

                for chunk in pending_chunks:
                    primary = chunk.object_keys[0]
                    directory = self._require_directory()
                    residency = directory.reserve_store(
                        primary,
                        length=adapter.packed_size_bytes(chunk),
                        alignment=self._config.alignment_bytes,
                    )
                    pending.append(
                        _PendingStore(
                            residency.residency_id, primary, chunk.object_keys
                        )
                    )
                    writing = directory.mark_writing(residency.residency_id)
                    if writing.descriptor is None:
                        raise RuntimeError("WRITING residency has no extent")
                    plans.append(adapter.build_store_plan(chunk, writing.descriptor))
                self._begin_operation(request.op_id, request.instance_id)
                operation_started = True

            for plan in plans:
                self._raise_if_cancelled(request.op_id)
                completion = self._execute_plan(request.op_id, plan)
                completions.append(completion)
                self._raise_if_cancelled(request.op_id)
                if completion.cuda_status != "ok":
                    raise RuntimeError(completion.error or "CUDA store failed")

            with self._lock:
                self._raise_if_cancelled(request.op_id)
                for chunk, residency, primary in existing:
                    self._record_event(
                        op_id=chunk.op_id,
                        instance_id=request.instance_id,
                        object_key=primary,
                        direction="store",
                        extent=residency.descriptor,
                        generation=residency.generation,
                        lease_id=None,
                        payload_checksum=request.payload_checksum_expected,
                        cuda_elapsed_ns=0,
                        state="ready",
                        terminal=True,
                    )
                for index, item in enumerate(pending):
                    ready_residency = self._require_directory().publish(
                        item.residency_id
                    )
                    self._publish_aliases(item.primary_key, item.aliases)
                    completion = completions[index]
                    self._record_event(
                        op_id=plans[index].op_id,
                        instance_id=request.instance_id,
                        object_key=item.primary_key,
                        direction="store",
                        extent=ready_residency.descriptor,
                        generation=ready_residency.generation,
                        lease_id=None,
                        payload_checksum=request.payload_checksum_expected,
                        cuda_elapsed_ns=completion.cuda_elapsed_ns,
                        state="ready",
                        terminal=True,
                        completion=completion,
                    )
            return self._result("ok", tuple(completions), payload_bytes=payload_bytes)
        except _OperationCancelled:
            terminal_state = "cancelled"
            return self._result(
                "cancelled", tuple(completions), payload_bytes=payload_bytes
            )
        except Exception as error:
            terminal_state = "error"
            diagnostic = str(error) or self._completion_error(completions)
            return self._result(
                "error",
                tuple(completions),
                diagnostic,
                payload_bytes=payload_bytes,
            )
        finally:
            with self._lock:
                if terminal_state is not None:
                    for index, item in enumerate(pending):
                        plan = plans[index]
                        terminal_completion = (
                            completions[index] if index < len(completions) else None
                        )
                        self._record_event(
                            op_id=plan.op_id,
                            instance_id=request.instance_id,
                            object_key=item.primary_key,
                            direction="store",
                            extent=plan.extent,
                            generation=plan.extent.generation,
                            lease_id=None,
                            payload_checksum=request.payload_checksum_expected,
                            cuda_elapsed_ns=(
                                None
                                if terminal_completion is None
                                else terminal_completion.cuda_elapsed_ns
                            ),
                            state=terminal_state,
                            terminal=True,
                            completion=terminal_completion,
                        )
                for item in pending:
                    if self._require_directory().lookup_ready(item.primary_key) is None:
                        try:
                            self._require_directory().abort(
                                item.residency_id,
                                "CXL STORE did not reach READY",
                            )
                        except (KeyError, RuntimeError):
                            pass
                for alias in pending_aliases:
                    if self._pending_aliases.get(alias) == request.op_id:
                        self._pending_aliases.pop(alias, None)
            if operation_started:
                self._finish_operation(request.op_id)

    def retrieve(self, request: VLLMTransferRequest) -> CXLTransferResult:
        """Restore READY chunks directly into HBM under read leases.

        Args:
            request: Complete engine transfer metadata for one RETRIEVE.

        Returns:
            OK, MISS, or ERROR without allocating a DRAM staging object.
        """
        leases: list[ReadLease] = []
        ready: list[tuple[VLLMTransferRequest, Residency, ObjectKey]] = []
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            all_chunks = adapter.split_chunks(request)
            chunks = self._retrievable_chunks(all_chunks, request)
            payload_bytes = sum(adapter.packed_size_bytes(chunk) for chunk in chunks)
            for chunk in chunks:
                primary = self._primary_for_chunk(chunk.object_keys)
                if primary is None:
                    self._record_event(
                        op_id=chunk.op_id,
                        instance_id=request.instance_id,
                        object_key=chunk.object_keys[0],
                        direction="retrieve",
                        extent=None,
                        generation=None,
                        lease_id=None,
                        payload_checksum=request.payload_checksum_expected,
                        cuda_elapsed_ns=None,
                        state="miss",
                        terminal=True,
                    )
                    return self._result("miss", payload_bytes=payload_bytes)
                residency = self._require_directory().lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    self._record_event(
                        op_id=chunk.op_id,
                        instance_id=request.instance_id,
                        object_key=primary,
                        direction="retrieve",
                        extent=None,
                        generation=None,
                        lease_id=None,
                        payload_checksum=request.payload_checksum_expected,
                        cuda_elapsed_ns=None,
                        state="miss",
                        terminal=True,
                    )
                    return self._result("miss", payload_bytes=payload_bytes)
                ready.append((chunk, residency, primary))
            for _, residency, primary in ready:
                lease = self._require_directory().acquire_read(
                    primary,
                    ttl_ns=self._LEASE_TTL_NS,
                    now_ns=self._clock_ns(),
                )
                leases.append(lease)
                if lease.generation != residency.generation:
                    raise RuntimeError("read lease generation changed")
            self._begin_operation(request.op_id, request.instance_id)
            for (chunk, residency, _), lease in zip(ready, leases, strict=True):
                self._record_event(
                    op_id=chunk.op_id,
                    instance_id=request.instance_id,
                    object_key=chunk.object_keys[0],
                    direction="retrieve",
                    extent=residency.descriptor,
                    generation=residency.generation,
                    lease_id=lease.lease_id,
                    payload_checksum=request.payload_checksum_expected,
                    cuda_elapsed_ns=None,
                    state="lease_acquired",
                    terminal=False,
                )

        completions: list[CompositeCompletion] = []
        terminal_indices: set[int] = set()
        terminal_state: Literal["error", "cancelled"] | None = None
        try:
            for index, (chunk, residency, _) in enumerate(ready):
                self._raise_if_cancelled(request.op_id)
                self._require_live_lease(leases[index])
                descriptor = residency.descriptor
                if descriptor is None:
                    raise RuntimeError("READY residency has no extent")
                plan = adapter.build_retrieve_plan(chunk, descriptor)
                completion = self._execute_plan(request.op_id, plan)
                completions.append(completion)
                self._raise_if_cancelled(request.op_id)
                self._require_live_lease(leases[index])
                if completion.cuda_status != "ok":
                    raise RuntimeError(completion.error or "CUDA retrieve failed")
                self._record_event(
                    op_id=chunk.op_id,
                    instance_id=request.instance_id,
                    object_key=chunk.object_keys[0],
                    direction="retrieve",
                    extent=residency.descriptor,
                    generation=residency.generation,
                    lease_id=leases[index].lease_id,
                    payload_checksum=request.payload_checksum_expected,
                    cuda_elapsed_ns=completion.cuda_elapsed_ns,
                    state="ok",
                    terminal=True,
                    completion=completion,
                )
                terminal_indices.add(index)
            return self._result("ok", tuple(completions), payload_bytes=payload_bytes)
        except _OperationCancelled:
            terminal_state = "cancelled"
            return self._result(
                "cancelled", tuple(completions), payload_bytes=payload_bytes
            )
        except Exception as error:
            terminal_state = "error"
            diagnostic = str(error) or self._completion_error(completions)
            return self._result(
                "error",
                tuple(completions),
                diagnostic,
                payload_bytes=payload_bytes,
            )
        finally:
            try:
                if terminal_state is not None:
                    for index, (chunk, residency, _) in enumerate(ready):
                        if index in terminal_indices:
                            continue
                        terminal_completion = (
                            completions[index] if index < len(completions) else None
                        )
                        self._record_event(
                            op_id=chunk.op_id,
                            instance_id=request.instance_id,
                            object_key=chunk.object_keys[0],
                            direction="retrieve",
                            extent=residency.descriptor,
                            generation=residency.generation,
                            lease_id=leases[index].lease_id,
                            payload_checksum=request.payload_checksum_expected,
                            cuda_elapsed_ns=(
                                None
                                if terminal_completion is None
                                else terminal_completion.cuda_elapsed_ns
                            ),
                            state=terminal_state,
                            terminal=True,
                            completion=terminal_completion,
                        )
                for lease in leases:
                    self._require_directory().release_read(
                        lease.lease_id, now_ns=self._clock_ns()
                    )
            finally:
                self._finish_operation(request.op_id)

    def contains(self, request: VLLMTransferRequest) -> bool:
        """Return whether every executable chunk has a READY CXL residency.

        Args:
            request: Candidate RETRIEVE metadata.

        Returns:
            True only when a direct plan can be built for every required chunk.

        Raises:
            KeyError: If the engine instance is not registered.
            ValueError: If request chunk metadata is malformed.
        """
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            chunks = self._retrievable_chunks(adapter.split_chunks(request), request)
            for chunk in chunks:
                primary = self._primary_for_chunk(chunk.object_keys)
                if primary is None:
                    return False
                residency = self._require_directory().lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    return False
            return True

    def lookup_ready(self, object_key: ObjectKey) -> Residency | None:
        """Return the READY residency addressed by an object or alias.

        Args:
            object_key: Logical packed object identity.

        Returns:
            The immutable READY residency, otherwise None.
        """
        with self._lock:
            primary = self._primary_by_alias.get(object_key, object_key)
            return self._require_directory().lookup_ready(primary)

    def count_ready_prefix(
        self, object_keys: Sequence[ObjectKey], keys_per_chunk: int
    ) -> int:
        """Count contiguous complete chunks with one READY packed residency.

        Args:
            object_keys: Chunk-major object keys from the scheduler lookup.
            keys_per_chunk: Number of ranks times object groups in each chunk.

        Returns:
            Number of complete READY chunks before the first miss.

        Raises:
            ValueError: If the chunk geometry is invalid.
        """
        if keys_per_chunk <= 0:
            raise ValueError("keys_per_chunk must be positive")
        if len(object_keys) % keys_per_chunk != 0:
            raise ValueError("object keys do not contain complete chunks")
        with self._lock:
            if self._directory is None:
                return 0
            found = 0
            for start in range(0, len(object_keys), keys_per_chunk):
                aliases = tuple(object_keys[start : start + keys_per_chunk])
                primary = self._primary_for_chunk(aliases)
                if primary is None:
                    break
                residency = self._directory.lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    break
                found += 1
            return found

    def evict(self, object_key: ObjectKey) -> None:
        """Evict a READY chunk after active readers release their leases.

        Args:
            object_key: Primary object key or any alias of the packed chunk.

        Raises:
            KeyError: If no residency is known for the object.
            RuntimeError: If the residency is not READY.
        """
        with self._lock:
            primary = self._primary_by_alias.get(object_key, object_key)
            self._require_directory().evict(primary)
            aliases = self._aliases_by_primary.pop(primary, ())
            for alias in aliases:
                self._primary_by_alias.pop(alias, None)

    def close(self) -> None:
        """Release engine registrations, CUDA mapping, and provider handles."""
        with self._condition:
            if self._closed:
                return
            self._closing = True
            for operation in self._active_operations.values():
                operation.cancelled = True
            self._condition.wait_for(lambda: not self._active_operations)
            for instance_id in tuple(self._adapters):
                self._require_executor().unregister_engine(instance_id)
            self._adapters.clear()
            if self._region_view is not None:
                self._region_view.close()
            if self._modeled_coordinator is not None:
                self._modeled_coordinator.close()
            self._provider.close()
            self._closed = True

    @staticmethod
    def _result(
        status: Literal["ok", "miss", "error", "cancelled"],
        completions: tuple[CompositeCompletion, ...] = (),
        error: str | None = None,
        *,
        payload_bytes: int = 0,
    ) -> CXLTransferResult:
        return CXLTransferResult(
            status=status,
            path="cxl_direct",
            completions=completions,
            payload_bytes=payload_bytes,
            dram_allocated_bytes_delta=0,
            error=error,
        )

    @staticmethod
    def _completion_error(completions: list[CompositeCompletion]) -> str | None:
        if completions and completions[-1].cuda_status == "error":
            return "CUDA transfer failed"
        return None

    def _adapter_for(self, instance_id: int) -> VLLMDataPlaneAdapter:
        self._require_open()
        try:
            return self._adapters[instance_id]
        except KeyError as error:
            raise KeyError("engine instance is not registered") from error

    def _record_event(
        self,
        *,
        op_id: str,
        instance_id: int,
        object_key: ObjectKey,
        direction: Literal["store", "retrieve"],
        extent: ExtentDescriptor | None,
        generation: int | None,
        lease_id: str | None,
        payload_checksum: str | None,
        cuda_elapsed_ns: int | None,
        state: CXLOperationState,
        terminal: bool,
        completion: CompositeCompletion | None = None,
    ) -> None:
        self._operation_sink.record(
            CXLOperationEvent(
                timestamp_ns=self._clock_ns(),
                op_id=op_id,
                instance_id=instance_id,
                object_key=object_key.to_encoded_object_key(),
                direction=direction,
                path="cxl_direct",
                extent=extent,
                generation=generation,
                lease_id=lease_id,
                layout_fingerprint=(
                    None if extent is None else extent.layout_fingerprint
                ),
                payload_checksum=payload_checksum,
                cuda_elapsed_ns=cuda_elapsed_ns,
                modeled_queue_ns=(
                    None if completion is None else completion.modeled_queue_ns
                ),
                modeled_service_ns=(
                    None if completion is None else completion.modeled_service_ns
                ),
                effective_elapsed_ns=(
                    None if completion is None else completion.effective_elapsed_ns
                ),
                state=state,
                terminal=terminal,
            )
        )

    def _begin_operation(self, op_id: str, instance_id: int) -> None:
        if op_id in self._active_operations:
            raise RuntimeError("operation ID is already active")
        self._active_operations[op_id] = _ActiveOperation(instance_id)

    def _execute_plan(
        self, parent_op_id: str, plan: TransferPlan
    ) -> CompositeCompletion:
        coordinator = self._modeled_coordinator
        if coordinator is None:
            return self._model_client.join(self._require_executor().submit(plan))
        with self._lock:
            operation = self._active_operations[parent_op_id]
            operation.modeled_op_ids.add(plan.op_id)
        try:
            return coordinator.run(
                op_id=plan.op_id,
                instance_id=plan.instance_id + 1,
                direction=plan.direction,
                offset=plan.extent.offset,
                bytes=plan.extent.length,
                launch=lambda: self._require_executor().submit(plan),
            )
        except BaseException:
            self._raise_if_cancelled(parent_op_id)
            raise
        finally:
            with self._lock:
                active = self._active_operations.get(parent_op_id)
                if active is not None:
                    active.modeled_op_ids.discard(plan.op_id)

    def _finish_operation(self, op_id: str) -> None:
        with self._condition:
            self._active_operations.pop(op_id, None)
            self._condition.notify_all()

    def _raise_if_cancelled(self, op_id: str) -> None:
        with self._lock:
            operation = self._active_operations.get(op_id)
            if operation is None or operation.cancelled:
                raise _OperationCancelled("CXL operation was cancelled")

    def _require_live_lease(self, lease: ReadLease) -> None:
        if lease.expires_at_ns <= self._clock_ns():
            raise RuntimeError("CXL read lease expired during transfer")

    @staticmethod
    def _validate_attention_layout(cache_context: Any) -> None:
        manager = cache_context.kv_layer_groups_manager
        windows = tuple(manager.get_attn_desc().num_chunks_in_sw)
        if len(windows) != len(manager.object_groups):
            raise ValueError("attention descriptor does not match object groups")
        if any(window != -1 for window in windows):
            raise ValueError(
                "Gate B CXL shared tier requires full-attention object groups"
            )

    def _require_directory(self) -> SingleResidencyDirectory:
        if self._directory is None:
            raise RuntimeError("no engine layout is registered")
        return self._directory

    def _require_executor(self) -> LMCacheIPCTransferExecutor:
        if self._executor is None:
            raise RuntimeError("no engine layout is registered")
        return self._executor

    def _require_layout(self) -> Any:
        if self._layout is None:
            raise RuntimeError("no engine layout is registered")
        return self._layout

    def _require_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("CXL shared tier is closed")

    def _publish_aliases(
        self, primary: ObjectKey, aliases: tuple[ObjectKey, ...]
    ) -> None:
        for alias in aliases:
            owner = self._primary_by_alias.get(alias)
            if owner is not None and owner != primary:
                raise RuntimeError("ObjectKey alias belongs to another residency")
        self._aliases_by_primary[primary] = aliases
        for alias in aliases:
            self._primary_by_alias[alias] = primary

    def _existing_primary(self, aliases: tuple[ObjectKey, ...]) -> ObjectKey | None:
        owners = {self._primary_by_alias.get(alias) for alias in aliases}
        known = owners - {None}
        if not known:
            return None
        if len(known) != 1 or None in owners:
            raise RuntimeError("ObjectKey alias belongs to another residency")
        primary = next(iter(known))
        assert primary is not None
        self._validate_aliases(primary, aliases)
        if self._require_directory().lookup_ready(primary) is None:
            raise RuntimeError("ObjectKey aliases reference a non-READY residency")
        return primary

    def _validate_aliases(
        self, primary: ObjectKey, aliases: tuple[ObjectKey, ...]
    ) -> None:
        if self._aliases_by_primary.get(primary) != aliases:
            raise RuntimeError("immutable CXL object aliases do not match")

    def _primary_for_chunk(self, aliases: tuple[ObjectKey, ...]) -> ObjectKey | None:
        primaries = {self._primary_by_alias.get(alias) for alias in aliases}
        if None in primaries or len(primaries) != 1:
            return None
        return next(iter(primaries))

    @staticmethod
    def _retrievable_chunks(
        chunks: tuple[VLLMTransferRequest, ...],
        request: VLLMTransferRequest,
    ) -> tuple[VLLMTransferRequest, ...]:
        retrievable: list[VLLMTransferRequest] = []
        for chunk_id, chunk in enumerate(chunks):
            chunk_start = chunk_id * request.token_count
            local_skip = max(0, request.skip_first_n_tokens - chunk_start)
            if local_skip >= request.token_count:
                continue
            retrievable.append(replace(chunk, skip_first_n_tokens=local_skip))
        return tuple(retrievable)
