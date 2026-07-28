# SPDX-License-Identifier: Apache-2.0
"""Gate B orchestration for immutable GPU-visible CXL proxy objects."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any, Callable, Literal, Protocol, Sequence
import hashlib
import json
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import EncodedObjectKey, ObjectKey
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.actions import (
    FetchDecision,
    LookupDecision,
    RecomputeDecision,
    TargetSpec,
    Tier,
)
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
from lmcache.v1.multiprocess.cxl.fallback import (
    FallbackCoordinator,
    FallbackResult,
    LoadCancelled,
)
from lmcache.v1.multiprocess.cxl.directory import (
    MultiResidencyDirectory,
    Residency,
    ResidencyState,
)
from lmcache.v1.multiprocess.cxl.layout import layout_fingerprint
from lmcache.v1.multiprocess.cxl.observations import (
    PlacementSnapshot,
    RequestObservation,
    TierObservation,
)
from lmcache.v1.multiprocess.cxl.placement import (
    BoundLookup,
    LookupCoordinator,
    StoreCoordinator,
    TargetTransferCompletion,
    snapshot_from_directory,
)
from lmcache.v1.multiprocess.cxl.policies import CostAwarePolicy
from lmcache.v1.multiprocess.cxl.policy_protocol import GateERequestEnvelope
from lmcache.v1.multiprocess.cxl.model_client import CXLMemSimModelClient
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import (
    CXLMemSimShmRegionProvider,
    LocalDRAMRegionProvider,
    PosixShmRegionProvider,
    RegionHandle,
    RegionProvider,
)
from lmcache.v1.multiprocess.cxl.residency import (
    ReadLease,
)
from lmcache.v1.multiprocess.cxl.tickets import TicketManager, TransferTicket
from lmcache.v1.multiprocess.cxl.telemetry import (
    LoggingPolicyEventSink,
    PolicyEvent,
    PolicyEventSink,
    ReasonCode,
)


logger = init_logger(__name__)


DirectPath = Literal["dram_direct", "cxl_direct", "multi_direct"]


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
    path: DirectPath
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
        logger.debug(
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
    path: DirectPath
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
        if self.path not in ("dram_direct", "cxl_direct", "multi_direct"):
            raise ValueError("shared-tier path is unsupported")
        if self.dram_allocated_bytes_delta != 0:
            raise ValueError("direct transfer cannot allocate DRAM staging")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.status == "error" and not self.error:
            raise ValueError("error result requires a diagnostic")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error result cannot contain a diagnostic")


@dataclass
class _StoreChunkSource:
    released: bool = False

    def release(self) -> None:
        self.released = True


class _StoreTargetExecutor:
    def __init__(
        self,
        transfer: Callable[
            [TargetSpec, Residency, _StoreChunkSource], TargetTransferCompletion
        ],
    ) -> None:
        self._transfer = transfer

    def transfer(
        self,
        target: TargetSpec,
        residency: Residency,
        source: _StoreChunkSource,
    ) -> TargetTransferCompletion:
        return self._transfer(target, residency, source)


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
        policy_event_sink: PolicyEventSink,
        modeled_coordinator: ModeledCompletionCoordinator | None,
        dram_provider: RegionProvider | None = None,
        dram_handle: RegionHandle | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._handle = handle
        self._native_ops = native_ops
        self._runtime = runtime
        self._clock_ns = clock_ns
        self._operation_sink = operation_sink
        self._policy_event_sink = policy_event_sink
        self._dram_provider = dram_provider
        self._dram_handle = dram_handle
        self._region_managers: dict[Tier, CXLRegionManager] = {}
        self._region_views: dict[Tier, RegisteredRegionView] = {}
        self._executors: dict[Tier, LMCacheIPCTransferExecutor] = {}
        self._directory: MultiResidencyDirectory | None = None
        self._ticket_manager: TicketManager | None = None
        self._lookup_coordinator: LookupCoordinator | None = None
        self._placement_policy: CostAwarePolicy | None = None
        self._lookup_tickets: dict[str, dict[ObjectKey, TransferTicket]] = {}
        self._lookup_observations: dict[str, dict[ObjectKey, RequestObservation]] = {}
        self._lookup_decision_reasons: dict[str, str] = {}
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
        policy_event_sink: PolicyEventSink | None = None,
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
        dram_provider: LocalDRAMRegionProvider | None = None
        try:
            handle = provider.provision()
            dram_handle = None
            if config.dram_capacity_bytes:
                dram_provider = LocalDRAMRegionProvider(
                    capacity=config.dram_capacity_bytes,
                    alignment=config.alignment_bytes,
                )
                dram_handle = dram_provider.provision()
            coordinator = None
            if config.model_mode == "cxlmemsim":
                if config.model_client_library is None:
                    raise ValueError("cxlmemsim model mode requires a client library")
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
                policy_event_sink=(
                    LoggingPolicyEventSink(logger)
                    if policy_event_sink is None
                    else policy_event_sink
                ),
                modeled_coordinator=coordinator,
                dram_provider=dram_provider,
                dram_handle=dram_handle,
            )
        except BaseException:
            if model_client is not None:
                model_client.close()
            if dram_provider is not None:
                dram_provider.close()
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
            if not self._executors:
                fingerprint = layout_fingerprint(layout)
                handles: dict[Tier, RegionHandle] = {"cxl": self._handle}
                if self._dram_handle is not None:
                    handles["dram"] = self._dram_handle
                for tier, region_handle in handles.items():
                    manager = CXLRegionManager(
                        region_handle,
                        layout_id=layout.layout_id,
                        layout_fingerprint=fingerprint,
                        tier=tier,
                    )
                    view = RegisteredRegionView.open(
                        region_handle,
                        descriptor_validator=manager.validate_descriptor,
                        native_ops=self._native_ops,
                        expected_layout_fingerprint=fingerprint,
                    )
                    self._region_managers[tier] = manager
                    self._region_views[tier] = view
                    self._executors[tier] = LMCacheIPCTransferExecutor(
                        view,
                        native_ops=self._native_ops,
                        runtime=self._runtime,
                        clock_ns=self._clock_ns,
                    )
                self._directory = MultiResidencyDirectory(self._region_managers)
                self._ticket_manager = TicketManager(
                    self._directory, ticket_ttl_ns=self._LEASE_TTL_NS
                )
                policy = CostAwarePolicy(
                    cuda_bandwidth_bytes_per_s=(
                        self._config.policy_cuda_bandwidth_bytes_per_s
                    ),
                    store_targets=self._store_targets(),
                )
                self._placement_policy = policy
                self._lookup_coordinator = LookupCoordinator(
                    policy,
                    self._ticket_manager,
                )
                self._layout = layout
            else:
                adapter.validate_context(cache_context, self._require_layout())
            for executor in self._executors.values():
                executor.register_engine(instance_id, cache_context)
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
            for executor in self._executors.values():
                executor.unregister_engine(instance_id)

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
        """Store chunks according to the online multi-residency policy.

        Args:
            request: Complete engine transfer metadata for one STORE request.

        Returns:
            A terminal result after every required target succeeds. Successful
            optional replicas remain READY when a sibling target fails.
        """
        payload_bytes = 0
        direct_path = self.store_path()
        pending_aliases: list[ObjectKey] = []
        completions: list[CompositeCompletion] = []
        existing: list[tuple[VLLMTransferRequest, Residency, ObjectKey]] = []
        operation_started = False
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
                    residency = self._lookup_ready(existing_primary)
                    if residency is None or residency.descriptor is None:
                        raise RuntimeError("existing residency is not READY")
                    existing.append((chunk, residency, existing_primary))
                new_aliases = [
                    alias for chunk in pending_chunks for alias in chunk.object_keys
                ]
                if len(new_aliases) != len(set(new_aliases)):
                    raise RuntimeError(
                        "ObjectKey alias appears in multiple new shared-tier chunks"
                    )
                if any(alias in self._pending_aliases for alias in new_aliases):
                    raise RuntimeError("ObjectKey alias has an active STORE")
                for alias in new_aliases:
                    self._pending_aliases[alias] = request.op_id
                    pending_aliases.append(alias)

                self._begin_operation(request.op_id, request.instance_id)
                operation_started = True

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

            for chunk in pending_chunks:
                self._raise_if_cancelled(request.op_id)
                primary = chunk.object_keys[0]
                length = adapter.packed_size_bytes(chunk)
                observation = RequestObservation(
                    request.op_id,
                    request.instance_id,
                    primary,
                    length,
                    request.token_count,
                    None,
                    1,
                    self._require_layout_fingerprint(),
                )
                snapshot = snapshot_from_directory(
                    self._require_directory(),
                    observation,
                    self._tier_observations(),
                    self._clock_ns(),
                )
                placement = self._require_placement_policy().plan_store(snapshot)
                executed: list[
                    tuple[TargetSpec, Residency, TransferPlan, CompositeCompletion]
                ] = []

                source = _StoreChunkSource()
                target_executor = _StoreTargetExecutor(
                    partial(
                        self._execute_store_target,
                        request=request,
                        chunk=chunk,
                        primary=primary,
                        adapter=adapter,
                        completions=completions,
                        executed=executed,
                    )
                )
                result = StoreCoordinator(
                    self._require_directory(), target_executor
                ).execute(
                    placement,
                    source,
                    length=length,
                    alignment=self._config.alignment_bytes,
                )
                if not source.released:
                    raise RuntimeError("STORE source lifetime was not released")
                self._raise_if_cancelled(request.op_id)
                if result.successful_residencies:
                    with self._lock:
                        self._publish_aliases(primary, chunk.object_keys)
                completion_by_tier = {
                    target.tier: completion for target, _, _, completion in executed
                }
                for ready_residency in result.successful_residencies:
                    completion = completion_by_tier[ready_residency.tier]
                    self._record_event(
                        op_id=completion.op_id,
                        instance_id=request.instance_id,
                        object_key=primary,
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
                if not result.required_satisfied:
                    errors = "; ".join(
                        f"{failed.tier}: {failed.error}"
                        for failed in result.failed_targets
                        if failed.required
                    )
                    raise RuntimeError(errors or "required STORE target failed")
            return self._result(
                "ok",
                tuple(completions),
                payload_bytes=payload_bytes,
                path=direct_path,
            )
        except _OperationCancelled:
            return self._result(
                "cancelled",
                tuple(completions),
                payload_bytes=payload_bytes,
                path=direct_path,
            )
        except Exception as error:
            diagnostic = str(error) or self._completion_error(completions)
            return self._result(
                "error",
                tuple(completions),
                diagnostic,
                payload_bytes=payload_bytes,
                path=direct_path,
            )
        finally:
            with self._lock:
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
        bound_tickets: list[TransferTicket | None] = []
        lookup_observations: list[RequestObservation | None] = []
        ready: list[tuple[VLLMTransferRequest, Residency, ObjectKey]] = []
        direct_path: DirectPath = "cxl_direct"
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            all_chunks = adapter.split_chunks(request)
            chunks = self._retrievable_chunks(all_chunks, request)
            payload_bytes = sum(adapter.packed_size_bytes(chunk) for chunk in chunks)
            selected: list[
                tuple[
                    VLLMTransferRequest,
                    Residency,
                    ObjectKey,
                    TransferTicket | None,
                    RequestObservation | None,
                ]
            ] = []
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
                ticket = None
                residency: Residency | None
                if request.external_request_id is not None:
                    ticket = self._lookup_tickets.get(
                        request.external_request_id, {}
                    ).get(primary)
                    if ticket is None:
                        return self._result("miss", payload_bytes=payload_bytes)
                    bound = self._require_ticket_manager().validate(
                        ticket, self._clock_ns()
                    )
                    residency = bound.residency
                    observation = self._lookup_observations.get(
                        request.external_request_id, {}
                    ).get(primary)
                else:
                    residency = self._lookup_ready(primary)
                    observation = None
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
                selected.append((chunk, residency, primary, ticket, observation))
            for chunk, residency, primary, ticket, observation in selected:
                if ticket is not None:
                    lease = ReadLease(
                        lease_id=ticket.lease_id,
                        residency_id=ticket.residency_id,
                        generation=ticket.generation,
                        expires_at_ns=ticket.expires_at_ns,
                    )
                else:
                    lease = self._require_directory().acquire_read(
                        residency.residency_id,
                        residency.generation,
                        self._LEASE_TTL_NS,
                        now_ns=self._clock_ns(),
                    )
                leases.append(lease)
                bound_tickets.append(ticket)
                lookup_observations.append(observation)
                ready.append((chunk, residency, primary))
                if lease.generation != residency.generation:
                    raise RuntimeError("read lease generation changed")
            direct_path = self._path_for_tiers(
                frozenset(residency.tier for _, residency, _ in ready)
            )
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
        fallback_used = False
        try:
            for index, (chunk, residency, _) in enumerate(ready):
                self._raise_if_cancelled(request.op_id)
                self._require_live_lease(leases[index])
                ticket = bound_tickets[index]
                observation = lookup_observations[index]
                terminal_lease_id = leases[index].lease_id
                if ticket is not None and observation is not None:
                    fallback, selected_residency, selected_ticket = (
                        self._execute_retrieve_with_fallback(
                            request,
                            chunk,
                            ticket,
                            observation,
                            adapter,
                            completions,
                            allow_alternate=not fallback_used,
                        )
                    )
                    fallback_used = fallback_used or fallback.attempts > 1
                    if fallback.status == "cancelled":
                        raise _OperationCancelled(fallback.reason)
                    if fallback.status == "recompute":
                        raise RuntimeError(fallback.reason)
                    if selected_residency is None or selected_ticket is None:
                        raise RuntimeError("successful fallback has no residency")
                    residency = selected_residency
                    ready[index] = (chunk, residency, ready[index][2])
                    terminal_lease_id = selected_ticket.lease_id
                    completion = completions[-1]
                else:
                    descriptor = residency.descriptor
                    if descriptor is None:
                        raise RuntimeError("READY residency has no extent")
                    plan = adapter.build_retrieve_plan(chunk, descriptor)
                    completion = self._execute_plan(request.op_id, plan, residency.tier)
                    completions.append(completion)
                    self._raise_if_cancelled(request.op_id)
                    self._require_live_lease(leases[index])
                    if not completion.success:
                        raise RuntimeError(
                            completion.error or "RETRIEVE completion failed"
                        )
                self._record_event(
                    op_id=chunk.op_id,
                    instance_id=request.instance_id,
                    object_key=chunk.object_keys[0],
                    direction="retrieve",
                    extent=residency.descriptor,
                    generation=residency.generation,
                    lease_id=terminal_lease_id,
                    payload_checksum=request.payload_checksum_expected,
                    cuda_elapsed_ns=completion.cuda_elapsed_ns,
                    state="ok",
                    terminal=True,
                    completion=completion,
                )
                terminal_indices.add(index)
            direct_path = self._path_for_tiers(
                frozenset(residency.tier for _, residency, _ in ready)
            )
            return self._result(
                "ok",
                tuple(completions),
                payload_bytes=payload_bytes,
                path=direct_path,
            )
        except _OperationCancelled:
            terminal_state = "cancelled"
            return self._result(
                "cancelled",
                tuple(completions),
                payload_bytes=payload_bytes,
                path=direct_path,
            )
        except Exception as error:
            terminal_state = "error"
            diagnostic = str(error) or self._completion_error(completions)
            return self._result(
                "error",
                tuple(completions),
                diagnostic,
                payload_bytes=payload_bytes,
                path=direct_path,
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
                for index, (lease, ticket) in enumerate(
                    zip(leases, bound_tickets, strict=True)
                ):
                    if ticket is not None:
                        if terminal_state == "cancelled":
                            self._retire_ticket(
                                ticket,
                                cancelled_reason="RETRIEVE was cancelled",
                            )
                        else:
                            outcome: Literal["ok", "error"] = (
                                "ok" if index in terminal_indices else "error"
                            )
                            self._retire_ticket(ticket, outcome=outcome)
                        continue
                    leased_primary = (
                        self._require_directory()
                        .get_residency(lease.residency_id)
                        .object_key
                    )
                    self._require_directory().release_read(
                        lease.lease_id, now_ns=self._clock_ns()
                    )
                    try:
                        self._require_directory().reclaim(lease.residency_id)
                    except (KeyError, RuntimeError):
                        pass
                    else:
                        self._remove_aliases(leased_primary)
                if request.external_request_id is not None:
                    self._lookup_tickets.pop(request.external_request_id, None)
                    self._lookup_observations.pop(request.external_request_id, None)
            finally:
                self._finish_operation(request.op_id)

    def contains(self, request: VLLMTransferRequest) -> bool:
        """Return whether every executable chunk has a bound READY residency.

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
            tickets = (
                None
                if request.external_request_id is None
                else self._lookup_tickets.get(request.external_request_id)
            )
            if request.external_request_id is not None and tickets is None:
                return False
            for chunk in chunks:
                primary = self._primary_for_chunk(chunk.object_keys)
                if primary is None:
                    return False
                residency: Residency | None
                if tickets is not None:
                    ticket = tickets.get(primary)
                    if ticket is None:
                        return False
                    try:
                        bound = self._require_ticket_manager().validate(
                            ticket, self._clock_ns()
                        )
                    except RuntimeError:
                        return False
                    residency = bound.residency
                else:
                    residency = self._lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    return False
            return True

    def store_path(self) -> DirectPath:
        """Return the configured direct STORE data path.

        Returns:
            CXL, DRAM, or multi-target direct path identity.
        """
        return self._path_for_tiers(
            frozenset(target.tier for target in self._store_targets())
        )

    def retrieve_path(self, request: VLLMTransferRequest) -> DirectPath:
        """Return the tier path bound for a RETRIEVE request.

        Args:
            request: Candidate RETRIEVE metadata.

        Returns:
            CXL, DRAM, or mixed direct path identity.

        Raises:
            RuntimeError: If no complete READY source set is available.
        """
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            tiers: set[Tier] = set()
            for chunk in self._retrievable_chunks(
                adapter.split_chunks(request), request
            ):
                primary = self._primary_for_chunk(chunk.object_keys)
                if primary is None:
                    raise RuntimeError("RETRIEVE chunk has no READY residency")
                residency: Residency | None
                if request.external_request_id is not None:
                    ticket = self._lookup_tickets.get(
                        request.external_request_id, {}
                    ).get(primary)
                    if ticket is None:
                        raise RuntimeError("RETRIEVE chunk has no bound ticket")
                    residency = (
                        self._require_ticket_manager()
                        .validate(ticket, self._clock_ns())
                        .residency
                    )
                else:
                    residency = self._lookup_ready(primary)
                    if residency is None:
                        raise RuntimeError("RETRIEVE chunk has no READY residency")
                tiers.add(residency.tier)
            return self._path_for_tiers(frozenset(tiers))

    def lookup_ready(self, object_key: ObjectKey) -> Residency | None:
        """Return the READY residency addressed by an object or alias.

        Args:
            object_key: Logical packed object identity.

        Returns:
            The immutable READY residency, otherwise None.
        """
        with self._lock:
            primary = self._primary_by_alias.get(object_key, object_key)
            return self._lookup_ready(primary)

    def list_residencies(self, object_key: ObjectKey) -> tuple[Residency, ...]:
        """Return every residency addressed by an object or alias.

        Args:
            object_key: Logical packed object identity.

        Returns:
            Immutable snapshots of all DRAM/CXL replicas.
        """
        with self._lock:
            primary = self._primary_by_alias.get(object_key, object_key)
            return self._require_directory().list_residencies(primary)

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
                residency = self._lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    break
                found += 1
            return found

    def bind_ready_prefix(
        self,
        request_id: str,
        object_keys: Sequence[ObjectKey],
        keys_per_chunk: int,
    ) -> int:
        """Bind one ticket per READY prefix chunk before reporting a hit."""
        if not request_id:
            raise ValueError("request_id must not be empty")
        if keys_per_chunk <= 0 or len(object_keys) % keys_per_chunk != 0:
            raise ValueError("object keys do not contain complete chunks")
        with self._lock:
            self.release_lookup_tickets(request_id, "lookup was replaced")
            bound: dict[ObjectKey, TransferTicket] = {}
            for start in range(0, len(object_keys), keys_per_chunk):
                aliases = tuple(object_keys[start : start + keys_per_chunk])
                primary = self._primary_for_chunk(aliases)
                if primary is None:
                    break
                residency = self._lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    break
                try:
                    ticket = self._require_ticket_manager().bind_fetch(
                        FetchDecision(
                            residency.residency_id,
                            max(1, residency.descriptor.length),
                            f"bound {residency.tier} direct prefix",
                        ),
                        request_id,
                        self._clock_ns(),
                    )
                except RuntimeError:
                    break
                bound[primary] = ticket
            if bound:
                self._lookup_tickets[request_id] = bound
            return len(bound)

    def policy_bind_ready_prefix(
        self,
        request_id: str,
        object_keys: Sequence[ObjectKey],
        keys_per_chunk: int,
        envelope: GateERequestEnvelope,
    ) -> int:
        """Apply the cost policy and bind every reported prefix chunk."""
        if envelope.request_id != request_id:
            raise ValueError("policy envelope request_id does not match request")
        if keys_per_chunk <= 0 or len(object_keys) % keys_per_chunk != 0:
            raise ValueError("object keys do not contain complete chunks")
        with self._lock:
            self.release_lookup_tickets(request_id, "policy lookup was replaced")
            bound: dict[ObjectKey, TransferTicket] = {}
            observations: dict[ObjectKey, RequestObservation] = {}
            reason = "no READY policy candidate"
            for start in range(0, len(object_keys), keys_per_chunk):
                aliases = tuple(object_keys[start : start + keys_per_chunk])
                primary = self._primary_for_chunk(aliases)
                if primary is None:
                    break
                candidates = self._ready_residencies(primary)
                if not candidates or candidates[0].descriptor is None:
                    break
                residency = candidates[0]
                if residency.descriptor is None:
                    raise RuntimeError("READY residency has no extent")
                request = RequestObservation(
                    request_id,
                    envelope.instance_id,
                    primary,
                    residency.descriptor.length,
                    0,
                    envelope.deadline_ns,
                    envelope.recompute_estimate_ns,
                    envelope.layout_fingerprint,
                )

                def fresh_snapshot(
                    request: RequestObservation = request,
                ) -> PlacementSnapshot:
                    return snapshot_from_directory(
                        self._require_directory(),
                        request,
                        self._tier_observations(),
                        self._clock_ns(),
                    )

                decision = self._require_lookup_coordinator().decide_and_bind(
                    request_id,
                    fresh_snapshot,
                    now_ns=self._clock_ns(),
                )
                reason = (
                    decision.decision.reason
                    if isinstance(decision, BoundLookup)
                    else decision.reason
                )
                if isinstance(decision, RecomputeDecision):
                    self._record_policy_decision(
                        request,
                        residency,
                        decision,
                        None,
                    )
                    break
                bound[primary] = decision.ticket
                observations[primary] = request
                chosen = self._require_directory().get_residency(
                    decision.ticket.residency_id
                )
                self._record_policy_decision(
                    request,
                    chosen,
                    decision.decision,
                    decision.ticket,
                )
            if bound:
                self._lookup_tickets[request_id] = bound
                self._lookup_observations[request_id] = observations
            self._lookup_decision_reasons[request_id] = reason
            return len(bound)

    def release_lookup_tickets(
        self,
        request_id: str,
        reason: str,
        object_keys: Sequence[ObjectKey] | None = None,
    ) -> None:
        """Cancel all unconsumed prefix tickets for one external request."""
        if not reason:
            raise ValueError("ticket release reason must not be empty")
        with self._lock:
            tickets = self._lookup_tickets.get(request_id, {})
            observations = self._lookup_observations.get(request_id, {})
            selected = (
                None
                if object_keys is None
                else {self._primary_by_alias.get(key, key) for key in object_keys}
            )
            for primary, ticket in tuple(tickets.items()):
                if selected is not None and primary not in selected:
                    continue
                self._retire_ticket(ticket, cancelled_reason=reason)
                tickets.pop(primary, None)
                observations.pop(primary, None)
            if not tickets:
                self._lookup_tickets.pop(request_id, None)
                self._lookup_observations.pop(request_id, None)
            if object_keys is None:
                self._lookup_decision_reasons.pop(request_id, None)

    def get_bound_ticket_ids(self, request_id: str) -> tuple[str, ...]:
        """Return ticket IDs already bound for a direct prefix lookup."""
        with self._lock:
            return tuple(
                ticket.op_id
                for ticket in self._lookup_tickets.get(request_id, {}).values()
            )

    def get_lookup_decision_reason(self, request_id: str) -> str:
        """Return the structured reason for the last policy prefix decision."""
        with self._lock:
            return self._lookup_decision_reasons.get(
                request_id, "ticket-bound policy prefix"
            )

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
            residency = self._lookup_ready(primary)
            if residency is None:
                raise KeyError("ObjectKey has no READY residency")
            directory = self._require_directory()
            directory.begin_evict(residency.residency_id)
            reclaimed = False
            try:
                directory.reclaim(residency.residency_id)
            except RuntimeError as error:
                if "active readers" not in str(error):
                    raise
            else:
                reclaimed = True
            if reclaimed and not self._ready_residencies(primary):
                self._remove_aliases(primary)

    def close(self) -> None:
        """Release engine registrations, CUDA mapping, and provider handles."""
        with self._condition:
            if self._closed:
                return
            self._closing = True
            for operation in self._active_operations.values():
                operation.cancelled = True
            self._condition.wait_for(lambda: not self._active_operations)
            for request_id in tuple(self._lookup_tickets):
                self.release_lookup_tickets(request_id, "shared tier closed")
            for instance_id in tuple(self._adapters):
                for executor in self._executors.values():
                    executor.unregister_engine(instance_id)
            self._adapters.clear()
            for view in self._region_views.values():
                view.close()
            if self._modeled_coordinator is not None:
                self._modeled_coordinator.close()
            if self._dram_provider is not None:
                self._dram_provider.close()
            self._provider.close()
            self._closed = True

    @staticmethod
    def _result(
        status: Literal["ok", "miss", "error", "cancelled"],
        completions: tuple[CompositeCompletion, ...] = (),
        error: str | None = None,
        *,
        payload_bytes: int = 0,
        path: DirectPath = "cxl_direct",
    ) -> CXLTransferResult:
        return CXLTransferResult(
            status=status,
            path=path,
            completions=completions,
            payload_bytes=payload_bytes,
            dram_allocated_bytes_delta=0,
            error=error,
        )

    @staticmethod
    def _completion_error(completions: list[CompositeCompletion]) -> str | None:
        if completions and not completions[-1].success:
            return completions[-1].error or "composite transfer failed"
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
                path=(
                    "dram_direct"
                    if extent is not None and extent.tier == "dram"
                    else "cxl_direct"
                ),
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

    def _record_policy_decision(
        self,
        request: RequestObservation,
        residency: Residency,
        decision: FetchDecision | RecomputeDecision,
        ticket: TransferTicket | None,
    ) -> None:
        reason = decision.reason
        try:
            detail = json.loads(reason)
        except json.JSONDecodeError:
            detail = {}
        estimates = detail.get("estimates", {})
        candidate_estimates = tuple(
            sorted(
                (str(name), int(value))
                for name, value in estimates.items()
                if isinstance(value, int) and value >= 0
            )
        )
        rejections = detail.get("rejections", {})
        reason_code: ReasonCode
        if any(value == "layout_mismatch" for value in rejections.values()):
            reason_code = "layout_rejection"
        elif any(value == "deadline_infeasible" for value in rejections.values()):
            reason_code = "deadline"
        elif "could not bind" in reason:
            reason_code = "ticket_conflict"
        elif not isinstance(decision, FetchDecision) and not estimates:
            reason_code = "no_candidate"
        else:
            reason_code = "minimum_cost"
        encoded = json.dumps(
            asdict(request.object_key.to_encoded_object_key()), sort_keys=True
        ).encode()
        descriptor = residency.descriptor
        self._policy_event_sink.record(
            PolicyEvent(
                timestamp_ns=self._clock_ns(),
                run_id="live",
                op_id="policy-" + request.request_id,
                request_id=request.request_id,
                object_id="sha256:" + hashlib.sha256(encoded).hexdigest(),
                residency_id=(
                    residency.residency_id
                    if isinstance(decision, FetchDecision)
                    else None
                ),
                instance_id=request.instance_id,
                event="lookup_decision",
                tier=(residency.tier if isinstance(decision, FetchDecision) else None),
                path=self._path_for_tiers(frozenset({residency.tier})),
                payload_bytes=request.required_bytes,
                tokens=request.external_matched_tokens,
                state=residency.state.value,
                generation=residency.generation,
                layout_fingerprint=(
                    None if descriptor is None else descriptor.layout_fingerprint
                ),
                candidate_estimates_ns=candidate_estimates,
                chosen_action=(
                    "fetch" if isinstance(decision, FetchDecision) else "recompute"
                ),
                reason_code=reason_code,
                reason=reason,
                queue_ns=None,
                cuda_estimate_ns=None,
                modeled_estimate_ns=(
                    decision.estimated_completion_ns
                    if isinstance(decision, FetchDecision)
                    else None
                ),
                recompute_estimate_ns=request.recompute_estimate_ns,
                cuda_actual_ns=None,
                modeled_actual_ns=None,
                effective_actual_ns=None,
                ticket_id=None if ticket is None else ticket.op_id,
                fallback_from=None,
                fallback_to=None,
                required=None,
                partial_success=False,
                invalidated_blocks=0,
                terminal_error=None,
            )
        )

    def _begin_operation(self, op_id: str, instance_id: int) -> None:
        if op_id in self._active_operations:
            raise RuntimeError("operation ID is already active")
        self._active_operations[op_id] = _ActiveOperation(instance_id)

    def _execute_store_target(
        self,
        target: TargetSpec,
        residency: Residency,
        source: _StoreChunkSource,
        *,
        request: VLLMTransferRequest,
        chunk: VLLMTransferRequest,
        primary: ObjectKey,
        adapter: VLLMDataPlaneAdapter,
        completions: list[CompositeCompletion],
        executed: list[tuple[TargetSpec, Residency, TransferPlan, CompositeCompletion]],
    ) -> TargetTransferCompletion:
        del source
        self._raise_if_cancelled(request.op_id)
        descriptor = residency.descriptor
        if descriptor is None:
            return TargetTransferCompletion(
                target.tier, False, "WRITING residency has no extent"
            )
        plan = adapter.build_store_plan(chunk, descriptor)
        plan = replace(plan, op_id=f"{plan.op_id}:{target.tier}")
        completion = self._execute_plan(request.op_id, plan, target.tier)
        completions.append(completion)
        executed.append((target, residency, plan, completion))
        try:
            self._raise_if_cancelled(request.op_id)
        except _OperationCancelled:
            self._record_store_terminal(request, primary, plan, completion, "cancelled")
            raise
        if not completion.success:
            self._record_store_terminal(request, primary, plan, completion, "error")
        return TargetTransferCompletion(
            target.tier,
            completion.success,
            None
            if completion.success
            else completion.error or "composite STORE completion failed",
        )

    def _execute_retrieve_with_fallback(
        self,
        request: VLLMTransferRequest,
        chunk: VLLMTransferRequest,
        initial: TransferTicket,
        observation: RequestObservation,
        adapter: VLLMDataPlaneAdapter,
        completions: list[CompositeCompletion],
        *,
        allow_alternate: bool,
    ) -> tuple[FallbackResult, Residency | None, TransferTicket | None]:
        """Execute one policy-bound fetch with at most one alternate source.

        Args:
            request: Parent RETRIEVE metadata.
            chunk: One independently restorable packed chunk.
            initial: Ticket bound before the external hit was reported.
            observation: Policy input retained from lookup time.
            adapter: Engine-specific transfer-plan adapter.
            completions: Request completion evidence accumulator.
            allow_alternate: Whether the request fallback budget remains.

        Returns:
            Fallback outcome, final residency, and final ticket.
        """
        invalidated: list[int] = []
        selected_residency: Residency | None = None
        selected_ticket: TransferTicket | None = None
        ticket_manager = self._require_ticket_manager()

        def transfer(ticket: TransferTicket, overwrite: bool) -> None:
            nonlocal selected_residency, selected_ticket
            if not overwrite:
                raise RuntimeError("fallback fetch must fully overwrite HBM")
            try:
                self._raise_if_cancelled(request.op_id)
            except _OperationCancelled as error:
                raise LoadCancelled(str(error)) from error
            bound = ticket_manager.validate(ticket, self._clock_ns())
            descriptor = bound.residency.descriptor
            if descriptor is None:
                raise RuntimeError("READY residency has no extent")
            plan = adapter.build_retrieve_plan(chunk, descriptor)
            completion = self._execute_plan(request.op_id, plan, bound.residency.tier)
            completions.append(completion)
            try:
                self._raise_if_cancelled(request.op_id)
            except _OperationCancelled as error:
                raise LoadCancelled(str(error)) from error
            if not completion.success:
                raise RuntimeError(completion.error or "RETRIEVE completion failed")
            selected_residency = bound.residency
            selected_ticket = ticket

        def choose_alternate(excluded: frozenset[str], now_ns: int) -> LookupDecision:
            if not allow_alternate:
                return RecomputeDecision("request fallback budget exhausted")
            snapshot = snapshot_from_directory(
                self._require_directory(),
                observation,
                self._tier_observations(),
                now_ns,
                excluded_residencies=excluded,
            )
            return self._require_placement_policy().decide_lookup(snapshot)

        destination_blocks = tuple(
            dict.fromkeys(
                block_id for group in chunk.block_ids_by_group for block_id in group
            )
        )
        result = FallbackCoordinator(
            ticket_manager,
            invalidate=invalidated.extend,
        ).execute(
            initial,
            transfer=transfer,
            choose_alternate=choose_alternate,
            destination_block_ids=destination_blocks,
            deadline_ns=observation.deadline_ns,
            now_ns=self._clock_ns,
        )
        if result.attempts > 1 or result.status != "ok":
            self._record_fallback_event(
                request,
                observation,
                result,
                selected_residency,
                selected_ticket,
                len(invalidated),
            )
        return result, selected_residency, selected_ticket

    def _record_fallback_event(
        self,
        request: VLLMTransferRequest,
        observation: RequestObservation,
        result: FallbackResult,
        residency: Residency | None,
        ticket: TransferTicket | None,
        invalidated_blocks: int,
    ) -> None:
        encoded = json.dumps(
            asdict(observation.object_key.to_encoded_object_key()), sort_keys=True
        ).encode()
        tiers = frozenset(
            tier
            for tier in (result.fallback_from, result.fallback_to)
            if tier in ("dram", "cxl")
        )
        self._policy_event_sink.record(
            PolicyEvent(
                timestamp_ns=self._clock_ns(),
                run_id="live",
                op_id=request.op_id,
                request_id=observation.request_id,
                object_id="sha256:" + hashlib.sha256(encoded).hexdigest(),
                residency_id=None if residency is None else residency.residency_id,
                instance_id=observation.instance_id,
                event="fallback",
                tier=None if residency is None else residency.tier,
                path=self._path_for_tiers(tiers),
                payload_bytes=observation.required_bytes,
                tokens=observation.external_matched_tokens,
                state=None if residency is None else residency.state.value,
                generation=None if residency is None else residency.generation,
                layout_fingerprint=observation.layout_fingerprint,
                candidate_estimates_ns=(),
                chosen_action="fetch" if result.status == "ok" else "recompute",
                reason_code=(
                    "cancelled" if result.status == "cancelled" else "transfer_error"
                ),
                reason=result.reason,
                queue_ns=None,
                cuda_estimate_ns=None,
                modeled_estimate_ns=None,
                recompute_estimate_ns=observation.recompute_estimate_ns,
                cuda_actual_ns=None,
                modeled_actual_ns=None,
                effective_actual_ns=None,
                ticket_id=None if ticket is None else ticket.op_id,
                fallback_from=result.fallback_from,
                fallback_to=result.fallback_to,
                required=None,
                partial_success=False,
                invalidated_blocks=invalidated_blocks,
                terminal_error=None if result.status == "ok" else result.reason,
            )
        )

    def _record_store_terminal(
        self,
        request: VLLMTransferRequest,
        primary: ObjectKey,
        plan: TransferPlan,
        completion: CompositeCompletion,
        state: Literal["error", "cancelled"],
    ) -> None:
        self._record_event(
            op_id=plan.op_id,
            instance_id=request.instance_id,
            object_key=primary,
            direction="store",
            extent=plan.extent,
            generation=plan.extent.generation,
            lease_id=None,
            payload_checksum=request.payload_checksum_expected,
            cuda_elapsed_ns=completion.cuda_elapsed_ns,
            state=state,
            terminal=True,
            completion=completion,
        )

    def _execute_plan(
        self, parent_op_id: str, plan: TransferPlan, tier: Tier = "cxl"
    ) -> CompositeCompletion:
        coordinator = self._modeled_coordinator
        if tier == "dram" or coordinator is None:
            return self._model_client.join(self._require_executor(tier).submit(plan))
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
                launch=lambda: self._require_executor(tier).submit(plan),
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

    def _require_directory(self) -> MultiResidencyDirectory:
        if self._directory is None:
            raise RuntimeError("no engine layout is registered")
        return self._directory

    def _require_ticket_manager(self) -> TicketManager:
        if self._ticket_manager is None:
            raise RuntimeError("no engine layout is registered")
        return self._ticket_manager

    def _require_lookup_coordinator(self) -> LookupCoordinator:
        if self._lookup_coordinator is None:
            raise RuntimeError("Gate E lookup coordinator is not initialized")
        return self._lookup_coordinator

    def _require_placement_policy(self) -> CostAwarePolicy:
        if self._placement_policy is None:
            raise RuntimeError("Gate E placement policy is not initialized")
        return self._placement_policy

    def _store_targets(self) -> tuple[TargetSpec, ...]:
        mode = self._config.policy_store_mode
        if mode == "cxl_required":
            return (TargetSpec("cxl", True, "configured CXL residency"),)
        if mode == "dram_required":
            return (TargetSpec("dram", True, "configured DRAM residency"),)
        if mode == "dram_required_cxl_optional":
            return (
                TargetSpec("dram", True, "required local residency"),
                TargetSpec("cxl", False, "optional shared residency"),
            )
        if mode == "both_required":
            return (
                TargetSpec("dram", True, "required local residency"),
                TargetSpec("cxl", True, "required shared residency"),
            )
        raise RuntimeError(f"unsupported STORE policy mode: {mode}")

    @staticmethod
    def _path_for_tiers(tiers: frozenset[Tier]) -> DirectPath:
        if tiers == frozenset({"dram"}):
            return "dram_direct"
        if tiers == frozenset({"cxl"}) or not tiers:
            return "cxl_direct"
        return "multi_direct"

    def _tier_observations(self) -> tuple[TierObservation, ...]:
        observations = []
        for tier, manager in sorted(self._region_managers.items()):
            if tier == "dram":
                bandwidth = self._config.policy_dram_bandwidth_bytes_per_s
                latency_ns = 0
            else:
                bandwidth = self._config.policy_cxl_bandwidth_bytes_per_s
                latency_ns = self._config.policy_cxl_latency_ns
            observations.append(
                TierObservation(
                    tier,
                    manager.capacity_bytes,
                    manager.used_bytes,
                    self._require_ticket_manager().queue_bytes(tier),
                    bandwidth,
                    latency_ns,
                )
            )
        return tuple(observations)

    def _retire_ticket(
        self,
        ticket: TransferTicket,
        *,
        outcome: Literal["ok", "error"] | None = None,
        cancelled_reason: str | None = None,
    ) -> None:
        if (outcome is None) == (cancelled_reason is None):
            raise ValueError("ticket retirement requires one terminal outcome")
        try:
            primary = (
                self._require_directory().get_residency(ticket.residency_id).object_key
            )
        except KeyError:
            primary = None
        if cancelled_reason is not None:
            self._require_ticket_manager().cancel(ticket, cancelled_reason)
        else:
            if outcome is None:
                raise ValueError("ticket outcome must be supplied")
            self._require_ticket_manager().complete(ticket, outcome)
        try:
            self._require_directory().reclaim(ticket.residency_id)
        except (KeyError, RuntimeError):
            return
        if primary is not None:
            self._remove_aliases(primary)

    def _require_executor(self, tier: Tier = "cxl") -> LMCacheIPCTransferExecutor:
        try:
            return self._executors[tier]
        except KeyError as error:
            raise RuntimeError(f"no {tier} engine layout is registered") from error

    def _require_layout(self) -> Any:
        if self._layout is None:
            raise RuntimeError("no engine layout is registered")
        return self._layout

    def _require_layout_fingerprint(self) -> str:
        return layout_fingerprint(self._require_layout())

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

    def _remove_aliases(self, primary: ObjectKey) -> None:
        aliases = self._aliases_by_primary.pop(primary, ())
        for alias in aliases:
            self._primary_by_alias.pop(alias, None)

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
        if self._lookup_ready(primary) is None:
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

    def _ready_residencies(self, object_key: ObjectKey) -> tuple[Residency, ...]:
        ready = (
            residency
            for residency in self._require_directory().list_residencies(object_key)
            if residency.state == ResidencyState.READY
        )
        return tuple(sorted(ready, key=lambda item: (item.tier != "dram", item.tier)))

    def _lookup_ready(self, object_key: ObjectKey) -> Residency | None:
        ready = self._ready_residencies(object_key)
        return None if not ready else ready[0]

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
