# SPDX-License-Identifier: Apache-2.0

"""LMCache engine module for fleet-wide HotPrefix state."""

# Future
from __future__ import annotations

# Standard
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, cast
import threading
import time
import uuid

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.hotprefix.admission import (
    HostAdmissionCandidate,
    HostAdmissionPolicy,
)
from lmcache.v1.multiprocess.hotprefix.global_tree import (
    GlobalHostPrefixTree,
    PrefixAccessObservation,
)
from lmcache.v1.multiprocess.hotprefix.residency import (
    HostResidencyDirectory,
    HostResidencyState,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.hotprefix import (
    HotPrefixAccessResponse,
    HotPrefixAdmissionResponse,
    HotPrefixHostCandidate,
    HotPrefixTransferTicket,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

_ResultT = TypeVar("_ResultT")


def _estimated_wire_bytes(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, (int, float, bool)) or value is None:
        return 8
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], int):
            return 8 * len(value)
        return sum(_estimated_wire_bytes(item) for item in value)
    return 16


class _MeasuredLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()

    def __enter__(self) -> None:
        started_ns = time.monotonic_ns()
        self._lock.acquire()
        self._local.wait_ns = time.monotonic_ns() - started_ns

    def __exit__(self, *_args: object) -> None:
        self._lock.release()

    def last_wait_ns(self) -> int:
        return int(getattr(self._local, "wait_ns", 0))


def _observe_control(
    method: str,
) -> Callable[[Callable[..., _ResultT]], Callable[..., _ResultT]]:
    def decorator(function: Callable[..., _ResultT]) -> Callable[..., _ResultT]:
        @wraps(function)
        def wrapper(self: "HotPrefixModule", *args: Any, **kwargs: Any) -> _ResultT:
            has_start_subscriber = self._event_bus.has_subscribers(
                EventType.HOTPREFIX_CONTROL_START
            )
            has_end_subscriber = self._event_bus.has_subscribers(
                EventType.HOTPREFIX_CONTROL_END
            )
            if not (has_start_subscriber or has_end_subscriber):
                return function(self, *args, **kwargs)
            supplied_operation_id = kwargs.get("control_operation_id")
            if supplied_operation_id is None and args and isinstance(args[-1], str):
                supplied_operation_id = args[-1]
            operation_id = str(supplied_operation_id or "")
            if not operation_id and has_start_subscriber:
                operation_id = f"hotprefix-{uuid.uuid4().hex}"
            started_ns = time.monotonic_ns()
            request_bytes = _estimated_wire_bytes(args) + _estimated_wire_bytes(kwargs)
            previous_operation_id = getattr(
                self._control_operation_local, "operation_id", ""
            )
            self._control_operation_local.operation_id = operation_id
            if has_start_subscriber:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_CONTROL_START,
                        session_id=operation_id,
                        metadata={"method": method, "request_bytes": request_bytes},
                    )
                )
            outcome = "success"
            response_bytes = 0
            try:
                result = function(self, *args, **kwargs)
                response_bytes = _estimated_wire_bytes(result)
                return result
            except Exception:
                outcome = "failure"
                raise
            finally:
                total_ns = time.monotonic_ns() - started_ns
                lock_wait_ns = self._lock.last_wait_ns()
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_CONTROL_END,
                        session_id=operation_id,
                        metadata={
                            "method": method,
                            "outcome": outcome,
                            "request_bytes": request_bytes,
                            "response_bytes": response_bytes,
                            "duration_ns": total_ns,
                            "lock_wait_ns": lock_wait_ns,
                            "handler_body_ns": max(0, total_ns - lock_wait_ns),
                        },
                    )
                )
                self._control_operation_local.operation_id = previous_operation_id

        return wrapper

    return decorator


class HotPrefixModule:
    """Own Global Host Prefix Trees keyed by model/cache namespace."""

    _gauge_target: ClassVar["HotPrefixModule | None"] = None
    _gauges_registered: ClassVar[bool] = False

    def __init__(
        self,
        ctx: MPCacheServerContext,
        *,
        max_value: int = 255,
        max_age: int = 255,
        aging_interval: int = 50,
        host_capacity_bytes: int = 1 << 30,
        frequency_threshold: int = 10,
        lease_ttl_seconds: float = 30.0,
        physical_publication_timeout_seconds: float = 5.0,
    ) -> None:
        if physical_publication_timeout_seconds <= 0:
            raise ValueError("physical_publication_timeout_seconds must be positive")
        self._ctx = ctx
        self._max_value = max_value
        self._max_age = max_age
        self._aging_interval = aging_interval
        self._host_capacity_bytes = host_capacity_bytes
        self._lease_ttl_seconds = lease_ttl_seconds
        self._physical_publication_timeout_seconds = (
            physical_publication_timeout_seconds
        )
        self._admission_policy = HostAdmissionPolicy(
            frequency_threshold=frequency_threshold
        )
        self._trees: dict[bytes, GlobalHostPrefixTree] = {}
        self._directories: dict[bytes, HostResidencyDirectory] = {}
        self._lock = _MeasuredLock()
        self._control_operation_local = threading.local()
        self._event_bus = get_event_bus()
        HotPrefixModule._gauge_target = self
        if (
            self._event_bus.has_subscribers(EventType.HOTPREFIX_DECISION)
            and not HotPrefixModule._gauges_registered
        ):
            HotPrefixModule._gauges_registered = True
            register_gauge(
                "lmcache.hotprefix",
                "lmcache_mp.hotprefix_residency_bytes",
                "HotPrefix logical residency bytes by state.",
                lambda: HotPrefixModule._logical_gauge("bytes"),
            )
            register_gauge(
                "lmcache.hotprefix",
                "lmcache_mp.hotprefix_generations",
                "HotPrefix logical generations by state.",
                lambda: HotPrefixModule._logical_gauge("generations"),
            )
            for metric_name, description, field_name in (
                (
                    "lmcache_mp.hotprefix_active_leases",
                    "Active HotPrefix generation read leases.",
                    "active_leases",
                ),
                (
                    "lmcache_mp.hotprefix_retained_keys",
                    "Physical L1 keys retained by HotPrefix generations.",
                    "retained_keys",
                ),
                (
                    "lmcache_mp.hotprefix_discarded_generations",
                    "Discarded physical HotPrefix generation tombstones.",
                    "discarded_generations",
                ),
                (
                    "lmcache_mp.hotprefix_failed_publications",
                    "Failed physical HotPrefix publications.",
                    "failed_publications",
                ),
                (
                    "lmcache_mp.hotprefix_invalidated_generations",
                    "Invalidated physical HotPrefix generations.",
                    "invalidated_generations",
                ),
            ):
                register_gauge(
                    "lmcache.hotprefix",
                    metric_name,
                    description,
                    lambda field_name=field_name: HotPrefixModule._scalar_gauge(
                        field_name
                    ),
                )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return the HotPrefix control handlers.

        Returns:
            Handler specifications registered by the MP server.
        """
        return [
            HandlerSpec(
                RequestType.HOT_PREFIX_ACCESS,
                self.access,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_ADMIT,
                self.admit,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_PUBLISH,
                self.publish,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_ABORT,
                self.abort,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_CANDIDATES,
                self.candidates,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_ACQUIRE,
                self.acquire,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_RELEASE,
                self.release,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_RENEW,
                self.renew,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.HOT_PREFIX_INVALIDATE,
                self.invalidate,
                ThreadPoolType.NORMAL,
            ),
        ]

    @_observe_control("access")
    def access(
        self,
        instance_id: int,
        local_event_seq: int,
        namespace: bytes,
        token_ids: list[int],
        matched_tokens: int,
        control_operation_id: str = "",
    ) -> HotPrefixAccessResponse:
        """Commit one request's prefix observation.

        Args:
            instance_id: Non-negative serving instance identity.
            local_event_seq: Positive sequence number within the instance.
            namespace: Stable model/cache namespace.
            token_ids: Complete request token path.
            matched_tokens: Tokens reported by local native APC.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            Idempotent global epoch and canonical path response.
        """
        observation = PrefixAccessObservation(
            instance_id,
            local_event_seq,
            tuple(token_ids),
            matched_tokens,
        )
        with self._lock:
            self._reconcile_physical_invalidations()
            tree = self._trees.get(namespace)
            if tree is None:
                tree = GlobalHostPrefixTree(
                    namespace=namespace,
                    max_value=self._max_value,
                    max_age=self._max_age,
                    aging_interval=self._aging_interval,
                )
                self._trees[namespace] = tree
            result = tree.observe(observation)
            if self._event_bus.has_subscribers(EventType.HOTPREFIX_DECISION):
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_DECISION,
                        session_id=self._current_control_operation_id(),
                        metadata={
                            "kind": "access",
                            "action": "observe",
                            "reason": "none",
                            "tokens": len(token_ids),
                            "bytes": 0,
                            "global_epoch": result.epoch,
                        },
                    )
                )
        return HotPrefixAccessResponse(
            result.epoch,
            result.global_matched_tokens,
            list(result.path),
        )

    @_observe_control("admit")
    def admit(
        self,
        namespace: bytes,
        prefix_id: bytes,
        size_bytes: int,
        generation: int,
        control_operation_id: str = "",
    ) -> HotPrefixAdmissionResponse:
        """Reserve Host capacity using server-authoritative Global Hotness.

        Args:
            namespace: Canonical model/revision/layout/cache-salt namespace.
            prefix_id: Evicted LogicalPrefix identifier.
            size_bytes: Complete physical payload size on this server.
            generation: Scheduler-proposed cross-server generation.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            Explainable DEDUP, REJECT, or ACCEPT response.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            tree = self._trees.get(namespace)
            snapshot = tree.get(prefix_id) if tree is not None else None
            if snapshot is None:
                if self._event_bus.has_subscribers(EventType.HOTPREFIX_DECISION):
                    self._event_bus.publish(
                        Event(
                            event_type=EventType.HOTPREFIX_DECISION,
                            session_id=self._current_control_operation_id(),
                            metadata={
                                "kind": "admission",
                                "action": "reject",
                                "reason": "prefix_absent",
                                "tokens": 0,
                                "bytes": size_bytes,
                            },
                        )
                    )
                return HotPrefixAdmissionResponse(
                    "reject", "prefix_absent_from_global_tree", [], None
                )
            assert tree is not None
            candidate = HostAdmissionCandidate(
                prefix_id,
                size_bytes,
                snapshot.frequency,
                snapshot.clock,
            )
            directory = self._directory(namespace)
            hotness_by_prefix = {
                item.prefix_id: (item.frequency, item.clock) for item in tree.snapshot()
            }
            decision = directory.reserve(
                candidate,
                self._admission_policy,
                hotness_by_prefix=hotness_by_prefix,
                generation=generation,
            )
            residency = directory.get(prefix_id)
            reserved_generation = (
                residency.generation if residency is not None else None
            )
            self._retire_invalid_physical_generations()
            if self._event_bus.has_subscribers(EventType.HOTPREFIX_DECISION):
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_DECISION,
                        session_id=self._current_control_operation_id(),
                        metadata={
                            "kind": "admission",
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "tokens": 0,
                            "bytes": size_bytes,
                            "global_frequency": snapshot.frequency,
                            "global_clock": snapshot.clock,
                        },
                    )
                )
        return HotPrefixAdmissionResponse(
            decision.action.value,
            decision.reason,
            list(decision.evict_prefixes),
            reserved_generation,
        )

    @_observe_control("publish")
    def publish(
        self,
        namespace: bytes,
        prefix_id: bytes,
        control_operation_id: str = "",
    ) -> bool:
        """Publish a fully written shared Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Reserved LogicalPrefix identifier.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            ``True`` after publication (including an idempotent retry).
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            directory = self._directory(namespace)
            residency = directory.get(prefix_id)
            if residency is None:
                return False
            storage_manager = self._physical_storage_manager()
            if storage_manager is not None:
                if not storage_manager.wait_for_residency(
                    prefix_id,
                    residency.generation,
                    self._physical_publication_timeout_seconds,
                ):
                    return False
                if not storage_manager.pin_generation(prefix_id, residency.generation):
                    return False
            victims = directory.replacement_victims(prefix_id)
            ready = directory.publish(prefix_id)
            if self._event_bus.has_subscribers(EventType.HOTPREFIX_RESIDENCY_CHANGED):
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_RESIDENCY_CHANGED,
                        session_id=self._current_control_operation_id(),
                        metadata={
                            "old_state": "reserved",
                            "new_state": "ready",
                            "bytes": ready.size_bytes,
                            "shared_keys": 0,
                        },
                    )
                )
            if storage_manager is not None:
                for victim in victims:
                    storage_manager.evict_generation(
                        victim.prefix_id, victim.generation
                    )
            self._retire_invalid_physical_generations()
        return True

    @_observe_control("abort")
    def abort(
        self,
        namespace: bytes,
        prefix_id: bytes,
        control_operation_id: str = "",
    ) -> bool:
        """Abort an incomplete shared Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Reserved LogicalPrefix identifier.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            ``True`` after rollback (including an idempotent retry).
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            directory = self._directory(namespace)
            residency = directory.get(prefix_id)
            storage_manager = self._physical_storage_manager()
            if (
                storage_manager is not None
                and residency is not None
                and residency.state is HostResidencyState.RESERVED
            ):
                storage_manager.evict_generation(prefix_id, residency.generation)
            directory.abort(prefix_id)
            if residency is not None and self._event_bus.has_subscribers(
                EventType.HOTPREFIX_RESIDENCY_CHANGED
            ):
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_RESIDENCY_CHANGED,
                        session_id=self._current_control_operation_id(),
                        metadata={
                            "old_state": residency.state.value,
                            "new_state": "absent",
                            "bytes": residency.size_bytes,
                            "shared_keys": 0,
                        },
                    )
                )
            self._retire_invalid_physical_generations()
        return True

    @_observe_control("candidates")
    def candidates(
        self,
        namespace: bytes,
        prefix_ids: list[bytes],
        control_operation_id: str = "",
    ) -> list[HotPrefixHostCandidate]:
        """Return READY Host sources among target-local hot prefixes.

        Args:
            namespace: Residency namespace.
            prefix_ids: Target-local candidates in hotness order.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            READY generations available in the requested set.
        """
        requested = set(prefix_ids)
        with self._lock:
            self._reconcile_physical_invalidations()
            directory = self._directories.get(namespace)
            if directory is None:
                return []
            residencies = directory.snapshot()
            tree = self._trees.get(namespace)
            current = (
                {item.prefix_id: item for item in tree.snapshot()}
                if tree is not None
                else {}
            )
        candidates: list[HotPrefixHostCandidate] = []
        for item in residencies:
            if (
                item.state is not HostResidencyState.READY
                or item.prefix_id not in requested
            ):
                continue
            global_node = current.get(item.prefix_id)
            frequency = (
                global_node.frequency if global_node is not None else item.frequency
            )
            clock = global_node.clock if global_node is not None else item.clock
            candidates.append(
                HotPrefixHostCandidate(
                    item.prefix_id,
                    item.size_bytes,
                    item.generation,
                    frequency,
                    clock,
                )
            )
        return candidates

    @_observe_control("acquire")
    def acquire(
        self,
        namespace: bytes,
        prefix_id: bytes,
        generation: int,
        ticket_id: bytes,
        control_operation_id: str = "",
    ) -> HotPrefixTransferTicket:
        """Bind a promotion read to an immutable READY generation.

        Args:
            namespace: Residency namespace.
            prefix_id: Promotion source LogicalPrefix.
            generation: Exact READY generation.
            ticket_id: Cross-server idempotency key.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            Renewable generation-bound transfer ticket.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            storage_manager = self._physical_storage_manager()
            if storage_manager is not None and not storage_manager.pin_generation(
                prefix_id, generation
            ):
                raise RuntimeError("promotion source physical generation is missing")
            lease = self._directory(namespace).acquire(
                prefix_id,
                generation,
                ticket_id,
            )
        return HotPrefixTransferTicket(
            lease.ticket_id,
            lease.prefix_id,
            lease.generation,
            lease.size_bytes,
        )

    @_observe_control("release")
    def release(
        self,
        namespace: bytes,
        ticket_id: bytes,
        control_operation_id: str = "",
    ) -> bool:
        """Release a terminal promotion's shared Host read lease.

        Args:
            namespace: Residency namespace.
            ticket_id: Transfer ticket identifier.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            ``True`` after release or an idempotent retry.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            self._directory(namespace).release(ticket_id, missing_ok=True)
            self._retire_invalid_physical_generations()
        return True

    @_observe_control("renew")
    def renew(
        self,
        namespace: bytes,
        ticket_id: bytes,
        control_operation_id: str = "",
    ) -> bool:
        """Renew an in-flight promotion lease.

        Args:
            namespace: Residency namespace.
            ticket_id: Active transfer ticket identifier.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            ``True`` when the lease was extended.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            self._directory(namespace).renew(ticket_id)
        return True

    @_observe_control("invalidate")
    def invalidate(
        self,
        namespace: bytes,
        prefix_id: bytes,
        generation: int,
        control_operation_id: str = "",
    ) -> bool:
        """Invalidate a READY generation after a verified physical miss.

        Args:
            namespace: Residency namespace.
            prefix_id: LogicalPrefix whose payload was absent.
            generation: Exact failed generation.
            control_operation_id: Client-generated trace join identifier.

        Returns:
            ``True`` only when that generation was removed.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            invalidated = self._directory(namespace).invalidate(prefix_id, generation)
            if invalidated and self._event_bus.has_subscribers(
                EventType.HOTPREFIX_RESIDENCY_CHANGED
            ):
                self._event_bus.publish(
                    Event(
                        event_type=EventType.HOTPREFIX_RESIDENCY_CHANGED,
                        session_id=self._current_control_operation_id(),
                        metadata={
                            "old_state": "ready",
                            "new_state": "invalid",
                            "bytes": 0,
                            "shared_keys": 0,
                        },
                    )
                )
            self._retire_invalid_physical_generations()
            return invalidated

    def report_status(self) -> dict[str, int]:
        """Return global tree and node counts.

        Returns:
            Counts suitable for the MP server status report.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            trees = tuple(self._trees.values())
            residencies = tuple(
                item
                for directory in self._directories.values()
                for item in directory.snapshot()
            )
            physical = self._physical_storage_manager()
            physical_stats = (
                physical.hotprefix_physical_stats() if physical is not None else None
            )
            return {
                "hotprefix_trees": len(trees),
                "hotprefix_nodes": sum(len(tree.snapshot()) for tree in trees),
                "hotprefix_generations": len(residencies),
                "hotprefix_residency_bytes": sum(
                    item.size_bytes for item in residencies
                ),
                "hotprefix_active_leases": sum(
                    directory.active_lease_count
                    for directory in self._directories.values()
                ),
                "hotprefix_retained_keys": (
                    physical_stats.retained_keys if physical_stats is not None else 0
                ),
                "hotprefix_discarded_generations": (
                    physical_stats.discarded_generations
                    if physical_stats is not None
                    else 0
                ),
            }

    def close(self) -> None:
        """Release all module-owned tree, residency, and lease state."""
        with self._lock:
            self._trees.clear()
            self._directories.clear()

    @staticmethod
    def _logical_gauge(
        kind: str,
    ) -> list[tuple[int | float, dict[str, object]]]:
        target = HotPrefixModule._gauge_target
        totals = {state.value: 0 for state in HostResidencyState}
        if target is not None:
            with target._lock:
                for directory in target._directories.values():
                    for residency in directory.snapshot():
                        totals[residency.state.value] += (
                            residency.size_bytes if kind == "bytes" else 1
                        )
        return [(value, {"state": state}) for state, value in totals.items()]

    def _current_control_operation_id(self) -> str:
        return str(getattr(self._control_operation_local, "operation_id", ""))

    @staticmethod
    def _scalar_gauge(field_name: str) -> int:
        target = HotPrefixModule._gauge_target
        if target is None:
            return 0
        if field_name == "active_leases":
            with target._lock:
                return sum(
                    directory.active_lease_count
                    for directory in target._directories.values()
                )
        storage_manager = target._physical_storage_manager()
        if storage_manager is None:
            return 0
        stats = storage_manager.hotprefix_physical_stats()
        return int(getattr(stats, field_name))

    def _directory(self, namespace: bytes) -> HostResidencyDirectory:
        directory = self._directories.get(namespace)
        if directory is None:
            directory = HostResidencyDirectory(
                capacity_bytes=self._host_capacity_bytes,
                lease_ttl_seconds=self._lease_ttl_seconds,
            )
            self._directories[namespace] = directory
        return directory

    def _physical_storage_manager(self) -> StorageManager | None:
        storage_manager = getattr(self._ctx, "storage_manager", None)
        return cast("StorageManager | None", storage_manager)

    def _reconcile_physical_invalidations(self) -> None:
        for directory in self._directories.values():
            directory.expire_leases()
        storage_manager = self._physical_storage_manager()
        if storage_manager is None:
            return
        for prefix_id, generation in storage_manager.take_invalidated_generations():
            matched = False
            for directory in self._directories.values():
                residency = directory.get(prefix_id)
                if residency is not None and residency.generation == generation:
                    matched = True
                    if residency.state is HostResidencyState.RESERVED:
                        directory.abort(prefix_id)
                        storage_manager.evict_generation(prefix_id, generation)
                    else:
                        directory.invalidate(prefix_id, generation)
                    continue
                if directory.invalidate_replacement_victim(prefix_id, generation):
                    matched = True
            if not matched:
                storage_manager.evict_generation(prefix_id, generation)
        self._retire_invalid_physical_generations()

    def _retire_invalid_physical_generations(self) -> None:
        storage_manager = self._physical_storage_manager()
        if storage_manager is None:
            return
        for directory in self._directories.values():
            for residency in directory.take_retired_invalid():
                storage_manager.evict_generation(
                    residency.prefix_id, residency.generation
                )
