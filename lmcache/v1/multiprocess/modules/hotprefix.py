# SPDX-License-Identifier: Apache-2.0

"""LMCache engine module for fleet-wide HotPrefix state."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, cast
import threading

# First Party
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


class HotPrefixModule:
    """Own Global Host Prefix Trees keyed by model/cache namespace."""

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
        self._lock = threading.Lock()

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

    def access(
        self,
        instance_id: int,
        local_event_seq: int,
        namespace: bytes,
        token_ids: list[int],
        matched_tokens: int,
    ) -> HotPrefixAccessResponse:
        """Commit one request's prefix observation.

        Args:
            instance_id: Non-negative serving instance identity.
            local_event_seq: Positive sequence number within the instance.
            namespace: Stable model/cache namespace.
            token_ids: Complete request token path.
            matched_tokens: Tokens reported by local native APC.

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
        return HotPrefixAccessResponse(
            result.epoch,
            result.global_matched_tokens,
            list(result.path),
        )

    def admit(
        self,
        namespace: bytes,
        prefix_id: bytes,
        size_bytes: int,
        generation: int,
    ) -> HotPrefixAdmissionResponse:
        """Reserve Host capacity using server-authoritative Global Hotness.

        Args:
            namespace: Canonical model/revision/layout/cache-salt namespace.
            prefix_id: Evicted LogicalPrefix identifier.
            size_bytes: Complete physical payload size on this server.
            generation: Scheduler-proposed cross-server generation.

        Returns:
            Explainable DEDUP, REJECT, or ACCEPT response.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            tree = self._trees.get(namespace)
            snapshot = tree.get(prefix_id) if tree is not None else None
            if snapshot is None:
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
        return HotPrefixAdmissionResponse(
            decision.action.value,
            decision.reason,
            list(decision.evict_prefixes),
            reserved_generation,
        )

    def publish(self, namespace: bytes, prefix_id: bytes) -> bool:
        """Publish a fully written shared Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Reserved LogicalPrefix identifier.

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
            directory.publish(prefix_id)
            if storage_manager is not None:
                for victim in victims:
                    storage_manager.evict_generation(
                        victim.prefix_id, victim.generation
                    )
            self._retire_invalid_physical_generations()
        return True

    def abort(self, namespace: bytes, prefix_id: bytes) -> bool:
        """Abort an incomplete shared Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Reserved LogicalPrefix identifier.

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
            self._retire_invalid_physical_generations()
        return True

    def candidates(
        self, namespace: bytes, prefix_ids: list[bytes]
    ) -> list[HotPrefixHostCandidate]:
        """Return READY Host sources among target-local hot prefixes.

        Args:
            namespace: Residency namespace.
            prefix_ids: Target-local candidates in hotness order.

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

    def acquire(
        self,
        namespace: bytes,
        prefix_id: bytes,
        generation: int,
        ticket_id: bytes,
    ) -> HotPrefixTransferTicket:
        """Bind a promotion read to an immutable READY generation.

        Args:
            namespace: Residency namespace.
            prefix_id: Promotion source LogicalPrefix.
            generation: Exact READY generation.
            ticket_id: Cross-server idempotency key.

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

    def release(self, namespace: bytes, ticket_id: bytes) -> bool:
        """Release a terminal promotion's shared Host read lease.

        Args:
            namespace: Residency namespace.
            ticket_id: Transfer ticket identifier.

        Returns:
            ``True`` after release or an idempotent retry.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            self._directory(namespace).release(ticket_id, missing_ok=True)
            self._retire_invalid_physical_generations()
        return True

    def renew(self, namespace: bytes, ticket_id: bytes) -> bool:
        """Renew an in-flight promotion lease.

        Args:
            namespace: Residency namespace.
            ticket_id: Active transfer ticket identifier.

        Returns:
            ``True`` when the lease was extended.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            self._directory(namespace).renew(ticket_id)
        return True

    def invalidate(self, namespace: bytes, prefix_id: bytes, generation: int) -> bool:
        """Invalidate a READY generation after a verified physical miss.

        Args:
            namespace: Residency namespace.
            prefix_id: LogicalPrefix whose payload was absent.
            generation: Exact failed generation.

        Returns:
            ``True`` only when that generation was removed.
        """
        with self._lock:
            self._reconcile_physical_invalidations()
            invalidated = self._directory(namespace).invalidate(prefix_id, generation)
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
            return {
                "hotprefix_trees": len(trees),
                "hotprefix_nodes": sum(len(tree.snapshot()) for tree in trees),
            }

    def close(self) -> None:
        """Release all module-owned tree, residency, and lease state."""
        with self._lock:
            self._trees.clear()
            self._directories.clear()

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
