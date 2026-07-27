# SPDX-License-Identifier: Apache-2.0
"""Gate B orchestration for immutable GPU-visible CXL proxy objects."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal
import threading
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.config import CXLSharedTierConfig
from lmcache.v1.multiprocess.cxl.completion import NoopModelClient
from lmcache.v1.multiprocess.cxl.contracts import CompositeCompletion
from lmcache.v1.multiprocess.cxl.cuda_executor import (
    LMCacheIPCTransferExecutor,
    RegisteredRegionView,
)
from lmcache.v1.multiprocess.cxl.data_plane_adapter import (
    VLLMDataPlaneAdapter,
    VLLMTransferRequest,
)
from lmcache.v1.multiprocess.cxl.layout import layout_fingerprint
from lmcache.v1.multiprocess.cxl.region_manager import CXLRegionManager
from lmcache.v1.multiprocess.cxl.region_provider import (
    PosixShmRegionProvider,
    RegionHandle,
)
from lmcache.v1.multiprocess.cxl.residency import (
    ReadLease,
    Residency,
    SingleResidencyDirectory,
)


@dataclass(frozen=True)
class CXLTransferResult:
    """Observable terminal result of one shared-tier request."""

    status: Literal["ok", "miss", "error"]
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
        if self.status not in ("ok", "miss", "error"):
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


class CXLSharedTierModule:
    """Coordinate immutable CXL STORE and RETRIEVE for one LMCache server."""

    _LEASE_TTL_NS = 30_000_000_000

    def __init__(
        self,
        config: CXLSharedTierConfig,
        provider: PosixShmRegionProvider,
        handle: RegionHandle,
        *,
        native_ops: Any | None,
        runtime: Any | None,
        clock_ns: Callable[[], int],
    ) -> None:
        self._config = config
        self._provider = provider
        self._handle = handle
        self._native_ops = native_ops
        self._runtime = runtime
        self._clock_ns = clock_ns
        self._region_manager: CXLRegionManager | None = None
        self._region_view: RegisteredRegionView | None = None
        self._executor: LMCacheIPCTransferExecutor | None = None
        self._directory: SingleResidencyDirectory | None = None
        self._layout: Any | None = None
        self._adapters: dict[int, VLLMDataPlaneAdapter] = {}
        self._primary_by_alias: dict[ObjectKey, ObjectKey] = {}
        self._aliases_by_primary: dict[ObjectKey, tuple[ObjectKey, ...]] = {}
        self._model_client = NoopModelClient()
        self._closed = False
        self._lock = threading.RLock()

    @classmethod
    def open(
        cls,
        config: CXLSharedTierConfig,
        *,
        native_ops: Any | None = None,
        runtime: Any | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> CXLSharedTierModule:
        """Open and validate the configured proxy region without fallback.

        Args:
            config: Complete enabled shared-tier configuration.
            native_ops: Optional replacement for the CUDA/native boundary.
            runtime: Optional replacement for CUDA timing and execution.
            clock_ns: Monotonic clock used for completions and leases.

        Returns:
            An open shared-tier module awaiting engine registration.

        Raises:
            ValueError: If the feature is disabled or provider is unsupported.
            OSError: If the configured POSIX SHM object cannot be opened.
            RuntimeError: If the region header or capacity is incompatible.
        """
        if not config.enabled:
            raise ValueError("CXL shared tier is disabled")
        if config.provider != "posix_shm":
            raise ValueError("Gate B supports only the posix_shm provider")
        if config.shm_name is None or config.capacity_bytes is None:
            raise ValueError("enabled CXL shared tier configuration is incomplete")
        provider = PosixShmRegionProvider(
            region_id=f"cxl-posix:{config.shm_name}",
            shm_name=config.shm_name,
            expected_capacity=config.capacity_bytes,
        )
        handle = provider.provision()
        return cls(
            config,
            provider,
            handle,
            native_ops=native_ops,
            runtime=runtime,
            clock_ns=clock_ns,
        )

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
        with self._lock:
            self._adapters.pop(instance_id, None)
            if self._executor is not None:
                self._executor.unregister_engine(instance_id)

    def store(self, request: VLLMTransferRequest) -> CXLTransferResult:
        """Store independently allocated chunks and publish after CUDA success.

        Args:
            request: Complete engine transfer metadata for one STORE request.

        Returns:
            A terminal direct-path result. CUDA errors abort every unpublished
            residency and never fall back to DRAM staging.
        """
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            chunks = adapter.split_chunks(request)
            payload_bytes = sum(adapter.packed_size_bytes(chunk) for chunk in chunks)
            pending_chunks: list[VLLMTransferRequest] = []
            pending: list[_PendingStore] = []
            completions: list[CompositeCompletion] = []
            try:
                for chunk in chunks:
                    if self._existing_primary(chunk.object_keys) is None:
                        pending_chunks.append(chunk)
                new_aliases = [
                    alias for chunk in pending_chunks for alias in chunk.object_keys
                ]
                if len(new_aliases) != len(set(new_aliases)):
                    raise RuntimeError(
                        "ObjectKey alias appears in multiple new CXL chunks"
                    )

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
                    plan = adapter.build_store_plan(chunk, writing.descriptor)
                    data_completion = self._require_executor().submit(plan)
                    completion = self._model_client.join(data_completion)
                    completions.append(completion)
                    if completion.cuda_status != "ok":
                        raise RuntimeError(data_completion.error or "CUDA store failed")
                for item in pending:
                    self._require_directory().publish(item.residency_id)
                    self._publish_aliases(item.primary_key, item.aliases)
                return self._result(
                    "ok", tuple(completions), payload_bytes=payload_bytes
                )
            except Exception as error:
                for item in pending:
                    if self._require_directory().lookup_ready(item.primary_key) is None:
                        try:
                            self._require_directory().abort(
                                item.residency_id, str(error) or "CXL store failed"
                            )
                        except (KeyError, RuntimeError):
                            pass
                diagnostic = str(error) or self._completion_error(completions)
                return self._result(
                    "error",
                    tuple(completions),
                    diagnostic,
                    payload_bytes=payload_bytes,
                )

    def retrieve(self, request: VLLMTransferRequest) -> CXLTransferResult:
        """Restore READY chunks directly into HBM under read leases.

        Args:
            request: Complete engine transfer metadata for one RETRIEVE.

        Returns:
            OK, MISS, or ERROR without allocating a DRAM staging object.
        """
        with self._lock:
            adapter = self._adapter_for(request.instance_id)
            all_chunks = adapter.split_chunks(request)
            chunks = self._retrievable_chunks(all_chunks, request)
            payload_bytes = sum(adapter.packed_size_bytes(chunk) for chunk in chunks)
            ready: list[tuple[VLLMTransferRequest, Residency, ObjectKey]] = []
            for chunk in chunks:
                primary = self._primary_for_chunk(chunk.object_keys)
                if primary is None:
                    return self._result("miss", payload_bytes=payload_bytes)
                residency = self._require_directory().lookup_ready(primary)
                if residency is None or residency.descriptor is None:
                    return self._result("miss", payload_bytes=payload_bytes)
                ready.append((chunk, residency, primary))

            leases: list[ReadLease] = []
            completions: list[CompositeCompletion] = []
            try:
                for chunk, residency, primary in ready:
                    lease = self._require_directory().acquire_read(
                        primary,
                        ttl_ns=self._LEASE_TTL_NS,
                        now_ns=self._clock_ns(),
                    )
                    leases.append(lease)
                    if lease.generation != residency.generation:
                        raise RuntimeError("read lease generation changed")
                    descriptor = residency.descriptor
                    if descriptor is None:
                        raise RuntimeError("READY residency has no extent")
                    plan = adapter.build_retrieve_plan(chunk, descriptor)
                    data_completion = self._require_executor().submit(plan)
                    completion = self._model_client.join(data_completion)
                    completions.append(completion)
                    if completion.cuda_status != "ok":
                        raise RuntimeError(
                            data_completion.error or "CUDA retrieve failed"
                        )
                return self._result(
                    "ok", tuple(completions), payload_bytes=payload_bytes
                )
            except Exception as error:
                diagnostic = str(error) or self._completion_error(completions)
                return self._result(
                    "error",
                    tuple(completions),
                    diagnostic,
                    payload_bytes=payload_bytes,
                )
            finally:
                for lease in leases:
                    self._require_directory().release_read(
                        lease.lease_id, now_ns=self._clock_ns()
                    )

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
        with self._lock:
            if self._closed:
                return
            for instance_id in tuple(self._adapters):
                self.unregister_engine(instance_id)
            if self._region_view is not None:
                self._region_view.close()
            self._provider.close()
            self._closed = True

    @staticmethod
    def _result(
        status: Literal["ok", "miss", "error"],
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
        if self._closed:
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
