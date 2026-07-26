# SPDX-License-Identifier: Apache-2.0
"""CXLMemSim bulk-shm L2 adapter for LMCache MP."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast
import os
import threading

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.cxl_memsim_client import (
    BulkClientStats,
    BulkTransferResult,
    CxlMemSimClient,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.platform.event_notifier import EventNotifier

logger = init_logger(__name__)


class _BulkClient(Protocol):
    @property
    def capacity(self) -> int: ...

    def write_from(
        self,
        offset: int,
        src_ptr: int,
        size: int,
    ) -> BulkTransferResult: ...

    def read_into(
        self,
        offset: int,
        dst_ptr: int,
        size: int,
    ) -> BulkTransferResult: ...

    def snapshot_stats(self) -> BulkClientStats: ...

    def close(self) -> None: ...


class _BulkClientFactory(Protocol):
    def __call__(
        self,
        *,
        library_path: str,
        control_name: str,
        timeout_ms: int,
    ) -> _BulkClient: ...


@dataclass
class _SlotEntry:
    slot_id: int
    payload_bytes: int
    cached_positions: torch.Tensor | None
    external_locks: int = 0
    read_borrows: int = 0
    pending_free: bool = False


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


class CxlMemSimL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the CXLMemSim bulk-shm MP L2 adapter."""

    def __init__(
        self,
        *,
        client_library: str,
        slot_bytes: int,
        control_name: str = "/cxlmemsim_bulk",
        offset_bytes: int = 0,
        capacity_bytes: int | None = None,
        timeout_ms: int = 5000,
        num_store_workers: int = 1,
        num_lookup_workers: int = 1,
        num_load_workers: int = min(4, os.cpu_count() or 1),
    ) -> None:
        """Initialize a validated adapter configuration.

        Args:
            client_library: Path to ``libcxlmemsim_client.so``.
            slot_bytes: Fixed bytes reserved for each stored object.
            control_name: CXLMemSim bulk-shm control object name.
            offset_bytes: First byte of the adapter-owned server region.
            capacity_bytes: Optional size of the adapter-owned region.
            timeout_ms: Native open and transfer timeout in milliseconds.
            num_store_workers: Store worker count.
            num_lookup_workers: Lookup worker count.
            num_load_workers: Load worker count.
        """
        self.client_library = _non_empty_string(client_library, "client_library")
        self.slot_bytes = _positive_int(slot_bytes, "slot_bytes")
        self.control_name = _non_empty_string(control_name, "control_name")
        self.offset_bytes = _non_negative_int(offset_bytes, "offset_bytes")
        self.capacity_bytes = (
            None
            if capacity_bytes is None
            else _positive_int(capacity_bytes, "capacity_bytes")
        )
        self.timeout_ms = _positive_int(timeout_ms, "timeout_ms")
        self.num_store_workers = _positive_int(
            num_store_workers,
            "num_store_workers",
        )
        self.num_lookup_workers = _positive_int(
            num_lookup_workers,
            "num_lookup_workers",
        )
        self.num_load_workers = _positive_int(num_load_workers, "num_load_workers")

    @classmethod
    def from_dict(cls, d: dict) -> CxlMemSimL2AdapterConfig:
        """Build an adapter config from a CLI JSON object.

        Args:
            d: Parsed ``--l2-adapter`` JSON object.

        Returns:
            Validated CXLMemSim adapter configuration.
        """
        return cls(
            client_library=d.get("client_library"),
            slot_bytes=d.get("slot_bytes"),
            control_name=d.get("control_name", "/cxlmemsim_bulk"),
            offset_bytes=d.get("offset_bytes", 0),
            capacity_bytes=d.get("capacity_bytes"),
            timeout_ms=d.get("timeout_ms", 5000),
            num_store_workers=d.get("num_store_workers", 1),
            num_lookup_workers=d.get("num_lookup_workers", 1),
            num_load_workers=d.get(
                "num_load_workers",
                min(4, os.cpu_count() or 1),
            ),
        )

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for this adapter type."""
        return (
            "CXLMemSim L2 adapter config fields:\n"
            "- client_library (str): path to libcxlmemsim_client.so\n"
            "- slot_bytes (int): fixed bytes per stored object\n"
            "- control_name (str): bulk-shm control name\n"
            "- offset_bytes (int): adapter region base offset\n"
            "- capacity_bytes (int): optional adapter region size\n"
            "- timeout_ms (int): native request timeout\n"
            "- num_store_workers/num_lookup_workers/num_load_workers (int)"
        )


class CxlMemSimL2Adapter(L2AdapterInterface):
    """Volatile fixed-slot key index backed by CXLMemSim bulk-shm."""

    def __init__(
        self,
        config: CxlMemSimL2AdapterConfig,
        *,
        client_factory: _BulkClientFactory = CxlMemSimClient,
    ) -> None:
        """Open the simulated CXL region and initialize worker resources.

        Args:
            config: Validated adapter configuration.
            client_factory: Native client constructor, injectable for tests.

        Raises:
            ValueError: If the configured arena is outside server capacity or
                cannot fit one complete slot.
        """
        self._config = config
        client = client_factory(
            library_path=config.client_library,
            control_name=config.control_name,
            timeout_ms=config.timeout_ms,
        )
        try:
            available = client.capacity - config.offset_bytes
            if available < 0:
                raise ValueError("offset_bytes exceeds CXLMemSim capacity")
            requested = config.capacity_bytes
            if requested is None:
                requested = available
            if requested > available:
                raise ValueError("configured arena exceeds CXLMemSim capacity")
            arena_bytes = requested // config.slot_bytes * config.slot_bytes
            if arena_bytes <= 0:
                raise ValueError("configured arena cannot fit one complete slot")
        except Exception:
            client.close()
            raise

        super().__init__(max_capacity_bytes=arena_bytes)
        self._client = client
        self._arena_bytes = arena_bytes
        self._max_slots = arena_bytes // config.slot_bytes
        self._lock = threading.Lock()
        self._state_condition = threading.Condition()
        self._entries: dict[ObjectKey, _SlotEntry] = {}
        self._inflight_stores: dict[ObjectKey, int] = {}
        self._free_slots = list(reversed(range(self._max_slots)))
        self._occupied_slots = 0
        self._next_task_id = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._inflight_store_tasks = 0
        self._inflight_lookup_tasks = 0
        self._inflight_load_tasks = 0
        self._closing = False
        self._closed = False
        self._store_efd: EventNotifier | None = None
        self._lookup_efd: EventNotifier | None = None
        self._load_efd: EventNotifier | None = None
        self._store_executor: ThreadPoolExecutor | None = None
        self._lookup_executor: ThreadPoolExecutor | None = None
        self._load_executor: ThreadPoolExecutor | None = None

        try:
            self._store_efd = create_event_notifier()
            self._lookup_efd = create_event_notifier()
            self._load_efd = create_event_notifier()
            self._store_executor = ThreadPoolExecutor(
                max_workers=config.num_store_workers,
                thread_name_prefix="cxl-memsim-l2-store",
            )
            self._lookup_executor = ThreadPoolExecutor(
                max_workers=config.num_lookup_workers,
                thread_name_prefix="cxl-memsim-l2-lookup",
            )
            self._load_executor = ThreadPoolExecutor(
                max_workers=config.num_load_workers,
                thread_name_prefix="cxl-memsim-l2-load",
            )
        except Exception:
            self.close()
            raise

    def get_store_event_fd(self) -> int:
        """Return the store completion event fd."""
        assert self._store_efd is not None
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the lookup completion event fd."""
        assert self._lookup_efd is not None
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the load completion event fd."""
        assert self._load_efd is not None
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous host-to-CXLMemSim store task."""
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")
        with self._lock:
            self._ensure_open_locked()
            task_id = self._next_task_id_locked()
            self._inflight_store_tasks += 1
        assert self._store_executor is not None
        self._store_executor.submit(
            self._execute_store_task,
            task_id,
            list(keys),
            list(objects),
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Drain completed store task results."""
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
            return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        """Submit an asynchronous lookup-and-lock task."""
        del layout_desc
        with self._lock:
            self._ensure_open_locked()
            task_id = self._next_task_id_locked()
            self._inflight_lookup_tasks += 1
        assert self._lookup_executor is not None
        self._lookup_executor.submit(self._execute_lookup_task, task_id, list(keys))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove one completed lookup result."""
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release external locks acquired by lookup tasks."""
        with self._state_condition:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and entry.external_locks > 0:
                    entry.external_locks -= 1

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a placeholder asynchronous load task."""
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")
        with self._lock:
            self._ensure_open_locked()
            task_id = self._next_task_id_locked()
            self._inflight_load_tasks += 1
        assert self._load_executor is not None
        self._load_executor.submit(self._finish_empty_load, task_id, len(keys))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove one completed load result."""
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def get_usage(self) -> AdapterUsage:
        """Return fixed-slot occupancy and per-salt accounting."""
        with self._state_condition:
            total_bytes_used = self._occupied_slots * self._config.slot_bytes
        base_usage = super().get_usage()
        return AdapterUsage(
            total_bytes_used=total_bytes_used,
            total_capacity_bytes=self._arena_bytes,
            bytes_by_cache_salt=MappingProxyType(dict(base_usage.bytes_by_cache_salt)),
        )

    def close(self) -> None:
        """Wait for submitted work and release native and event resources."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            store_executor = self._store_executor
            lookup_executor = self._lookup_executor
            load_executor = self._load_executor
            self._store_executor = None
            self._lookup_executor = None
            self._load_executor = None

        for executor in (store_executor, lookup_executor, load_executor):
            if executor is not None:
                executor.shutdown(wait=True)

        self._client.close()
        for notifier in (self._store_efd, self._lookup_efd, self._load_efd):
            if notifier is not None:
                notifier.close()

        with self._lock:
            self._closed = True

    def report_status(self) -> dict[str, Any]:
        """Return adapter capacity, task, and transport status."""
        with self._lock:
            closing = self._closing
            closed = self._closed
            inflight_store = self._inflight_store_tasks
            inflight_lookup = self._inflight_lookup_tasks
            inflight_load = self._inflight_load_tasks
        with self._state_condition:
            live_slots = len(self._entries)
            inflight_stores = len(self._inflight_stores)
            occupied_slots = self._occupied_slots
            locked_keys = sum(
                entry.external_locks > 0 for entry in self._entries.values()
            )
        return {
            "is_healthy": not closed,
            "type": "cxl_memsim",
            "client_library": self._config.client_library,
            "control_name": self._config.control_name,
            "offset_bytes": self._config.offset_bytes,
            "capacity_bytes": self._arena_bytes,
            "slot_bytes": self._config.slot_bytes,
            "max_slots": self._max_slots,
            "occupied_slot_count": occupied_slots,
            "live_slot_count": live_slots,
            "locked_key_count": locked_keys,
            "inflight_key_store_count": inflight_stores,
            "inflight_store_tasks": inflight_store,
            "inflight_lookup_tasks": inflight_lookup,
            "inflight_load_tasks": inflight_load,
            "closing": closing,
            "supports_restart_recovery": False,
            "transport": asdict(self._client.snapshot_stats()),
        }

    def _ensure_open_locked(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("CxlMemSimL2Adapter is closing or closed")

    def _next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    @staticmethod
    def _signal(notifier: EventNotifier | None) -> None:
        if notifier is None:
            return
        try:
            notifier.notify()
        except OSError:
            logger.debug("Skipping event notification during adapter shutdown")

    def _execute_store_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        successful = True
        bytes_transferred = 0
        stored_keys: list[ObjectKey] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                ok, transferred, inserted = self._store_one(key, obj)
                successful = successful and ok
                bytes_transferred += transferred
                if inserted:
                    stored_keys.append(key)
        except Exception:
            logger.exception("Unexpected CXLMemSim store task failure")
            successful = False
        finally:
            with self._lock:
                self._completed_store_tasks[task_id] = L2StoreResult(
                    successful,
                    bytes_transferred,
                )
                self._inflight_store_tasks -= 1
            if stored_keys:
                self._notify_keys_stored(
                    stored_keys,
                    [self._config.slot_bytes] * len(stored_keys),
                )
            self._signal(self._store_efd)

    def _store_one(
        self,
        key: ObjectKey,
        obj: MemoryObj,
    ) -> tuple[bool, int, bool]:
        with self._state_condition:
            while key in self._inflight_stores:
                self._state_condition.wait()
            if key in self._entries:
                return True, 0, False
            if not self._free_slots:
                return False, 0, False
            slot_id = self._free_slots.pop()
            self._occupied_slots += 1
            self._inflight_stores[key] = slot_id

        buffer = self._host_buffer(obj)
        if buffer is None:
            self._rollback_store(key, slot_id)
            return False, 0, False
        pointer, payload_bytes = buffer
        offset = self._config.offset_bytes + slot_id * self._config.slot_bytes

        try:
            self._client.write_from(offset, pointer, payload_bytes)
        except Exception:
            logger.exception("CXLMemSim write failed for key %s", key)
            self._rollback_store(key, slot_id)
            return False, 0, False

        cached_positions = obj.metadata.cached_positions
        if cached_positions is not None:
            cached_positions = cached_positions.clone()
        with self._state_condition:
            self._inflight_stores.pop(key, None)
            self._entries[key] = _SlotEntry(
                slot_id=slot_id,
                payload_bytes=payload_bytes,
                cached_positions=cached_positions,
            )
            self._state_condition.notify_all()
        return True, payload_bytes, True

    def _rollback_store(self, key: ObjectKey, slot_id: int) -> None:
        with self._state_condition:
            self._inflight_stores.pop(key, None)
            self._free_slots.append(slot_id)
            self._occupied_slots -= 1
            self._state_condition.notify_all()

    def _host_buffer(self, obj: MemoryObj) -> tuple[int, int] | None:
        raw_tensor = obj.raw_tensor
        if (
            raw_tensor is None
            or raw_tensor.device.type != "cpu"
            or not raw_tensor.is_contiguous()
        ):
            return None
        payload_bytes = obj.get_size()
        physical_bytes = obj.get_physical_size()
        tensor_bytes = raw_tensor.numel() * raw_tensor.element_size()
        if (
            payload_bytes <= 0
            or payload_bytes > self._config.slot_bytes
            or payload_bytes > tensor_bytes
            or (physical_bytes > 0 and payload_bytes > physical_bytes)
            or obj.data_ptr <= 0
        ):
            return None
        return obj.data_ptr, payload_bytes

    def _execute_lookup_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
    ) -> None:
        bitmap = Bitmap(len(keys))
        try:
            with self._state_condition:
                for index, key in enumerate(keys):
                    entry = self._entries.get(key)
                    if entry is None:
                        continue
                    entry.external_locks += 1
                    bitmap.set(index)
        except Exception:
            logger.exception("CXLMemSim lookup failed")
        finally:
            with self._lock:
                self._completed_lookup_tasks[task_id] = bitmap
                self._inflight_lookup_tasks -= 1
            self._signal(self._lookup_efd)

    def _finish_empty_load(self, task_id: L2TaskId, size: int) -> None:
        with self._lock:
            self._completed_load_tasks[task_id] = Bitmap(size)
            self._inflight_load_tasks -= 1
        self._signal(self._load_efd)


register_l2_adapter_type("cxl_memsim", CxlMemSimL2AdapterConfig)


def _create_cxl_memsim_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    del l1_memory_desc
    return CxlMemSimL2Adapter(cast(CxlMemSimL2AdapterConfig, config))


register_l2_adapter_factory("cxl_memsim", _create_cxl_memsim_adapter)
