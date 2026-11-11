# SPDX-License-Identifier: Apache-2.0
"""Two bounded peer pools using the shared L1 directory and native KV copies.

The CXL pool is a CUDA-registered view of Device-DAX or a CXLMemSim payload.
Allocation does not stage or migrate payload. The caller must hold directory
locks until both CUDA and model completion, including error/cancel paths.
"""

# Standard
import threading

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.cxl_sim_client import (
    ModeledClient,
    PayloadMapping,
    read_payload_descriptor,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.hybrid_config import HybridMemoryConfig
from lmcache.v1.distributed.hybrid_policy import (
    AllPoolsLRUPolicy,
    DramFirstPolicy,
    ForcedPoolPolicy,
    Medium,
    PlacementPolicy,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.physical_dax import PhysicalDaxMapping
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj
from lmcache.v1.platform import current_device_spec


class HybridL1MemoryManager:
    """Own bounded peer allocations, with atomic rollback on batch exhaustion."""

    def __init__(
        self, config: HybridMemoryConfig, policy: PlacementPolicy | None = None
    ) -> None:
        """Map, validate and pin the peer pools, failing closed on pin failure.

        Args:
            config: Pool quotas and either an assigned DAX region or model endpoints.
            policy: Replaceable allocation preference; defaults to DRAM first.
        Raises:
            ValueError: For invalid capacities/alignment or quota beyond backing.
            RuntimeError: If CUDA registration or the server protocol fails.
        """
        if min(config.dram_bytes, config.cxl_bytes) < 0 or (
            config.dram_bytes + config.cxl_bytes == 0
        ):
            raise ValueError("pool quotas must be nonnegative with positive total")
        if config.alignment <= 0 or config.alignment & (config.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if any(
            size % config.alignment for size in (config.dram_bytes, config.cxl_bytes)
        ):
            raise ValueError("pool quotas must be aligned")
        self.config = config
        cxl_medium: Medium = "cxl_dax" if config.cxl_backend == "devdax" else "cxl_sim"
        self.policy = policy or (
            AllPoolsLRUPolicy(cxl_medium)
            if config.policy == "all_lru"
            else DramFirstPolicy(cxl_medium)
            if config.policy == "dram_first"
            else ForcedPoolPolicy(
                "dram" if config.policy == "force_dram" else cxl_medium
            )
        )
        self.lock = threading.RLock()
        self.live: dict[int, Medium] = {}
        self.closed = False
        self.mapping: PayloadMapping | PhysicalDaxMapping | None = None
        self._client: ModeledClient | None = None
        self.registered = False
        self.pools: dict[Medium, TensorMemoryAllocator] = {}
        try:
            if not current_device_spec.is_pin_supported:
                raise RuntimeError("hybrid direct transfers require host pinning")
            if config.cxl_bytes:
                if config.cxl_backend == "devdax":
                    assert config.dax_path is not None
                    self.mapping = PhysicalDaxMapping(
                        config.dax_path,
                        config.dax_offset_bytes,
                        config.dax_region_bytes,
                        config.cxl_bytes,
                    )
                else:
                    descriptor = read_payload_descriptor(config.bulk_control)
                    if config.cxl_bytes > descriptor.capacity:
                        raise ValueError(
                            "CXL quota exceeds the published payload capacity"
                        )
                    self.mapping = PayloadMapping(descriptor)
                    self._client = ModeledClient(
                        config.client_library,
                        config.modeled_control,
                        descriptor.capacity,
                    )
                if not current_device_spec.pin_memory(
                    self.mapping.address, self.mapping.descriptor.capacity
                ):
                    raise RuntimeError(
                        "CUDA registration of CXL payload failed; no staged fallback"
                    )
                self.registered = True
                assert self.mapping.mapping is not None
                cxl_tensor = torch.frombuffer(
                    self.mapping.mapping, dtype=torch.uint8, count=config.cxl_bytes
                )
                self.pools[cxl_medium] = TensorMemoryAllocator(
                    cxl_tensor, config.alignment
                )
            if config.dram_bytes:
                dram_tensor = torch.empty(
                    config.dram_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                )
                self.pools["dram"] = TensorMemoryAllocator(
                    dram_tensor, config.alignment
                )
        except BaseException:
            self.pools.clear()
            # A partially created frombuffer view can otherwise outlive mmap.
            cxl_tensor = None
            if self.registered:
                assert self.mapping is not None
                current_device_spec.unpin_memory(self.mapping.address)
            if self._client is not None:
                self._client.close()
            if self.mapping is not None:
                self.mapping.close()
            raise

    @property
    def client(self) -> ModeledClient:
        """Return the attached model client after successful initialization.

        Raises:
            RuntimeError: If model client initialization did not complete.
        """
        if self._client is None:
            raise RuntimeError("hybrid modeled client is not initialized")
        return self._client

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate one physical slot per object, rolling back an incomplete batch.

        Args:
            layout_desc: Defined shapes/dtypes of each immutable KV version.
            count: Nonnegative number of distinct objects to allocate.
        Returns:
            SUCCESS and all objects, or OUT_OF_MEMORY and an empty list.
        Raises:
            ValueError: For negative count.
            RuntimeError: If this manager is closed.
        """
        if count < 0:
            raise ValueError("count must be nonnegative")
        with self.lock:
            if self.closed:
                raise RuntimeError("hybrid manager is closed")
            objects: list[MemoryObj] = []
            for _ in range(count):
                allocated = None
                for medium in self.policy.pool_order():
                    pool = self.pools.get(medium)
                    if pool is None:
                        continue
                    allocated = pool.allocate(layout_desc.shapes, layout_desc.dtypes)
                    if allocated is not None:
                        self.live[allocated.data_ptr] = medium
                        break
                if allocated is None:
                    self.free(objects)
                    return L1Error.OUT_OF_MEMORY, []
                objects.append(allocated)
            return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Release owned slots after the directory has released all protections.

        Args:
            mem_objs: Live objects allocated by this manager, without duplicates.
        Returns:
            SUCCESS after all objects are invalidated.
        Raises:
            ValueError: If an object is foreign, duplicated, or already freed.
        """
        with self.lock:
            pointers = [obj.data_ptr for obj in mem_objs]
            if len(set(pointers)) != len(pointers) or any(
                p not in self.live for p in pointers
            ):
                raise ValueError(
                    "free requires distinct live objects owned by this pool"
                )
            for pointer, obj in zip(pointers, mem_objs, strict=True):
                medium = self.live.pop(pointer)
                self.pools[medium].free(obj)
                if isinstance(obj, TensorMemoryObj):
                    obj.raw_data = torch.empty(0, dtype=torch.uint8)
            return L1Error.SUCCESS

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the owning medium for a live object, never infer it from a name."""
        with self.lock:
            return L1BackendType(self.live[memory_obj.data_ptr])

    def max_object_count(self, layout: MemoryLayoutDesc) -> int:
        """Return the maximum slot count across pools allowed by the policy."""
        size = sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
        )
        size = (
            (size + self.config.alignment - 1)
            // self.config.alignment
            * self.config.alignment
        )
        if size <= 0:
            return 0
        with self.lock:
            return sum(
                self.pools[medium].address_manager.get_heap_size() // size
                for medium in self.policy.pool_order()
                if medium in self.pools
            )

    def pool_usage(self) -> dict[str, dict[str, int]]:
        """Return actual allocated/backing bytes separately for each medium."""
        with self.lock:
            return {
                medium: {
                    "used_bytes": pool.total_allocated_size,
                    "capacity_bytes": pool.address_manager.get_heap_size(),
                    "backing_bytes": (
                        self.mapping.descriptor.capacity
                        if medium != "dram" and self.mapping is not None
                        else self.config.dram_bytes
                    ),
                    "active_allocations": pool.num_active_allocations,
                }
                for medium, pool in self.pools.items()
            }

    def can_allocate_after_free(
        self, layout: MemoryLayoutDesc, count: int, victims: list[MemoryObj]
    ) -> bool:
        """Check exact aligned-slot feasibility without changing retained payload.

        Args:
            layout: Common layout of the requested objects.
            count: Nonnegative number of requested objects.
            victims: Distinct live objects that the directory may legally evict.
        Returns:
            Whether actual free ranges plus those victims can hold the batch.
            Separate holes are not combined across protected objects or pools.
        Raises:
            ValueError: For invalid counts or foreign/duplicate victim objects.
        Notes:
            The caller must hold the directory mutation lock through subsequent
            eviction/allocation. This preview neither frees nor copies payload.
        """
        if count < 0:
            raise ValueError("count must be nonnegative")
        with self.lock:
            pointers = [obj.data_ptr for obj in victims]
            if len(set(pointers)) != len(pointers) or any(
                pointer not in self.live for pointer in pointers
            ):
                raise ValueError("preview requires distinct owned victim objects")
            ranges = {
                medium: pool.address_manager.get_free_ranges()
                for medium, pool in self.pools.items()
            }
            for obj in victims:
                ranges[self.live[obj.data_ptr]].append(
                    (obj.meta.address, obj.meta.phy_size)
                )
            raw_size = sum(
                shape.numel() * dtype.itemsize
                for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
            )
            slots = 0
            for medium in dict.fromkeys(self.policy.pool_order()):
                if medium not in self.pools:
                    continue
                size = self.pools[medium].address_manager.compute_aligned_size(raw_size)
                if size <= 0:
                    return count == 0
                end, length = 0, 0
                for start, span in sorted(ranges[medium]):
                    if start < end:
                        raise ValueError("overlapping free or victim ranges")
                    if start == end:
                        length += span
                    else:
                        slots += length // size
                        length = span
                    end = start + span
                slots += length // size
            return slots >= count

    def get_memory_usage(self) -> tuple[int, int]:
        """Return summed actual physical allocation and quota bytes."""
        usage = self.pool_usage().values()
        return (
            sum(p["used_bytes"] for p in usage),
            sum(p["capacity_bytes"] for p in usage),
        )

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None: peer pools cannot be advertised as one contiguous region."""
        return None

    def memcheck(self) -> bool:
        """Check allocator accounting and correspondence to live object count."""
        with self.lock:
            return all(pool.memcheck() for pool in self.pools.values()) and sum(
                p.num_active_allocations for p in self.pools.values()
            ) == len(self.live)

    def close(self) -> None:
        """Unregister and unmap only after all objects and model operations drain.

        Raises:
            RuntimeError: If an allocation or model access remains active.
            BufferError: If external Python views still reference the payload.
        """
        with self.lock:
            if self.closed:
                return
            if self.live or (self._client is not None and self._client.active):
                raise RuntimeError(
                    "drain live objects and modeled transfers before close"
                )
            if self._client is not None:
                self._client.close()
            for pool in self.pools.values():
                pool.buffer = torch.empty(0, dtype=torch.uint8)
            self.pools.clear()
            if self.registered:
                assert self.mapping is not None
                if not current_device_spec.unpin_memory(self.mapping.address):
                    raise RuntimeError(
                        "CXL CUDA unregistration failed; mapping retained"
                    )
                self.registered = False
            if self.mapping is not None:
                self.mapping.close()
            self.closed = True
