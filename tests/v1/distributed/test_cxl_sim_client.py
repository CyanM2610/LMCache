# SPDX-License-Identifier: Apache-2.0
"""Contract tests for metadata-only CXLMemSim access and mapping validation.

No GPU is required. Live-server tests use CXLSIM_TEST_BUILD and launch a private
process; these are protocol/state tests, not physical CXL performance evidence.
"""

# Standard
from pathlib import Path
from unittest.mock import patch
import os
import struct
import subprocess
import tempfile
import time
import unittest
import uuid

# First Party
from lmcache.v1.distributed.cxl_sim_client import (
    ModeledClient,
    PayloadMapping,
    read_payload_descriptor,
    shm_path,
)


class DescriptorTests(unittest.TestCase):
    """Reject invalid descriptors before mapping any payload."""

    def test_rejects_names_which_escape_shm(self) -> None:
        """A control or data name cannot select arbitrary filesystem paths."""
        for name in ("", "/", "relative", "/../x", "/..", "/.", "/x/y", "/x\0"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                shm_path(name)

    def test_accepts_only_ready_v3_page_aligned_mapping(self) -> None:
        """Bounds, actual file size and readiness guard against unsafe mmap."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_bytes(bytes(8192))
            raw = bytearray(256)
            struct.pack_into(
                "<Q6I4Q",
                raw,
                0,
                0x43584C42554C4B31,
                3,
                8448,
                64,
                64,
                1,
                0,
                1,
                8192,
                4096,
                4096,
            )
            raw[96:102] = b"/data\0"
            (root / "control").write_bytes(raw)
            with patch(
                "lmcache.v1.distributed.cxl_sim_client.shm_path",
                side_effect=lambda name: root / name.lstrip("/"),
            ):
                descriptor = read_payload_descriptor("/control")
                self.assertEqual(descriptor.capacity, 4096)
                mapping = PayloadMapping(descriptor)
                mapping.mapping[:4] = b"KV01"
                mapping.close()
                self.assertEqual((root / "data").read_bytes()[4096:4100], b"KV01")
                for offset, fmt, value in (
                    (8, "<I", 2),
                    (24, "<I", 0),
                    (48, "<Q", 4097),
                    (56, "<Q", 8192),
                    (40, "<Q", 16384),
                ):
                    changed = bytearray(raw)
                    struct.pack_into(fmt, changed, offset, value)
                    (root / "control").write_bytes(changed)
                    with self.subTest(offset=offset), self.assertRaises(ValueError):
                        read_payload_descriptor("/control")


@unittest.skipUnless(os.environ.get("CXLSIM_TEST_BUILD"), "requires rebuilt CXLMemSim")
class LiveProtocolTests(unittest.TestCase):
    """Exercise native server semantics without pretending to execute CUDA."""

    def setUp(self) -> None:
        """Start a uniquely named server with a two-second startup bound."""
        self.build = Path(os.environ["CXLSIM_TEST_BUILD"]).resolve()
        self.prefix = "/l2hybrid_test_" + uuid.uuid4().hex
        self.bulk = self.prefix + "_bulk"
        self.modeled = self.prefix + "_model"
        self.data = self.prefix + "_data"
        self.server = subprocess.Popen(
            [
                str(self.build / "cxlmemsim_server"),
                "--comm-mode=bulk-shm",
                "--capacity=1",
                "--bulk-shm-name=" + self.bulk,
                "--data-shm-name=" + self.data,
                "--modeled-access-shm-name=" + self.modeled,
                "--enable-gpu-direct-modeled-access=true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop_server)
        deadline = time.monotonic() + 10
        while not shm_path(self.modeled).exists():
            if self.server.poll() is not None or time.monotonic() >= deadline:
                self.fail("private CXLMemSim did not start")
            time.sleep(0.01)
        self.descriptor = read_payload_descriptor(self.bulk)
        self.client = ModeledClient(
            str(self.build / "libcxlmemsim_modeled_client.so"),
            self.modeled,
            self.descriptor.capacity,
        )

    def stop_server(self) -> None:
        """Stop only this test's process and unlink its unique objects."""
        self.server.terminate()
        try:
            self.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=5)
        for name in (self.bulk, self.bulk + ".lock", self.modeled, self.data):
            shm_path(name).unlink(missing_ok=True)

    def test_cancelled_lookup_releases_unconsumed_read_reservations(self) -> None:
        """Cancellation before LOAD must retire polled and unpolled hit lookups."""
        # Standard
        from dataclasses import replace

        # Third Party
        import torch

        # First Party
        from lmcache.v1.distributed.api import MemoryLayoutDesc
        from lmcache.v1.distributed.config import (
            EvictionConfig,
            L1ManagerConfig,
            L1MemoryManagerConfig,
            StorageManagerConfig,
        )
        from lmcache.v1.distributed.hybrid_config import HybridMemoryConfig
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
        from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
        from lmcache.v1.multiprocess.modules.lookup import LookupModule

        for polled in (False, True):
            with self.subTest(polled=polled):
                config = StorageManagerConfig(
                    L1ManagerConfig(
                        L1MemoryManagerConfig(4096, False, shm_name=""),
                        hybrid_config=HybridMemoryConfig(
                            4096,
                            8192,
                            self.bulk,
                            self.modeled,
                            str(self.build / "libcxlmemsim_modeled_client.so"),
                            policy="force_cxl",
                        ),
                    ),
                    EvictionConfig(eviction_policy="noop"),
                )
                ctx = MPCacheServerContext(config, chunk_size=4)
                lookup = LookupModule(ctx)
                layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
                ctx.layout_desc_registry.register("model", 1, layout)
                key = IPCCacheServerKey(
                    model_name="model",
                    world_size=1,
                    num_kv_readers=1,
                    worker_id=0,
                    token_ids=(1, 2, 3, 4),
                    start=0,
                    end=4,
                    request_id="seed",
                )
                keys = ctx.resolve_obj_keys(key, [0])[0]
                ctx.session_manager.remove("seed")
                ctx.storage_manager.reserve_write(keys, layout, "new")
                ctx.storage_manager.finish_write(keys)
                try:
                    request = replace(key, worker_id=None, request_id="cancelled")
                    lookup.lookup(request, 1)
                    if polled:
                        self.assertEqual(lookup.query_prefetch_status("cancelled"), 1)
                    lookup.end_session("cancelled")
                    # END_SESSION may also overtake a queued LOOKUP. Replayed
                    # work must not recreate the session or reserve again.
                    lookup.lookup(request, 1)
                    self.assertEqual(lookup.query_prefetch_status("cancelled"), 0)
                    status = ctx.storage_manager.report_status()["l1_manager"]
                    self.assertEqual(status["read_locked_count"], 0)
                    self.assertEqual(lookup.report_status()["active_prefetch_jobs"], 0)
                    self.assertEqual(ctx.session_manager.active_count(), 0)
                finally:
                    # Keep the deliberately failing regression safe to tear down.
                    if ctx.storage_manager.report_status()["l1_manager"][
                        "read_locked_count"
                    ]:
                        ctx.storage_manager.finish_read_prefetched(keys)
                    lookup.close()
                    ctx.close()
        self.client.close()

    def test_lookup_end_preserves_other_reader_and_transferred_pins(self) -> None:
        """Unused lookup shares and already transferred physical pins are distinct."""
        # Standard
        from dataclasses import replace

        # Third Party
        import torch

        # First Party
        from lmcache.v1.distributed.api import MemoryLayoutDesc
        from lmcache.v1.distributed.config import (
            EvictionConfig,
            L1ManagerConfig,
            L1MemoryManagerConfig,
            StorageManagerConfig,
        )
        from lmcache.v1.distributed.hybrid_config import HybridMemoryConfig
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
        from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
        from lmcache.v1.multiprocess.modules.lookup import LookupModule

        config = StorageManagerConfig(
            L1ManagerConfig(
                L1MemoryManagerConfig(4096, False, shm_name=""),
                hybrid_config=HybridMemoryConfig(
                    4096,
                    8192,
                    self.bulk,
                    self.modeled,
                    str(self.build / "libcxlmemsim_modeled_client.so"),
                    policy="force_cxl",
                ),
            ),
            EvictionConfig(eviction_policy="noop"),
        )
        ctx = MPCacheServerContext(config, chunk_size=4)
        lookup = LookupModule(ctx)
        layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
        ctx.layout_desc_registry.register("model", 1, layout)
        key = IPCCacheServerKey(
            model_name="model",
            world_size=1,
            num_kv_readers=1,
            worker_id=0,
            token_ids=tuple(range(8)),
            start=0,
            end=8,
            request_id="seed",
        )
        keys = ctx.resolve_obj_keys(key, [0])[0]
        ctx.session_manager.remove("seed")
        ctx.storage_manager.reserve_write(keys, layout, "new")
        ctx.storage_manager.finish_write(keys)
        pins = False
        try:
            for request_id in ("A", "B"):
                lookup.lookup(replace(key, worker_id=None, request_id=request_id), 1)
                self.assertEqual(lookup.query_prefetch_status(request_id), 2)
            session = ctx.session_manager.get("A")
            assert session is not None
            with ctx.storage_manager.read_prefetched_results(
                keys,
                read_lease=session.hybrid_read_lease,
                release_on_error=False,
            ) as objects:
                self.assertIsNotNone(objects)
                pins = True
            lookup.end_session("A")
            lookup.end_session("A")
            self.assertEqual(
                ctx.storage_manager.report_status()["l1_manager"]["read_locked_count"],
                2,
            )
            prefix = replace(key, worker_id=None, request_id="B", end=4)
            lookup.free_lookup_locks(prefix, 1)
            lookup.free_lookup_locks(prefix, 1)
            self.assertEqual(
                ctx.storage_manager.report_status()["l1_manager"]["read_locked_count"],
                1,
            )
            lookup.end_session("B")
            status = ctx.storage_manager.report_status()["l1_manager"]
            self.assertEqual(status["read_locked_count"], 0)
            self.assertEqual(status["transfer_reservation_count"], 2)
            next_key = replace(
                key, token_ids=(20, 21, 22, 23), end=4, request_id="next"
            )
            new_keys = ctx.resolve_obj_keys(next_key, [0])[0]
            ctx.session_manager.remove("next")
            self.assertEqual(
                ctx.storage_manager.reserve_write(new_keys, layout, "new"), {}
            )
            ctx.storage_manager.release_transfer(keys)
            pins = False
            self.assertEqual(
                len(ctx.storage_manager.reserve_write(new_keys, layout, "new")), 1
            )
            ctx.storage_manager.finish_write(new_keys)
        finally:
            if pins:
                ctx.storage_manager.release_transfer(keys)
            lookup.close()
            ctx.close()
            self.client.close()

    def test_model_completion_alone_does_not_publish_payload(self) -> None:
        """A report of payload success is required even after model time elapses."""
        token = self.client.begin("load", 0, 64).access_token
        self.assertIn(self.client.poll(token).state, (0, 1))
        with self.assertRaises(RuntimeError):
            self.client.close()
        self.client.data_complete(token, True)
        self.assertEqual(self.client.poll(token).state, 3)
        self.assertEqual(len(self.client.active), 0)
        self.client.close()

    def test_cancel_requires_local_transfer_drain(self) -> None:
        """A terminal model cancellation does not release a DMA reservation."""
        token = self.client.begin("store", 0, 64).access_token
        self.client.cancel(token, "request cancelled")
        self.assertEqual(self.client.poll(token).state, 4)
        with self.assertRaises(RuntimeError):
            self.client.close()
        self.client.data_complete(token, False)
        self.assertEqual(self.client.poll(token).state, 4)
        self.assertEqual(len(self.client.active), 0)
        self.client.close()

    def test_failure_never_reports_success(self) -> None:
        """Failed payload transfer reaches ERROR, then releases its reservation."""
        token = self.client.begin("store", 0, 64).access_token
        self.client.data_complete(token, False)
        self.assertEqual(self.client.poll(token).state, 5)
        self.assertEqual(len(self.client.active), 0)
        with self.assertRaises(ValueError):
            self.client.poll(token)
        self.client.close()

    def test_out_of_bounds_is_rejected_without_reservation(self) -> None:
        """Capacity is physical mapped data bytes, excluding the header page."""
        with self.assertRaises(ValueError):
            self.client.begin("load", self.descriptor.capacity, 1)
        with self.assertRaises(ValueError):
            self.client.begin("load", -1, 1)
        self.assertEqual(len(self.client.active), 0)
        self.client.close()

    def test_capacity_mismatch_fails_closed(self) -> None:
        """Do not attach a mapped pool with different modeled capacity."""
        with self.assertRaises(ValueError):
            ModeledClient(
                str(self.build / "libcxlmemsim_modeled_client.so"),
                self.modeled,
                self.descriptor.capacity + 4096,
            )
        self.client.close()

    def test_peer_pools_roll_back_partial_batch_without_touching_residents(
        self,
    ) -> None:
        """Capacity failure must release only the new batch's reservations."""
        # Third Party
        import torch

        # First Party
        from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.memory_manager.hybrid_l1_memory_manager import (
            HybridL1MemoryManager,
            HybridMemoryConfig,
        )

        if not torch.cuda.is_available():
            self.client.close()
            self.skipTest("peer-pool registration requires actual CUDA")
        manager = HybridL1MemoryManager(
            HybridMemoryConfig(
                dram_bytes=4096,
                cxl_bytes=8192,
                bulk_control=self.bulk,
                modeled_control=self.modeled,
                client_library=str(self.build / "libcxlmemsim_modeled_client.so"),
            )
        )
        layout = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
        objects = []
        try:
            error, first = manager.allocate(layout, 1)
            objects.extend(first)
            self.assertEqual(error, L1Error.SUCCESS)
            self.assertEqual(manager.get_backend_type(first[0]), L1BackendType.DRAM)
            tensor = first[0].raw_tensor
            assert tensor is not None
            tensor.fill_(13)
            error, rejected = manager.allocate(layout, 3)
            self.assertEqual(error, L1Error.OUT_OF_MEMORY)
            self.assertEqual(rejected, [])
            self.assertEqual(manager.get_memory_usage(), (4096, 12288))
            self.assertTrue(torch.all(first[0].raw_tensor == 13))
            error, rest = manager.allocate(layout, 2)
            objects.extend(rest)
            self.assertEqual(error, L1Error.SUCCESS)
            for obj in rest:
                self.assertEqual(manager.get_backend_type(obj), L1BackendType.CXL_SIM)
                assert obj.raw_tensor is not None
                self.assertTrue(obj.raw_tensor.is_pinned())
                self.assertGreaterEqual(obj.data_ptr, manager.mapping.address)
                self.assertLess(obj.data_ptr, manager.mapping.address + 8192)
            with self.assertRaises(RuntimeError):
                manager.close()
            with self.assertRaises(ValueError):
                manager.free([first[0], first[0]])
            self.assertEqual(manager.get_memory_usage(), (12288, 12288))
            self.assertTrue(manager.memcheck())
        finally:
            manager.free(objects)
            self.assertEqual(manager.get_memory_usage(), (0, 12288))
            manager.close()
            self.client.close()

    def test_directory_protects_readers_and_dma_under_cxl_eviction(self) -> None:
        """Only released CXL versions can be evicted; zero TTL never revokes pins."""
        # Third Party
        import torch

        # First Party
        from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
        from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.hybrid_config import HybridMemoryConfig
        from lmcache.v1.distributed.l1_manager import L1Manager

        if not torch.cuda.is_available():
            self.client.close()
            self.skipTest("requires actual CUDA registration")
        cfg = HybridMemoryConfig(
            4096,
            8192,
            self.bulk,
            self.modeled,
            str(self.build / "libcxlmemsim_modeled_client.so"),
        )
        manager = L1Manager(
            L1ManagerConfig(
                L1MemoryManagerConfig(4096, False, shm_name=""),
                write_ttl_seconds=0,
                read_ttl_seconds=0,
                hybrid_config=cfg,
            )
        )
        layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
        dram, cold, other, new, blocked = [
            ObjectKey(bytes([i]), "m", 0) for i in range(5)
        ]
        keys = [dram, cold, other, new, blocked]
        try:
            reserved = manager.reserve_write(
                [dram, cold, other, other], [False] * 4, layout, "new"
            )
            self.assertEqual(len(reserved), 3)
            self.assertEqual(manager.get_memory_usage()[0], 12288)
            self.assertEqual(
                manager.reserve_read([cold])[cold][0], L1Error.KEY_NOT_READABLE
            )
            self.assertEqual(
                manager.delete([cold], force=True)[cold], L1Error.KEY_IS_LOCKED
            )
            manager.finish_write([dram, cold, other])
            manager.reserve_read([cold], read_locks=2)
            result = manager.reserve_write([new], [False], layout, "new")
            self.assertEqual(result[new][0], L1Error.SUCCESS)
            self.assertIsNone(manager.get_object_state(other))
            self.assertIsNotNone(manager.get_object_state(dram))
            manager.finish_write([new])
            manager.unsafe_read([cold], pin_transfer=True)
            manager.finish_read([cold], read_locks=2)
            manager.reserve_read([new])
            self.assertEqual(
                manager.reserve_write([blocked], [False], layout, "new")[blocked][0],
                L1Error.OUT_OF_MEMORY,
            )
            self.assertEqual(
                manager.delete([cold], force=True)[cold], L1Error.KEY_IS_LOCKED
            )
            manager.release_transfer([cold])
            self.assertEqual(
                manager.reserve_write([blocked], [False], layout, "new")[blocked][0],
                L1Error.SUCCESS,
            )
            self.assertIsNone(manager.get_object_state(cold))
            self.assertIsNotNone(manager.get_object_state(dram))
            manager.finish_write([blocked])
            manager.finish_read([new])
            self.assertEqual(manager.report_status()["transfer_reservation_count"], 0)
            self.assertTrue(manager.memcheck())
        finally:
            for key in keys:
                state = manager.get_object_state(key)
                if state is None:
                    continue
                while state.transfer_refs:
                    manager.release_transfer([key])
                while state.read_lock.is_locked():
                    manager.finish_read([key])
                if state.write_lock.is_locked():
                    manager.abort_write([key])
            manager.close()
            self.client.close()


if __name__ == "__main__":
    unittest.main()
