# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.hotprefix_residency import (
    HotPrefixPhysicalResidencyManager,
)


class _ObjectStore:
    def __init__(self, keys: list[ObjectKey]) -> None:
        self.keys = set(keys)
        self.pin_counts = {key: 0 for key in keys}
        self.unreadable: set[ObjectKey] = set()
        self.deleted: list[ObjectKey] = []
        self.on_pin: Callable[[list[ObjectKey]], None] | None = None

    def pin_retention(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        results = {
            key: (
                L1Error.KEY_NOT_EXIST
                if key not in self.keys
                else (
                    L1Error.KEY_NOT_READABLE
                    if key in self.unreadable
                    else L1Error.SUCCESS
                )
            )
            for key in keys
        }
        if any(error is not L1Error.SUCCESS for error in results.values()):
            return results
        for key in keys:
            self.pin_counts[key] += 1
        if self.on_pin is not None:
            self.on_pin(keys)
        return results

    def unpin_retention(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        results: dict[ObjectKey, L1Error] = {}
        for key in keys:
            if key not in self.keys:
                results[key] = L1Error.KEY_NOT_EXIST
                continue
            self.pin_counts[key] -= 1
            results[key] = L1Error.SUCCESS
        return results

    def request_delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        results: dict[ObjectKey, L1Error] = {}
        for key in keys:
            if key not in self.keys:
                results[key] = L1Error.KEY_NOT_EXIST
            elif self.pin_counts[key] > 0:
                results[key] = L1Error.KEY_IS_LOCKED
            else:
                self.keys.remove(key)
                self.deleted.append(key)
                results[key] = L1Error.SUCCESS
        return results


def _key(value: int, *, group: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=value.to_bytes(4, byteorder="big"),
        model_name="model",
        kv_rank=0,
        object_group_id=group,
    )


def test_shared_objects_remain_pinned_until_last_generation_is_evicted() -> None:
    shared = _key(1)
    first_only = _key(2)
    second_only = _key(3, group=1)
    store = _ObjectStore([shared, first_only, second_only])
    manager = HotPrefixPhysicalResidencyManager(store)

    assert manager.publish_residency(b"first", 1, [shared, first_only])
    assert manager.publish_residency(b"second", 2, [shared, second_only])
    assert store.pin_counts == {shared: 1, first_only: 1, second_only: 1}
    stats = manager.stats_snapshot()
    assert stats.generations == 2
    assert stats.retained_keys == 3

    assert manager.evict_generation(b"first", 1)
    assert store.pin_counts[shared] == 1
    assert first_only in store.deleted
    assert shared not in store.deleted

    assert manager.evict_generation(b"second", 2)
    assert store.pin_counts[shared] == 0
    assert shared in store.deleted
    assert second_only in store.deleted
    stats = manager.stats_snapshot()
    assert stats.generations == 0
    assert stats.retained_keys == 0
    assert stats.discarded_generations == 2


def test_physical_deletion_tombstones_every_affected_generation() -> None:
    shared = _key(1)
    suffix = _key(2)
    store = _ObjectStore([shared, suffix])
    manager = HotPrefixPhysicalResidencyManager(store)
    manager.publish_residency(b"first", 7, [shared])
    manager.publish_residency(b"second", 8, [shared, suffix])

    manager.on_l1_keys_deleted_by_manager([shared])

    assert manager.take_invalidated_generations() == (
        (b"first", 7),
        (b"second", 8),
    )
    assert not manager.wait_for_residency(b"first", 7, 0)
    assert not manager.wait_for_residency(b"second", 8, 0)


def test_incomplete_binding_fails_closed_and_wakes_publication() -> None:
    present = _key(1)
    missing = _key(2)
    store = _ObjectStore([present])
    manager = HotPrefixPhysicalResidencyManager(store)

    assert not manager.publish_residency(b"prefix", 9, [present, missing])
    assert not manager.wait_for_residency(b"prefix", 9, 0)
    assert store.pin_counts[present] == 0
    assert manager.snapshot() == ()


def test_staged_binding_retries_after_overlapping_writer_finishes() -> None:
    ready = _key(1)
    still_writing = _key(2)
    store = _ObjectStore([ready])
    manager = HotPrefixPhysicalResidencyManager(store)

    assert manager.stage_residency_publication(b"prefix", 12, [ready, still_writing])
    assert manager.snapshot() == ()

    store.keys.add(still_writing)
    store.pin_counts[still_writing] = 0
    manager.on_l1_keys_write_finished([still_writing])

    assert manager.wait_for_residency(b"prefix", 12, 0.1)
    assert manager.snapshot()[0].object_keys == (ready, still_writing)


def test_old_generation_eviction_preserves_keys_for_staged_generation() -> None:
    shared = _key(1)
    old_only = _key(2)
    new_only = _key(3)
    store = _ObjectStore([shared, old_only, new_only])
    manager = HotPrefixPhysicalResidencyManager(store)
    assert manager.publish_residency(b"old", 1, [shared, old_only])
    store.unreadable.add(new_only)
    assert manager.stage_residency_publication(b"new", 2, [shared, new_only])

    assert manager.evict_generation(b"old", 1)

    assert shared not in store.deleted
    assert old_only in store.deleted
    store.unreadable.clear()
    assert manager.wait_for_residency(b"new", 2, 0.1)


def test_eviction_rejects_a_late_stream_completion() -> None:
    key = _key(1)
    store = _ObjectStore([key])
    manager = HotPrefixPhysicalResidencyManager(store)

    assert not manager.evict_generation(b"prefix", 10)
    assert not manager.publish_residency(b"prefix", 10, [key])
    assert store.pin_counts[key] == 0


def test_failed_store_cleanup_keeps_objects_shared_by_ready_generation() -> None:
    shared = _key(1)
    unbound = _key(2)
    store = _ObjectStore([shared, unbound])
    manager = HotPrefixPhysicalResidencyManager(store)
    manager.publish_residency(b"ready", 1, [shared])

    manager.delete_unbound_objects([shared, unbound])

    assert shared in store.keys
    assert unbound in store.deleted


def test_eviction_callback_during_publication_prevents_stale_binding() -> None:
    key = _key(1)
    store = _ObjectStore([key])
    manager = HotPrefixPhysicalResidencyManager(store)
    store.on_pin = lambda keys: manager.on_physical_object_evicted(keys[0])

    assert not manager.publish_residency(b"prefix", 11, [key])
    assert manager.take_invalidated_generations() == ((b"prefix", 11),)
    assert manager.snapshot() == ()
    assert store.pin_counts[key] == 0
