# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# First Party
from lmcache.v1.multiprocess.modules.hotprefix import HotPrefixModule


class _PhysicalStorage:
    def __init__(self) -> None:
        self.published: set[tuple[bytes, int]] = set()
        self.invalidated: list[tuple[bytes, int]] = []
        self.evicted: list[tuple[bytes, int]] = []

    def wait_for_residency(
        self, prefix_id: bytes, generation: int, timeout_seconds: float
    ) -> bool:
        del timeout_seconds
        return (prefix_id, generation) in self.published

    def pin_generation(self, prefix_id: bytes, generation: int) -> bool:
        return (prefix_id, generation) in self.published

    def evict_generation(self, prefix_id: bytes, generation: int) -> bool:
        generation_id = (prefix_id, generation)
        self.evicted.append(generation_id)
        self.published.discard(generation_id)
        return True

    def take_invalidated_generations(self) -> tuple[tuple[bytes, int], ...]:
        invalidated = tuple(self.invalidated)
        self.invalidated.clear()
        return invalidated


def test_two_instances_share_residency_and_hold_independent_read_leases() -> None:
    module = HotPrefixModule(
        object(),  # type: ignore[arg-type]
        aging_interval=100,
        host_capacity_bytes=4096,
        frequency_threshold=2,
    )
    namespace = b"model-a\0tenant-a"
    instance_a = module.access(101, 1, namespace, [1, 2, 3, 4], 0)
    instance_b = module.access(202, 1, namespace, [1, 2, 3, 4], 0)
    prefix_id = instance_a.path[-1]

    admission = module.admit(namespace, prefix_id, 4096, 42)
    assert admission.action == "accept"
    assert module.publish(namespace, prefix_id) is True

    candidates = module.candidates(namespace, instance_b.path)
    assert len(candidates) == 1
    assert candidates[0].prefix_id == prefix_id
    assert candidates[0].frequency == 2
    assert candidates[0].generation == 42

    first = module.acquire(namespace, prefix_id, 42, b"instance-a-ticket")
    second = module.acquire(namespace, prefix_id, 42, b"instance-b-ticket")
    assert first.ticket_id != second.ticket_id
    assert module.release(namespace, first.ticket_id) is True
    assert module.release(namespace, second.ticket_id) is True


def test_publish_evicts_victim_and_abort_discards_candidate() -> None:
    storage = _PhysicalStorage()
    module = HotPrefixModule(
        SimpleNamespace(storage_manager=storage),  # type: ignore[arg-type]
        aging_interval=100,
        host_capacity_bytes=100,
        frequency_threshold=1,
    )
    namespace = b"model\0tenant"
    cold = module.access(1, 1, namespace, [1], 0).path[-1]
    module.admit(namespace, cold, 100, 11)
    storage.published.add((cold, 11))
    assert module.publish(namespace, cold)

    hot = module.access(1, 2, namespace, [2], 0).path[-1]
    module.access(1, 3, namespace, [2], 0)
    replacement = module.admit(namespace, hot, 100, 12)
    assert replacement.evict_prefixes == [cold]
    storage.published.add((hot, 12))
    assert module.publish(namespace, hot)
    assert (cold, 11) in storage.evicted

    hotter = module.access(1, 4, namespace, [3], 0).path[-1]
    module.access(1, 5, namespace, [3], 0)
    module.access(1, 6, namespace, [3], 0)
    module.admit(namespace, hotter, 100, 13)
    storage.published.add((hotter, 13))
    assert module.abort(namespace, hotter)
    assert (hotter, 13) in storage.evicted


def test_physical_callback_removes_generation_from_candidate_view() -> None:
    storage = _PhysicalStorage()
    module = HotPrefixModule(
        SimpleNamespace(storage_manager=storage),  # type: ignore[arg-type]
        aging_interval=100,
        host_capacity_bytes=100,
        frequency_threshold=1,
    )
    namespace = b"model\0tenant"
    access = module.access(1, 1, namespace, [1], 0)
    prefix_id = access.path[-1]
    module.admit(namespace, prefix_id, 100, 21)
    storage.published.add((prefix_id, 21))
    assert module.publish(namespace, prefix_id)

    storage.invalidated.append((prefix_id, 21))

    assert module.candidates(namespace, access.path) == []
    assert (prefix_id, 21) in storage.evicted


def test_physical_tombstone_preserves_generation_until_reader_releases() -> None:
    storage = _PhysicalStorage()
    module = HotPrefixModule(
        SimpleNamespace(storage_manager=storage),  # type: ignore[arg-type]
        aging_interval=100,
        host_capacity_bytes=100,
        frequency_threshold=1,
    )
    namespace = b"model\0tenant"
    access = module.access(1, 1, namespace, [1], 0)
    prefix_id = access.path[-1]
    module.admit(namespace, prefix_id, 100, 31)
    storage.published.add((prefix_id, 31))
    assert module.publish(namespace, prefix_id)
    ticket = module.acquire(namespace, prefix_id, 31, b"reader")

    storage.invalidated.append((prefix_id, 31))
    assert module.candidates(namespace, access.path) == []
    assert (prefix_id, 31) not in storage.evicted

    assert module.release(namespace, ticket.ticket_id)
    assert (prefix_id, 31) in storage.evicted
