# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.multiprocess.modules.hotprefix import HotPrefixModule


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
