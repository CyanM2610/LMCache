# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.multiprocess.hotprefix.admission import (
    AdmissionAction,
    HostAdmissionCandidate,
    HostAdmissionPolicy,
    HostResidencyObservation,
)


def test_admission_deduplicates_ready_shared_residency() -> None:
    policy = HostAdmissionPolicy(frequency_threshold=2)
    candidate = HostAdmissionCandidate(b"prefix-a", 100, 3, 9)
    existing = (HostResidencyObservation(b"prefix-a", 100, 1, 4, True),)

    decision = policy.decide(candidate, existing, capacity_bytes=100, used_bytes=100)

    assert decision.action is AdmissionAction.DEDUP
    assert decision.evict_prefixes == ()


def test_admission_replaces_colder_residency_when_capacity_is_full() -> None:
    policy = HostAdmissionPolicy(frequency_threshold=2)
    candidate = HostAdmissionCandidate(b"prefix-hot", 100, 4, 10)
    existing = (
        HostResidencyObservation(b"prefix-cold", 100, 1, 3, True),
        HostResidencyObservation(b"prefix-warmer", 100, 2, 10, True),
    )

    decision = policy.decide(candidate, existing, capacity_bytes=200, used_bytes=200)

    assert decision.action is AdmissionAction.ACCEPT
    assert decision.evict_prefixes == (b"prefix-cold",)


def test_admission_rejects_frequency_and_hotness_below_shared_thresholds() -> None:
    policy = HostAdmissionPolicy(frequency_threshold=2)
    existing = (HostResidencyObservation(b"prefix-resident", 100, 2, 10, True),)

    low_frequency = policy.decide(
        HostAdmissionCandidate(b"prefix-new", 100, 1, 255),
        existing,
        capacity_bytes=100,
        used_bytes=100,
    )
    low_hotness = policy.decide(
        HostAdmissionCandidate(b"prefix-new", 100, 2, 5),
        existing,
        capacity_bytes=100,
        used_bytes=100,
    )

    assert low_frequency.action is AdmissionAction.REJECT
    assert low_frequency.reason == "frequency_below_threshold"
    assert low_hotness.action is AdmissionAction.REJECT
    assert low_hotness.reason == "not_hotter_than_replacement"
