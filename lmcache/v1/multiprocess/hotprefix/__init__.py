# SPDX-License-Identifier: Apache-2.0

"""Multi-instance HotPrefix control-plane primitives."""

# Local
from .admission import (
    AdmissionAction,
    HostAdmissionCandidate,
    HostAdmissionDecision,
    HostAdmissionPolicy,
    HostResidencyObservation,
)
from .global_tree import (
    GlobalHostPrefixTree,
    GlobalPrefixNodeSnapshot,
    PrefixAccessObservation,
    PrefixAccessResult,
)
from .residency import (
    HostReadLease,
    HostResidency,
    HostResidencyDirectory,
    HostResidencyState,
)

__all__ = [
    "AdmissionAction",
    "GlobalHostPrefixTree",
    "GlobalPrefixNodeSnapshot",
    "HostAdmissionCandidate",
    "HostAdmissionDecision",
    "HostAdmissionPolicy",
    "HostReadLease",
    "HostResidencyObservation",
    "HostResidency",
    "HostResidencyDirectory",
    "HostResidencyState",
    "PrefixAccessObservation",
    "PrefixAccessResult",
]
