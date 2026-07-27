# SPDX-License-Identifier: Apache-2.0
"""Independent multi-target STORE orchestration for Gate E."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

# Local
from .actions import (
    FetchDecision,
    LookupDecision,
    RecomputeDecision,
    StorePlacementPlan,
    TargetSpec,
    Tier,
    validate_store_plan,
)
from .directory import MultiResidencyDirectory, Residency, ResidencyState
from .observations import (
    PlacementSnapshot,
    RequestObservation,
    ResidencyObservation,
    TierObservation,
)
from .policies import PlacementPolicy
from .tickets import TicketManager, TransferTicket


@dataclass(frozen=True)
class TargetTransferCompletion:
    """Terminal evidence for one independent STORE target."""

    tier: Tier
    success: bool
    error: str | None

    def __post_init__(self) -> None:
        if self.success == (self.error is not None):
            raise ValueError("completion success and error fields disagree")


@dataclass(frozen=True)
class FailedTarget:
    """One target that did not publish."""

    tier: Tier
    required: bool
    error: str


@dataclass(frozen=True)
class StorePlacementResult:
    """Independent per-target STORE outcome."""

    successful_residencies: tuple[Residency, ...]
    failed_targets: tuple[FailedTarget, ...]
    required_satisfied: bool
    partial_success: bool


class StoreSource(Protocol):
    """HBM source lifetime controlled by the STORE coordinator."""

    def release(self) -> None: ...


class StoreTargetExecutor(Protocol):
    """Execute one target-specific transfer without choosing placement."""

    def transfer(
        self, target: TargetSpec, residency: Residency, source: Any
    ) -> TargetTransferCompletion: ...


class StoreCoordinator:
    """Reserve, execute, and publish independent required/optional targets."""

    def __init__(
        self,
        directory: MultiResidencyDirectory,
        executor: StoreTargetExecutor,
    ) -> None:
        self._directory = directory
        self._executor = executor

    def execute(
        self,
        plan: StorePlacementPlan,
        source: StoreSource,
        *,
        length: int,
        alignment: int,
    ) -> StorePlacementResult:
        """Execute every target while retaining source HBM until all finish."""
        validate_store_plan(plan, self._directory.available_tiers)
        reserved: list[tuple[TargetSpec, Residency]] = []
        failures: list[FailedTarget] = []
        successful: list[Residency] = []
        successful_tiers: set[Tier] = set()
        try:
            for target in plan.targets:
                try:
                    residency = self._directory.reserve_residency(
                        plan.object_key,
                        target,
                        length=length,
                        alignment=alignment,
                    )
                    reserved.append((target, residency))
                except Exception as error:
                    failures.append(
                        FailedTarget(
                            target.tier,
                            target.required,
                            str(error) or type(error).__name__,
                        )
                    )

            for target, residency in reserved:
                current = residency
                try:
                    current = self._directory.mark_writing(residency.residency_id)
                    completion = self._executor.transfer(target, current, source)
                    if not completion.success:
                        raise RuntimeError(completion.error or "target transfer failed")
                    ready = self._directory.publish(residency.residency_id, completion)
                    successful.append(ready)
                    successful_tiers.add(target.tier)
                except Exception as error:
                    failures.append(
                        FailedTarget(
                            target.tier,
                            target.required,
                            str(error) or type(error).__name__,
                        )
                    )
                    try:
                        self._directory.abort(
                            residency.residency_id,
                            str(error) or "target transfer failed",
                        )
                    except (KeyError, RuntimeError):
                        pass
        finally:
            source.release()

        required_satisfied = all(
            not target.required or target.tier in successful_tiers
            for target in plan.targets
        )
        return StorePlacementResult(
            successful_residencies=tuple(successful),
            failed_targets=tuple(failures),
            required_satisfied=required_satisfied,
            partial_success=bool(successful) and bool(failures),
        )


@dataclass(frozen=True)
class BoundLookup:
    """Policy decision paired with the ticket required to execute it."""

    decision: FetchDecision
    ticket: TransferTicket


def snapshot_from_directory(
    directory: MultiResidencyDirectory,
    request: RequestObservation,
    tiers: tuple[TierObservation, ...],
    timestamp_ns: int,
    *,
    excluded_residencies: frozenset[str] = frozenset(),
) -> PlacementSnapshot:
    """Build a domain-only lookup snapshot from immutable directory state."""
    observations = []
    for residency in directory.list_residencies(request.object_key):
        if (
            residency.residency_id in excluded_residencies
            or residency.state is ResidencyState.RESERVED
            or residency.descriptor is None
        ):
            continue
        observations.append(
            ResidencyObservation(
                residency.residency_id,
                residency.object_key,
                residency.tier,
                residency.state.value,
                residency.descriptor.length,
                residency.generation,
                residency.descriptor.layout_fingerprint,
                residency.last_access_ns,
                residency.access_count,
                residency.active_readers,
                residency.pinned,
            )
        )
    return PlacementSnapshot(
        timestamp_ns,
        request,
        tuple(observations),
        tiers,
    )


class LookupCoordinator:
    """Plan and atomically bind a fetch, with one fresh-snapshot replan."""

    def __init__(self, policy: PlacementPolicy, tickets: TicketManager) -> None:
        self._policy = policy
        self._tickets = tickets

    def decide_and_bind(
        self,
        request_id: str,
        snapshot_factory: Callable[[], PlacementSnapshot],
        *,
        now_ns: int,
    ) -> BoundLookup | RecomputeDecision:
        """Return only a bound FETCH or an explicit RECOMPUTE decision."""
        last_error = ""
        for _ in range(2):
            decision: LookupDecision = self._policy.decide_lookup(snapshot_factory())
            if isinstance(decision, RecomputeDecision):
                return decision
            try:
                ticket = self._tickets.bind_fetch(decision, request_id, now_ns)
            except RuntimeError as error:
                last_error = str(error)
                continue
            return BoundLookup(decision, ticket)
        return RecomputeDecision(
            f"FETCH could not bind after one fresh replan: {last_error}"
        )
