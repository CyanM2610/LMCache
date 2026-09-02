# SPDX-License-Identifier: Apache-2.0

"""Physical object ownership for HotPrefix Host residencies."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from typing import Protocol
import threading
import time

# First Party
from lmcache.utils import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener

logger = init_logger(__name__)

PhysicalGeneration = tuple[bytes, int]


class RetentionObjectStore(Protocol):
    """Object-store operations needed by physical HotPrefix ownership."""

    def pin_retention(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Acquire non-expiring retention pins atomically."""

    def unpin_retention(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Release retention pins."""

    def request_delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete keys now or after their active readers finish."""


@dataclass(frozen=True)
class PhysicalHostResidency:
    """One generation's binding to every local physical object shard."""

    prefix_id: bytes
    generation: int
    object_keys: tuple[ObjectKey, ...]
    pinned: bool = True
    valid: bool = True


@dataclass(frozen=True)
class HotPrefixPhysicalStats:
    """Current physical residency and tombstone counts."""

    generations: int
    retained_keys: int
    discarded_generations: int
    failed_publications: int
    invalidated_generations: int


class HotPrefixPhysicalResidencyManager(L1ManagerListener):
    """Bind HotPrefix generations to L1 objects and own their retention.

    A physical object is pinned once even when several logical prefixes share
    it. The last generation release unpins and requests deletion. L1 deletion
    callbacks invalidate every generation that referenced the removed object.

    Args:
        object_store: L1 object store implementing retention pin operations.
    """

    def __init__(self, object_store: RetentionObjectStore) -> None:
        self._object_store = object_store
        self._residencies: dict[PhysicalGeneration, PhysicalHostResidency] = {}
        self._generations_by_key: dict[ObjectKey, set[PhysicalGeneration]] = {}
        self._publishing_by_key: dict[ObjectKey, set[PhysicalGeneration]] = {}
        self._pending_publications: dict[PhysicalGeneration, tuple[ObjectKey, ...]] = {}
        self._pending_pin_errors: dict[PhysicalGeneration, dict[str, int]] = {}
        self._invalidated: set[PhysicalGeneration] = set()
        self._failed_publications: set[PhysicalGeneration] = set()
        self._discarded: set[PhysicalGeneration] = set()
        self._condition = threading.Condition(threading.Lock())
        self._operation_lock = threading.Lock()

    def publish_residency(
        self,
        prefix_id: bytes,
        generation: int,
        physical_object_keys: list[ObjectKey],
    ) -> bool:
        """Bind and retention-pin a fully written physical generation.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Positive immutable generation identifier.
            physical_object_keys: Every chunk/object-group key on this server.

        Returns:
            ``True`` when the complete generation is bound and pinned. An
            idempotent retry with the same keys also returns ``True``.

        Raises:
            ValueError: If the generation or object-key set is invalid.
            RuntimeError: If an existing generation is rebound to other keys.
        """
        if not prefix_id:
            raise ValueError("prefix_id must not be empty")
        if generation <= 0:
            raise ValueError("generation must be positive")
        object_keys = tuple(dict.fromkeys(physical_object_keys))
        if not object_keys:
            raise ValueError("physical_object_keys must not be empty")
        residency_id = (prefix_id, generation)

        with self._operation_lock:
            with self._condition:
                if residency_id in self._discarded:
                    return False
                existing = self._residencies.get(residency_id)
                if existing is not None:
                    if existing.object_keys != object_keys:
                        raise RuntimeError(
                            "physical generation is already bound to other keys"
                        )
                    if not existing.valid:
                        return False
                    if existing.pinned:
                        return True
                keys_to_pin = self._keys_needing_pin(object_keys, residency_id)
                for key in object_keys:
                    self._publishing_by_key.setdefault(key, set()).add(residency_id)

            if keys_to_pin and not self._pin_keys(keys_to_pin):
                with self._condition:
                    self._remove_publishing(object_keys, residency_id)
                    self._failed_publications.add(residency_id)
                    self._condition.notify_all()
                return False

            rejected = False
            with self._condition:
                self._remove_publishing(object_keys, residency_id)
                if residency_id in self._discarded or residency_id in self._invalidated:
                    rejected = True
                else:
                    existing = self._residencies.get(residency_id)
                    if existing is None:
                        residency = PhysicalHostResidency(
                            prefix_id,
                            generation,
                            object_keys,
                        )
                        self._residencies[residency_id] = residency
                        for key in object_keys:
                            self._generations_by_key.setdefault(key, set()).add(
                                residency_id
                            )
                    else:
                        self._residencies[residency_id] = replace(existing, pinned=True)
                    self._failed_publications.discard(residency_id)
                    self._condition.notify_all()
            if rejected:
                if keys_to_pin:
                    self._object_store.unpin_retention(keys_to_pin)
                return False
            return True

    def stage_residency_publication(
        self,
        prefix_id: bytes,
        generation: int,
        physical_object_keys: list[ObjectKey],
    ) -> bool:
        """Stage a generation that may share keys with an active writer.

        Canonical STOREs for different logical prefixes can overlap on
        content-addressed chunks.  A stream may therefore complete while one
        of its deduplicated keys is still write-locked by another stream.  The
        logical publication waiter retries the atomic retention pin after that
        writer finishes instead of treating this transient state as data loss.

        Returns:
            ``True`` when the publication was accepted for immediate or
            deferred pinning; ``False`` for a tombstoned generation.
        """
        if not prefix_id:
            raise ValueError("prefix_id must not be empty")
        if generation <= 0:
            raise ValueError("generation must be positive")
        object_keys = tuple(dict.fromkeys(physical_object_keys))
        if not object_keys:
            raise ValueError("physical_object_keys must not be empty")
        residency_id = (prefix_id, generation)
        with self._operation_lock:
            with self._condition:
                if residency_id in self._discarded or residency_id in self._invalidated:
                    return False
                existing = self._residencies.get(residency_id)
                if existing is not None:
                    if existing.object_keys != object_keys:
                        raise RuntimeError(
                            "physical generation is already bound to other keys"
                        )
                    return existing.valid
                pending = self._pending_publications.get(residency_id)
                if pending is not None and pending != object_keys:
                    raise RuntimeError(
                        "physical generation publication changed object keys"
                    )
                self._pending_publications[residency_id] = object_keys
                for key in object_keys:
                    self._publishing_by_key.setdefault(key, set()).add(residency_id)
                self._condition.notify_all()
            self._try_publish_pending_locked(residency_id)
            with self._condition:
                return (
                    residency_id not in self._discarded
                    and residency_id not in self._invalidated
                )

    def pin_generation(self, prefix_id: bytes, generation: int) -> bool:
        """Ensure an intact generation owns retention pins.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Exact physical generation.

        Returns:
            ``True`` when the generation exists, is valid, and is pinned.
        """
        residency_id = (prefix_id, generation)
        with self._operation_lock:
            with self._condition:
                residency = self._residencies.get(residency_id)
                if residency is None or not residency.valid:
                    return False
                if residency.pinned:
                    return True
                keys_to_pin = self._keys_needing_pin(
                    residency.object_keys, residency_id
                )
            if keys_to_pin and not self._pin_keys(keys_to_pin):
                return False
            lost = False
            with self._condition:
                current = self._residencies.get(residency_id)
                if current is None or not current.valid:
                    lost = True
                else:
                    self._residencies[residency_id] = replace(current, pinned=True)
                    self._condition.notify_all()
            if lost:
                if keys_to_pin:
                    self._object_store.unpin_retention(keys_to_pin)
                return False
            return True

    def unpin_generation(self, prefix_id: bytes, generation: int) -> bool:
        """Release one generation's retention ownership without deleting it.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Exact physical generation.

        Returns:
            ``True`` when the generation exists, including idempotent retries.
        """
        residency_id = (prefix_id, generation)
        with self._operation_lock:
            with self._condition:
                residency = self._residencies.get(residency_id)
                if residency is None:
                    return False
                if not residency.pinned:
                    return True
                self._residencies[residency_id] = replace(residency, pinned=False)
                keys_to_unpin = self._keys_without_other_pins(
                    residency.object_keys, residency_id
                )
            if keys_to_unpin:
                self._object_store.unpin_retention(keys_to_unpin)
            return True

    def evict_generation(self, prefix_id: bytes, generation: int) -> bool:
        """Retire a generation and delete its unshared objects.

        Active L1 readers are preserved: their objects are marked for deletion
        and reclaimed after the final read finishes.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Exact physical generation.

        Returns:
            ``True`` when a binding was retired, or ``False`` when no binding
            existed. In both cases late publication of the generation is denied.
        """
        residency_id = (prefix_id, generation)
        with self._operation_lock:
            with self._condition:
                self._discarded.add(residency_id)
                self._failed_publications.discard(residency_id)
                self._invalidated.discard(residency_id)
                self._pending_pin_errors.pop(residency_id, None)
                pending = self._pending_publications.pop(residency_id, None)
                if pending is not None:
                    self._remove_publishing(pending, residency_id)
                residency = self._residencies.pop(residency_id, None)
                if residency is None:
                    self._condition.notify_all()
                    return False
                keys_to_unpin = (
                    self._keys_without_other_pins(residency.object_keys, residency_id)
                    if residency.pinned
                    else []
                )
                keys_to_delete: list[ObjectKey] = []
                for key in residency.object_keys:
                    generations = self._generations_by_key.get(key)
                    if generations is None:
                        continue
                    generations.discard(residency_id)
                    if not generations:
                        del self._generations_by_key[key]
                        keys_to_delete.append(key)
                self._condition.notify_all()
            if keys_to_unpin:
                self._object_store.unpin_retention(keys_to_unpin)
            if keys_to_delete:
                self._object_store.request_delete(keys_to_delete)
            return True

    def mark_publication_failed(self, prefix_id: bytes, generation: int) -> None:
        """Wake publication waiters after a physical STORE failure.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Exact failed generation.
        """
        residency_id = (prefix_id, generation)
        with self._condition:
            if residency_id not in self._discarded:
                self._failed_publications.add(residency_id)
            self._condition.notify_all()

    def delete_unbound_objects(self, keys: list[ObjectKey]) -> None:
        """Delete failed STORE objects not referenced by another generation.

        Args:
            keys: Newly allocated keys from the failed physical publication.
        """
        with self._operation_lock:
            with self._condition:
                unbound = [key for key in keys if key not in self._generations_by_key]
            if unbound:
                self._object_store.request_delete(unbound)

    def wait_for_residency(
        self,
        prefix_id: bytes,
        generation: int,
        timeout_seconds: float,
    ) -> bool:
        """Wait for stream-ordered STORE completion to publish physical keys.

        Args:
            prefix_id: Canonical LogicalPrefix identifier.
            generation: Exact expected generation.
            timeout_seconds: Maximum wait duration.

        Returns:
            ``True`` only for a complete, valid, pinned binding.

        Raises:
            ValueError: If ``timeout_seconds`` is negative.
        """
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        residency_id = (prefix_id, generation)
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._condition:
                residency = self._residencies.get(residency_id)
                if residency is not None and residency.valid and residency.pinned:
                    return True
                if (
                    residency_id in self._failed_publications
                    or residency_id in self._discarded
                    or residency_id in self._invalidated
                ):
                    return False
                pending = residency_id in self._pending_publications
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if pending:
                        logger.warning(
                            "Timed out publishing HotPrefix physical generation "
                            "prefix=%s generation=%d pin_errors=%s",
                            prefix_id.hex(),
                            generation,
                            self._pending_pin_errors.get(residency_id, {}),
                        )
                    return False
            if pending:
                with self._operation_lock:
                    self._try_publish_pending_locked(residency_id)
            with self._condition:
                self._condition.wait(min(remaining, 0.01))

    def take_invalidated_generations(self) -> tuple[PhysicalGeneration, ...]:
        """Drain generations tombstoned by physical object deletion callbacks.

        Returns:
            Deterministically ordered ``(prefix_id, generation)`` pairs.
        """
        with self._condition:
            invalidated = tuple(sorted(self._invalidated))
            self._invalidated.clear()
            return invalidated

    def snapshot(self) -> tuple[PhysicalHostResidency, ...]:
        """Return physical bindings in deterministic generation order."""
        with self._condition:
            return tuple(self._residencies[key] for key in sorted(self._residencies))

    def stats_snapshot(self) -> HotPrefixPhysicalStats:
        """Return pull-based physical state without draining tombstones."""
        with self._condition:
            return HotPrefixPhysicalStats(
                generations=len(self._residencies),
                retained_keys=len(self._generations_by_key),
                discarded_generations=len(self._discarded),
                failed_publications=len(self._failed_publications),
                invalidated_generations=len(self._invalidated),
            )

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """Ignore temporary data-plane read locks."""

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Ignore temporary data-plane read-lock release."""

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """Ignore objects until their immutable payload is complete."""

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """Wake staged generations that may depend on completed shared keys."""
        if keys:
            with self._condition:
                self._condition.notify_all()

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        """Ignore temporary prefetch objects."""

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        """Tombstone every generation affected by a physical deletion."""
        for key in keys:
            self.on_physical_object_evicted(key)

    def on_physical_object_evicted(self, key: ObjectKey) -> None:
        """Tombstone generations affected by one backend eviction callback.

        Args:
            key: Exact physical object key removed from the Host tier.
        """
        with self._condition:
            affected = self._generations_by_key.pop(key, set())
            affected.update(self._publishing_by_key.pop(key, set()))
            for residency_id in affected:
                residency = self._residencies.get(residency_id)
                if residency is None:
                    self._invalidated.add(residency_id)
                    continue
                if not residency.valid:
                    continue
                self._residencies[residency_id] = replace(residency, valid=False)
                self._invalidated.add(residency_id)
            self._condition.notify_all()

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Ignore generic access recency; Global Hotness owns retention."""

    def _pin_keys(self, keys: list[ObjectKey]) -> bool:
        results = self._object_store.pin_retention(keys)
        return all(results.get(key) is L1Error.SUCCESS for key in keys)

    def _try_publish_pending_locked(self, residency_id: PhysicalGeneration) -> bool:
        """Try one staged publication while ``_operation_lock`` is held."""
        with self._condition:
            object_keys = self._pending_publications.get(residency_id)
            if object_keys is None:
                residency = self._residencies.get(residency_id)
                return bool(residency is not None and residency.valid)
            if residency_id in self._discarded or residency_id in self._invalidated:
                return False
            keys_to_pin = self._keys_needing_pin(object_keys, residency_id)
        if keys_to_pin:
            pin_results = self._object_store.pin_retention(keys_to_pin)
            pin_errors: dict[str, int] = {}
            for key in keys_to_pin:
                error = pin_results.get(key, L1Error.KEY_NOT_EXIST)
                if error is not L1Error.SUCCESS:
                    pin_errors[error.name] = pin_errors.get(error.name, 0) + 1
            if pin_errors:
                with self._condition:
                    self._pending_pin_errors[residency_id] = pin_errors
                return False

        rejected = False
        with self._condition:
            if residency_id in self._discarded or residency_id in self._invalidated:
                rejected = True
            else:
                prefix_id, generation = residency_id
                self._residencies[residency_id] = PhysicalHostResidency(
                    prefix_id,
                    generation,
                    object_keys,
                )
                for key in object_keys:
                    self._generations_by_key.setdefault(key, set()).add(residency_id)
                self._pending_publications.pop(residency_id, None)
                self._pending_pin_errors.pop(residency_id, None)
                self._remove_publishing(object_keys, residency_id)
                self._failed_publications.discard(residency_id)
                self._condition.notify_all()
        if rejected and keys_to_pin:
            self._object_store.unpin_retention(keys_to_pin)
        return not rejected

    def _remove_publishing(
        self,
        keys: tuple[ObjectKey, ...],
        residency_id: PhysicalGeneration,
    ) -> None:
        for key in keys:
            generations = self._publishing_by_key.get(key)
            if generations is None:
                continue
            generations.discard(residency_id)
            if not generations:
                del self._publishing_by_key[key]

    def _keys_needing_pin(
        self,
        keys: tuple[ObjectKey, ...],
        residency_id: PhysicalGeneration,
    ) -> list[ObjectKey]:
        return [
            key
            for key in keys
            if not any(
                other_id != residency_id
                and (other := self._residencies.get(other_id)) is not None
                and other.pinned
                for other_id in self._generations_by_key.get(key, ())
            )
        ]

    def _keys_without_other_pins(
        self,
        keys: tuple[ObjectKey, ...],
        residency_id: PhysicalGeneration,
    ) -> list[ObjectKey]:
        return [
            key
            for key in keys
            if not any(
                other_id != residency_id
                and (other := self._residencies.get(other_id)) is not None
                and other.pinned
                for other_id in self._generations_by_key.get(key, ())
            )
        ]
