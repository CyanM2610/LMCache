# SPDX-License-Identifier: Apache-2.0
"""Owned lookup references that can be cancelled without touching other readers."""

# Standard
from collections import Counter
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager, L1OperationResult


class HybridReadLease:
    """Move this lookup's logical reservations into independent physical pins."""

    def __init__(self, manager: L1Manager, keys: list[ObjectKey], readers: int) -> None:
        """Adopt exactly the successful, retained reservations of one lookup.

        Args:
            manager: Directory that already owns the read reservations.
            keys: Retained keys from the prefetch handle, including multiplicity.
            readers: Number of read locks reserved for each listed key.
        Raises:
            ValueError: If the reader count is not positive.
        """
        if readers <= 0:
            raise ValueError("lookup reader count must be positive")
        self.manager = manager
        self.remaining = Counter(keys)
        for key in self.remaining:
            self.remaining[key] *= readers
        self.lock = threading.RLock()
        self.closed = False

    def acquire_transfer(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically exchange one owned read per key for physical copy protection.

        Args:
            keys: Unique keys for this worker's submitted LOAD.
        Returns:
            Directory read results. Cancellation/missing ownership returns only
            KEY_NOT_READABLE results and acquires no physical reservations.
            On success the caller owns one transfer pin per key and must release
            it after the actual device/model operation drains, even on failure.
        Raises:
            ValueError: If a physical key is requested more than once.
            RuntimeError: If the directory contradicts this lease's ownership.
        """
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate keys in one physical transfer")
        denied: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_READABLE, None) for key in keys
        }
        with self.lock:
            if self.closed or any(self.remaining[key] < 1 for key in keys):
                return denied
            results = self.manager.unsafe_read(keys, pin_transfer=True)
            good = [
                key
                for key, (error, obj) in results.items()
                if error == L1Error.SUCCESS and obj is not None
            ]
            if len(good) != len(keys):
                self.manager.release_transfer(good)
                return denied
            self.release(keys, 1)
            return results

    def release(self, keys: list[ObjectKey], readers: int) -> None:
        """Release only this lease's unconsumed share; duplicate cleanup is safe.

        Args:
            keys: Owned keys to release, normally an HBM-covered prefix.
            readers: Maximum number of this lookup's readers to release per key.
        Raises:
            ValueError: If the release count is not positive.
            RuntimeError: If an owned directory reservation is unexpectedly absent.
        """
        if readers <= 0:
            raise ValueError("lookup release count must be positive")
        with self.lock:
            groups: dict[int, list[ObjectKey]] = {}
            for key, repeats in Counter(keys).items():
                count = min(self.remaining[key], readers * repeats)
                if count:
                    groups.setdefault(count, []).append(key)
            for count, group in groups.items():
                result = self.manager.finish_read(group, read_locks=count)
                if any(error != L1Error.SUCCESS for error in result.values()):
                    raise RuntimeError("owned lookup reservation is absent")
                for key in group:
                    self.remaining[key] -= count
                    if not self.remaining[key]:
                        del self.remaining[key]

    def close(self) -> None:
        """Release all unused lookup references; leave submitted transfers alone.

        This is idempotent and mutually exclusive with acquire_transfer. It does
        not cancel CUDA or decrement the physical pins already handed to LOAD.
        RuntimeError propagates if the directory has lost an owned reservation.
        """
        with self.lock:
            self.closed = True
            for key, count in list(self.remaining.items()):
                self.release([key], count)

    def reservation_count(self) -> int:
        """Return unconsumed logical reader references, excluding physical pins."""
        with self.lock:
            return sum(self.remaining.values())
