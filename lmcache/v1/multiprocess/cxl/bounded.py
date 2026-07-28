# SPDX-License-Identifier: Apache-2.0
"""Small bounded collections for terminal-operation bookkeeping."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from collections.abc import Hashable
from typing import Generic, TypeVar


_T = TypeVar("_T", bound=Hashable)


class BoundedSet(Generic[_T]):
    """Retain at most the newest ``capacity`` unique values."""

    def __init__(self, capacity: int = 4096) -> None:
        """Create an insertion-ordered bounded set.

        Args:
            capacity: Maximum number of retained values.

        Raises:
            ValueError: If capacity is not positive.
        """
        if capacity <= 0:
            raise ValueError("bounded set capacity must be positive")
        self._capacity = capacity
        self._values: set[_T] = set()
        self._order: deque[_T] = deque()

    def add(self, value: _T) -> None:
        """Add a value and evict the oldest value at capacity."""
        if value in self._values:
            return
        if len(self._values) == self._capacity:
            self._values.remove(self._order.popleft())
        self._values.add(value)
        self._order.append(value)

    def discard(self, value: _T) -> None:
        """Remove a value when present."""
        if value not in self._values:
            return
        self._values.remove(value)
        self._order.remove(value)

    def __contains__(self, value: object) -> bool:
        """Return whether a value is retained."""
        return value in self._values

    def __len__(self) -> int:
        """Return the number of retained values."""
        return len(self._values)
