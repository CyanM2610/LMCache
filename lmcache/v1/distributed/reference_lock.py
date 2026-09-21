# SPDX-License-Identifier: Apache-2.0
"""Reference protection that never expires while a copy/reader owns it."""

# Standard
import threading


class ReferenceLock:
    """TTLLock-compatible explicit release semantics for hybrid object lifetimes."""

    def __init__(self) -> None:
        self.count = 0
        self.mutex = threading.Lock()

    def lock(self) -> None:
        """Acquire one reference; passage of time cannot revoke it."""
        with self.mutex:
            self.count += 1

    def unlock(self) -> None:
        """Release one owned reference; raise on unmatched release."""
        with self.mutex:
            if self.count == 0:
                raise RuntimeError("unmatched reference release")
            self.count -= 1

    def is_locked(self) -> bool:
        """Return whether any explicit reference remains."""
        with self.mutex:
            return self.count > 0

    def reset(self) -> None:
        """Release the writer's complete exclusive reservation."""
        with self.mutex:
            self.count = 0
