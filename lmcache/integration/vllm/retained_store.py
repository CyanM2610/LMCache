# SPDX-License-Identifier: Apache-2.0
"""Committed STORE obligations retaining their original producer dependency.

These objects are worker-owned operations, not cache capacity reservations.
Their futures never report completion before the actual transfer is terminal.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import time

# First Party
from lmcache.integration.vllm.kv_page_ownership import TerminalFuture


@dataclass(frozen=True)
class StoreReceipt:
    """One STORE range observation returned through native worker metadata."""

    operation_id: str
    request_id: str
    start: int
    end: int
    phase: str
    succeeded: bool | None = None


@dataclass
class RetainedStore:
    """A committed range and its unchanged producer event.

    ``result`` never initiates a transfer. Callers must explicitly submit before
    waiting; ambiguous submission or completion errors keep the obligation live.
    """

    operation_id: str
    request_id: str
    op: Any
    producer_event: Any
    cache_salt: str = ""
    request_configs: dict[str, Any] | None = None
    future: TerminalFuture | None = None
    submission_error: BaseException | None = None
    terminal_result: bool | None = None
    terminal_observed: bool = False

    def submit(self, callback: Callable[["RetainedStore"], TerminalFuture]) -> bool:
        """Submit at most once, retaining errors with uncertain transfer outcome.

        Args:
            callback: Existing worker transfer submission with the retained event.
        Returns:
            True only when this call creates the transfer future.
        Raises:
            Exception: Original submission error, without releasing protection.
        """
        if self.submission_error is not None:
            raise self.submission_error
        if self.future is not None:
            return False
        try:
            self.future = callback(self)
            if self.future is None:
                raise RuntimeError("committed STORE submission returned no future")
        except BaseException as error:
            self.submission_error = error
            raise
        return True

    def query(self) -> bool:
        """Return actual transfer readiness, never readiness of an unsent range."""
        if self.submission_error is not None:
            raise self.submission_error
        return self.terminal_observed or (
            self.future is not None and self.future.query()
        )

    def result(self, timeout: float | None = None) -> bool:
        """Observe a terminal transfer while retaining unknown/error obligations.

        Args:
            timeout: Forwarded to the real terminal transfer future.
        Returns:
            Terminal transfer success, including False for a proven failed copy.
        Raises:
            TimeoutError: The operation has not been submitted.
            Exception: Original uncertain submission/completion error.
        """
        if self.submission_error is not None:
            raise self.submission_error
        if self.terminal_observed:
            return bool(self.terminal_result)
        if self.future is None:
            raise TimeoutError("committed STORE has not been submitted")
        value = bool(self.future.result(timeout=timeout))
        self.terminal_result = value
        self.terminal_observed = True
        return value


@dataclass
class RequestStoreFence:
    """Native request release waits for every committed range, not the last one."""

    operations: list[RetainedStore] = field(default_factory=list)

    def query(self) -> bool:
        """Return whether all committed ranges have terminal transfer results."""
        return bool(self.operations) and all(op.query() for op in self.operations)

    def result(self, timeout: float | None = None) -> bool:
        """Wait within one timeout budget for all native STORE obligations.

        Args:
            timeout: Total timeout shared across retained operation futures.
        Returns:
            True if all ranges succeeded; False after proven terminal failure.
        Raises:
            Exception: Any uncertain operation error, preserving native release.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        results = [
            op.result(
                timeout=None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            for op in self.operations
        ]
        return all(results)
