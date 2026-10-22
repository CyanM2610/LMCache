# SPDX-License-Identifier: Apache-2.0
"""Confirmed logical-session cleanup independent of scheduler lookup results."""

# Standard
from dataclasses import dataclass
from typing import Any
import threading
import time

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient


@dataclass
class _PendingEnd:
    future: MessagingFuture[Any] | None = None
    submitted: float = 0.0
    attempts: int = 0
    generation_mismatch: bool = False


class ConfirmedSessionCleanup:
    """Keep logical cleanup owned until its original server acknowledges it."""

    def __init__(
        self,
        clients: dict[str, RequestClient],
        health: dict[str, threading.Event],
        timeout: float,
    ) -> None:
        """Handshake with each hybrid backend and start an idle cleanup owner.

        Args:
            clients: Existing clients, kept alive until cleanup drains.
            health: Adapter health gates, cleared on uncertain cleanup.
            timeout: RPC timeout and minimum retry spacing in seconds.
        Raises:
            TimeoutError: If a generation handshake cannot complete.
            RuntimeError: If a peer does not advertise a hybrid generation.
        """
        self.clients = clients
        self.health = health
        self.timeout = max(timeout, 0.01)
        self.generations: dict[str, str] = {}
        for url, client in clients.items():
            generation = client.get_server_generation().result(timeout=timeout)
            if not isinstance(generation, str) or not generation:
                raise RuntimeError("hybrid backend generation is unavailable")
            self.generations[url] = generation
        self.condition = threading.Condition()
        self.pending: dict[tuple[str, str], _PendingEnd] = {}
        self.confirmed = 0
        self.retries = 0
        self.last_error: str | None = None
        self.stopped = False
        self.thread = threading.Thread(
            target=self._run, name="lmcache-hybrid-session-cleanup", daemon=True
        )
        self.thread.start()

    def retire(self, request_id: str) -> None:
        """Retain one END obligation per server without blocking native scheduling.

        Args:
            request_id: Unique engine request ID whose demand has ended.
        Raises:
            RuntimeError: If this owner has already stopped.
        """
        with self.condition:
            if self.stopped:
                raise RuntimeError("session cleanup owner is closed")
            for url in self.clients:
                self.pending.setdefault((url, request_id), _PendingEnd())
            self.condition.notify_all()

    def recover(self, url: str) -> bool:
        """Permit recovery only in the same generation with no unconfirmed END.

        Args:
            url: Backend endpoint whose heartbeat has recovered.
        Returns:
            False for outstanding cleanup, changed generation or RPC failure.
            A changed incarnation requires explicit backend/engine restart.
        """
        try:
            generation = (
                self.clients[url]
                .get_server_generation()
                .result(timeout=min(self.timeout, 1.0))
            )
        except Exception:
            return False
        with self.condition:
            if generation != self.generations[url]:
                self.last_error = f"backend generation changed: {url}"
                return False
            for (server, _), job in self.pending.items():
                if server == url and job.generation_mismatch:
                    job.generation_mismatch = False
                    job.future = None
                    job.submitted = 0.0
            self.condition.notify_all()
            return not any(server == url for server, _ in self.pending)

    def report_status(self) -> dict[str, Any]:
        """Return pending ownership, original generations and confirmation counts."""
        with self.condition:
            return {
                "pending": len(self.pending),
                "confirmed": self.confirmed,
                "retries": self.retries,
                "generations": dict(self.generations),
                "requests": [
                    {
                        "server": url,
                        "request_id": request_id,
                        "generation": self.generations[url],
                        "attempts": job.attempts,
                        "generation_mismatch": job.generation_mismatch,
                    }
                    for (url, request_id), job in self.pending.items()
                ],
                "last_error": self.last_error,
            }

    def close(self, timeout: float) -> None:
        """Drain confirmed ENDs before the adapter closes its transport clients.

        Args:
            timeout: Maximum shutdown wait in seconds.
        Raises:
            RuntimeError: If ownership is still unconfirmed. The caller must
                retain clients and this owner; no DMA pin or TTL is released.
        """
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("hybrid session cleanup is unconfirmed")
                self.condition.wait(timeout=remaining)
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1)

    def _run(self) -> None:
        with self.condition:
            while not self.stopped:
                if not self.pending:
                    self.condition.wait()
                    continue
                now = time.monotonic()
                for (url, request_id), job in list(self.pending.items()):
                    if job.generation_mismatch:
                        continue
                    if job.future is not None and job.future.query():
                        try:
                            accepted = job.future.result(timeout=0)
                        except Exception as error:
                            self.last_error = str(error)
                            self.health[url].clear()
                            job.future = None
                        else:
                            if accepted is True:
                                del self.pending[url, request_id]
                                self.confirmed += 1
                                continue
                            job.generation_mismatch = True
                            self.last_error = (
                                f"END rejected for original generation: {url}"
                            )
                            self.health[url].clear()
                            continue
                    if job.attempts and now - job.submitted < self.timeout:
                        continue
                    if job.attempts:
                        self.health[url].clear()
                        self.retries += 1
                    job.attempts += 1
                    job.submitted = now
                    try:
                        job.future = self.clients[url].end_session_guarded(
                            request_id, self.generations[url]
                        )
                    except Exception as error:
                        job.future = None
                        self.last_error = str(error)
                        self.health[url].clear()
                self.condition.notify_all()
                self.condition.wait(timeout=0.005)
