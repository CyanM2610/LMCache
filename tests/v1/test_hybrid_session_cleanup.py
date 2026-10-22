# SPDX-License-Identifier: Apache-2.0
"""Confirmed cleanup contracts at the asynchronous transport boundary."""

# Standard
from collections import deque
from collections.abc import Callable
from typing import Any, cast
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.hybrid_session_cleanup import ConfirmedSessionCleanup
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient


def ready(value: Any) -> MessagingFuture[Any]:
    """Return an actual completed transport future."""
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


def wait_until(predicate: Callable[[], bool]) -> None:
    """Require an asynchronous public condition within one second."""
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.002)


class Peer:
    """Controllable transport peer with explicit incarnation and END replies."""

    def __init__(self) -> None:
        self.generation = "original"
        self.calls: list[tuple[str, str]] = []
        self.replies: deque[MessagingFuture[Any]] = deque()

    def get_server_generation(self) -> MessagingFuture[Any]:
        """Return the incarnation actually reached by this endpoint."""
        return ready(self.generation)

    def end_session_guarded(
        self, request_id: str, generation: str
    ) -> MessagingFuture[Any]:
        """Reject another incarnation without pretending to clear its leases."""
        self.calls.append((request_id, generation))
        if generation != self.generation:
            return ready(False)
        return self.replies.popleft() if self.replies else ready(True)


def test_unconfirmed_end_prevents_shutdown_without_discarding_ownership() -> None:
    """A socket send alone is not evidence that the server released its lease."""
    peer = Peer()
    reply: MessagingFuture[Any] = MessagingFuture()
    peer.replies.append(reply)
    health = threading.Event()
    health.set()
    owner = ConfirmedSessionCleanup(
        {"peer": cast(RequestClient, peer)}, {"peer": health}, 1
    )
    try:
        owner.retire("r")
        wait_until(lambda: bool(peer.calls))
        with pytest.raises(RuntimeError, match="unconfirmed"):
            owner.close(0.001)
        assert owner.report_status()["pending"] == 1
        reply.set_result(True)
        wait_until(lambda: owner.report_status()["pending"] == 0)
    finally:
        reply.set_result(True)
        owner.close(1)


def test_lost_end_ack_retries_original_generation_and_ignores_late_reply() -> None:
    """A retry confirms the original obligation; a late older reply cannot undo it."""
    peer = Peer()
    lost: MessagingFuture[Any] = MessagingFuture()
    peer.replies.append(lost)
    health = threading.Event()
    health.set()
    owner = ConfirmedSessionCleanup(
        {"peer": cast(RequestClient, peer)}, {"peer": health}, 0.02
    )
    try:
        owner.retire("r")
        wait_until(lambda: owner.report_status()["confirmed"] == 1)
        assert peer.calls == [("r", "original"), ("r", "original")]
        assert owner.report_status()["retries"] == 1
        lost.set_result(False)
        assert owner.report_status()["pending"] == 0
        assert owner.recover("peer")
    finally:
        owner.close(1)


def test_different_generation_cannot_acknowledge_original_server_cleanup() -> None:
    """A changed endpoint retains ownership until the original incarnation returns."""
    peer = Peer()
    health = threading.Event()
    health.set()
    owner = ConfirmedSessionCleanup(
        {"peer": cast(RequestClient, peer)}, {"peer": health}, 0.02
    )
    try:
        peer.generation = "replacement"
        owner.retire("r")
        wait_until(lambda: owner.report_status()["requests"][0]["generation_mismatch"])
        assert not owner.recover("peer")
        assert not health.is_set()
        with pytest.raises(RuntimeError, match="unconfirmed"):
            owner.close(0.001)
        assert owner.report_status()["generations"] == {"peer": "original"}
    finally:
        peer.generation = "original"
        owner.recover("peer")
        wait_until(lambda: owner.report_status()["pending"] == 0)
        owner.close(1)
