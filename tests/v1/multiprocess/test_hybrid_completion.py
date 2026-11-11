# SPDX-License-Identifier: Apache-2.0
"""Controlled-fence unit tests. Real CUDA/data paths are tested by the 7B probes."""

# Future
from __future__ import annotations

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
import threading

# First Party
from lmcache.v1.multiprocess import hybrid_completion as mod
from lmcache.v1.multiprocess.hybrid_completion import HybridAccess, HybridCompletionLoop


def make_loop(monkeypatch):
    monkeypatch.setattr(mod.torch_dev, "device", lambda _device: nullcontext())
    client = ManualModel()
    memory = SimpleNamespace(
        client=client,
        config=SimpleNamespace(
            completion_timeout_s=1.0,
            poll_interval_s=0.001,
            diagnostic_failure=None,
        ),
    )
    return HybridCompletionLoop(memory), client, SimpleNamespace(device="test")


def test_model_and_cuda_must_both_finish_without_head_of_line_blocking(monkeypatch):
    """Ready second operations may reply while the first retains its DMA hold."""
    loop, client, context = make_loop(monkeypatch)
    first, second = ManualFence(), ManualFence(True)
    client.gates[2].set()
    finalize = lambda ok: (b"event", ok)
    f1 = loop.defer(first, context, [HybridAccess({}, token=1)], True, finalize, "r1#1")
    f2 = loop.defer(
        second, context, [HybridAccess({}, token=2)], True, finalize, "r2#2"
    )
    try:
        assert f2.result(timeout=2) == (b"event", True)
        assert not f1.done()
        assert not f1.cancel(), "submitted copy ownership cannot be cancelled"
        client.observed.clear()
        first.ready.set()
        assert client.observed.wait(2)
        assert not f1.done(), "DMA alone must not bypass modeled completion"
        client.gates[1].set()
        assert f1.result(timeout=2) == (b"event", True)
        loop.drain_context(context)
        assert loop.report_status()["peak_pending"] == 2
    finally:
        first.ready.set()
        client.gates[1].set()
        loop.close()


def test_failed_submitted_copy_keeps_reservation_until_device_fence(monkeypatch):
    """Failure status does not license reuse before the queued device work drains."""
    loop, client, context = make_loop(monkeypatch)
    fence = ManualFence()
    finalized = []

    def finalize(ok):
        finalized.append(ok)
        return b"event", ok

    future = loop.defer(
        fence, context, [HybridAccess({}, token=1)], False, finalize, "failed#1"
    )
    try:
        assert fence.queried.wait(2)
        assert not future.done() and not finalized and not client.reported
        fence.ready.set()
        assert future.result(timeout=2) == (b"event", False)
        assert finalized == [False]
        loop.drain_context(context)
        assert loop.report_status()["failed"] == 1
    finally:
        fence.ready.set()
        loop.close()


def test_cuda_query_error_cannot_pretend_the_copy_drained(monkeypatch):
    """Only a later successful terminal query may release a failed operation."""
    loop, _client, context = make_loop(monkeypatch)
    fence = ManualFence()
    fence.broken = True
    future = loop.defer(fence, context, [], True, lambda ok: (b"event", ok), "error#1")
    try:
        assert fence.queried.wait(2)
        assert not future.done()
        fence.broken = False
        fence.ready.set()
        assert future.result(timeout=2) == (b"event", False)
    finally:
        fence.broken = False
        fence.ready.set()
        loop.close()


def test_combined_control_failure_cannot_publish_success(monkeypatch):
    """After the DMA fence, a failed metadata exchange retires the backend."""
    loop, client, context = make_loop(monkeypatch)
    fence = ManualFence()
    abandoned = []

    def fail_after_report(token, success):
        client.data_complete(token, success)
        raise RuntimeError("control server stopped")

    monkeypatch.setattr(client, "complete_and_poll", fail_after_report)
    monkeypatch.setattr(client, "abandon_drained", abandoned.append, raising=False)
    future = loop.defer(
        fence,
        context,
        [HybridAccess({}, token=1)],
        True,
        lambda ok: (b"event", ok),
        "control-error#1",
    )
    try:
        assert fence.queried.wait(2)
        assert not abandoned and not future.done()
        fence.ready.set()
        assert future.result(timeout=2) == (b"event", False)
        assert abandoned == [1]
        assert client.reported == {1: True}
        assert "control server stopped" in loop.report_status()["protocol_failure"]
    finally:
        fence.ready.set()
        loop.close()


class ManualModel:
    """Controllable terminal acknowledgments, without timing or payload simulation."""

    def __init__(self) -> None:
        self.gates = {1: threading.Event(), 2: threading.Event()}
        self.reported: dict[int, bool] = {}
        self.observed = threading.Event()

    def data_complete(self, token: int, success: bool) -> None:
        self.reported[token] = success
        self.observed.set()

    def complete_and_poll(self, token: int, success: bool) -> SimpleNamespace:
        """Report the controlled fence while preserving the independent model gate."""
        self.data_complete(token, success)
        return self.poll(token)

    def poll(self, token: int) -> SimpleNamespace:
        state = (
            5 if not self.reported[token] else (3 if self.gates[token].is_set() else 2)
        )
        return SimpleNamespace(
            state=state, queue_ns=0, service_ns=1, modeled_complete_ns=1
        )


class ManualFence:
    """Query-only fence: completion code cannot hide a synchronize call here."""

    def __init__(self, ready: bool = False) -> None:
        self.ready = threading.Event()
        self.queried = threading.Event()
        self.broken = False
        if ready:
            self.ready.set()

    def query(self) -> bool:
        broken = self.broken
        self.queried.set()
        if broken:
            raise RuntimeError("query failure is not DMA completion")
        return self.ready.is_set()
