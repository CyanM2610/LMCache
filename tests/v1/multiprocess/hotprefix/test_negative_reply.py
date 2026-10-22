# SPDX-License-Identifier: Apache-2.0
"""A valid negative publication reply must not disable its rollback RPC."""

from concurrent.futures import Future
import threading

from lmcache.integration.vllm import vllm_multi_process_adapter as module
from lmcache.v1.multiprocess.protocols.base import RequestType


def adapter():
    value = module.LMCacheMPSchedulerAdapter.__new__(module.LMCacheMPSchedulerAdapter)
    value._hotprefix_enabled = True
    value._hotprefix_observability_enabled = False
    value._hotprefix_trace_enabled = False
    value._server_urls = ["a", "b"]
    value._mq_timeout = 0.1
    value.mq_clients = {name: name for name in value._server_urls}
    value._health_events = {name: threading.Event() for name in value._server_urls}
    for event in value._health_events.values():
        event.set()
    return value


def test_negative_publication_reply_still_allows_reservation_rollback(monkeypatch):
    client = adapter()
    reserved = {"a", "b"}
    calls = []

    def send(server, request_type, payload):
        calls.append((server, request_type))
        future = Future()
        if request_type is RequestType.HOT_PREFIX_PUBLISH:
            future.set_result(False)  # Complete RPC; physical residency absent.
        else:
            assert request_type is RequestType.HOT_PREFIX_ABORT
            reserved.discard(server)
            future.set_result(True)
        return future

    monkeypatch.setattr(module, "send_lmcache_request", send)
    assert client.hotprefix_publish(b"namespace", b"prefix") is False
    assert client.hotprefix_abort(b"namespace", b"prefix") is True
    assert not reserved
    assert client.is_healthy
    assert calls[-2:] == [
        ("a", RequestType.HOT_PREFIX_ABORT),
        ("b", RequestType.HOT_PREFIX_ABORT),
    ]


def test_transport_timeout_still_marks_server_unhealthy(monkeypatch):
    client = adapter()

    def send(*args):
        future = Future()
        future.set_exception(TimeoutError())
        return future

    monkeypatch.setattr(module, "send_lmcache_request", send)
    assert client.hotprefix_publish(b"namespace", b"prefix") is False
    assert not client.is_healthy


def test_abort_after_timeout_is_sent_even_when_unhealthy(monkeypatch):
    client = adapter()
    reserved = {"a", "b"}

    def send(server, request_type, payload):
        future = Future()
        if request_type is RequestType.HOT_PREFIX_PUBLISH:
            future.set_exception(TimeoutError())
        else:
            assert request_type is RequestType.HOT_PREFIX_ABORT
            reserved.discard(server)
            future.set_result(True)
        return future

    monkeypatch.setattr(module, "send_lmcache_request", send)
    assert client.hotprefix_publish(b"namespace", b"prefix") is False
    assert not client.is_healthy
    assert client.hotprefix_abort(b"namespace", b"prefix") is True
    assert not reserved
    assert not client.is_healthy  # Cleanup does not forge a healthy heartbeat.


def test_abort_attempts_remaining_servers_after_one_timeout(monkeypatch):
    client = adapter()
    for event in client._health_events.values():
        event.clear()
    calls = []

    def send(server, request_type, payload):
        assert request_type is RequestType.HOT_PREFIX_ABORT
        calls.append(server)
        future = Future()
        if server == "a":
            future.set_exception(TimeoutError())
        else:
            future.set_result(True)
        return future

    monkeypatch.setattr(module, "send_lmcache_request", send)
    assert client.hotprefix_abort(b"namespace", b"prefix") is False
    assert calls == ["a", "b"]
