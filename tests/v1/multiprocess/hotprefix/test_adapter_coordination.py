# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_module
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.hotprefix import (
    HotPrefixAdmissionResponse,
    HotPrefixHostCandidate,
    HotPrefixTransferTicket,
)


class _Future:
    def __init__(self, value: object = None, error: Exception | None = None) -> None:
        self._value = value
        self._error = error

    def result(self, timeout: float) -> object:
        del timeout
        if self._error is not None:
            raise self._error
        return self._value


def _scheduler_adapter() -> LMCacheMPSchedulerAdapter:
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter._hotprefix_enabled = True
    adapter._server_urls = ["server-a", "server-b"]
    adapter._mq_timeout = 1.0
    adapter._health_events = {url: threading.Event() for url in adapter._server_urls}
    for event in adapter._health_events.values():
        event.set()
    adapter.mq_clients = {url: MagicMock(name=url) for url in adapter._server_urls}
    adapter._hotprefix_control_lock = threading.Lock()
    adapter._hotprefix_control_observations = []
    adapter.lmcache_tokens_per_chunk = 16
    adapter.model_name = "model"
    adapter.parallel_strategy = MagicMock(
        kv_world_size=1,
        kv_tp_size=1,
        num_kv_readers=1,
    )
    return adapter


def test_scheduler_heartbeats_start_lazily_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    adapter._heartbeat_interval = 0.25
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeats = {}
    created: list[MagicMock] = []

    def make_heartbeat(**kwargs: object) -> MagicMock:
        heartbeat = MagicMock(**kwargs)
        created.append(heartbeat)
        return heartbeat

    monkeypatch.setattr(adapter_module, "HeartbeatThread", make_heartbeat)

    adapter._ensure_heartbeat_started()
    adapter._ensure_heartbeat_started()

    assert set(adapter._heartbeats) == set(adapter._server_urls)
    assert len(created) == len(adapter._server_urls)
    for heartbeat in created:
        heartbeat.start.assert_called_once_with()


def test_hotprefix_retrieve_prepares_exact_range_read_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", MagicMock())
    client_to_url = {client: url for url, client in adapter.mq_clients.items()}
    calls: list[tuple[str, RequestType]] = []

    def send(client: Any, request_type: RequestType, payloads: list[Any]):
        url = client_to_url[client]
        calls.append((url, request_type))
        if request_type is RequestType.WAIT_PREFETCH_STATUS:
            assert payloads == ["promotion-1", 1.0]
            return _Future(2)
        assert request_type is RequestType.LOOKUP
        key = payloads[0]
        assert key.start == 16
        assert key.end == 48
        return _Future(None)

    monkeypatch.setattr(adapter_module, "send_lmcache_request", send)

    assert adapter.prepare_hotprefix_retrieve(
        "promotion-1", list(range(64)), 16, 48, "tenant"
    )
    assert calls == [
        ("server-a", RequestType.LOOKUP),
        ("server-b", RequestType.LOOKUP),
        ("server-a", RequestType.WAIT_PREFETCH_STATUS),
        ("server-b", RequestType.WAIT_PREFETCH_STATUS),
    ]


def test_hotprefix_retrieve_short_hit_releases_partial_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", MagicMock())
    client_to_url = {client: url for url, client in adapter.mq_clients.items()}
    calls: list[tuple[str, RequestType]] = []

    def send(client: Any, request_type: RequestType, payloads: list[Any]):
        del payloads
        url = client_to_url[client]
        calls.append((url, request_type))
        if request_type is RequestType.WAIT_PREFETCH_STATUS:
            return _Future(2 if url == "server-a" else 1)
        return _Future(None)

    monkeypatch.setattr(adapter_module, "send_lmcache_request", send)

    assert not adapter.prepare_hotprefix_retrieve(
        "promotion-1", list(range(64)), 16, 48, "tenant"
    )
    assert calls == [
        ("server-a", RequestType.LOOKUP),
        ("server-b", RequestType.LOOKUP),
        ("server-a", RequestType.WAIT_PREFETCH_STATUS),
        ("server-b", RequestType.WAIT_PREFETCH_STATUS),
        ("server-a", RequestType.FREE_LOOKUP_LOCKS),
        ("server-b", RequestType.FREE_LOOKUP_LOCKS),
        ("server-a", RequestType.END_SESSION),
        ("server-b", RequestType.END_SESSION),
    ]


def test_partial_admission_rolls_back_every_contacted_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    client_to_url = {client: url for url, client in adapter.mq_clients.items()}
    calls: list[tuple[str, RequestType]] = []

    def send(client: Any, request_type: RequestType, payloads: list[Any]):
        url = client_to_url[client]
        calls.append((url, request_type))
        if request_type is RequestType.HOT_PREFIX_ADMIT:
            if url == "server-b":
                return _Future(error=TimeoutError())
            return _Future(
                HotPrefixAdmissionResponse("accept", "admitted", [], int(payloads[3]))
            )
        assert request_type is RequestType.HOT_PREFIX_ABORT
        return _Future(True)

    monkeypatch.setattr(adapter_module, "send_lmcache_request", send)

    assert adapter.hotprefix_admit(b"namespace", b"prefix", 4096) is None
    assert not adapter._health_events["server-b"].is_set()
    assert calls == [
        ("server-a", RequestType.HOT_PREFIX_ADMIT),
        ("server-b", RequestType.HOT_PREFIX_ADMIT),
        ("server-a", RequestType.HOT_PREFIX_ABORT),
        ("server-b", RequestType.HOT_PREFIX_ABORT),
    ]


def test_candidates_are_intersected_by_generation_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    client_to_url = {client: url for url, client in adapter.mq_clients.items()}
    shared = HotPrefixHostCandidate(b"shared", 4096, 7, 3, 4)
    mismatched_a = HotPrefixHostCandidate(b"mismatch", 4096, 8, 2, 3)
    mismatched_b = HotPrefixHostCandidate(b"mismatch", 8192, 8, 2, 3)

    def send(client: Any, request_type: RequestType, payloads: list[Any]):
        del payloads
        assert request_type is RequestType.HOT_PREFIX_CANDIDATES
        values = (
            [shared, mismatched_a]
            if client_to_url[client] == "server-a"
            else [shared, mismatched_b]
        )
        return _Future(values)

    monkeypatch.setattr(adapter_module, "send_lmcache_request", send)

    assert adapter.hotprefix_candidates(b"namespace", [b"mismatch", b"shared"]) == [
        shared
    ]
    observations = adapter.drain_hotprefix_control_stats()["control_observations"]
    assert len(observations) == 1
    assert observations[0]["method"] == "candidates"
    assert observations[0]["outcome"] == "success"
    assert float(observations[0]["duration_seconds"]) >= 0
    assert adapter.drain_hotprefix_control_stats() == {"control_observations": []}


def test_partial_acquire_releases_ticket_on_every_contacted_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _scheduler_adapter()
    client_to_url = {client: url for url, client in adapter.mq_clients.items()}
    candidate = HotPrefixHostCandidate(b"prefix", 4096, 7, 3, 4)
    calls: list[tuple[str, RequestType]] = []

    def send(client: Any, request_type: RequestType, payloads: list[Any]):
        url = client_to_url[client]
        calls.append((url, request_type))
        if request_type is RequestType.HOT_PREFIX_ACQUIRE:
            if url == "server-b":
                return _Future(error=RuntimeError("server failed"))
            ticket_id = bytes(payloads[3])
            return _Future(HotPrefixTransferTicket(ticket_id, b"prefix", 7, 4096))
        assert request_type is RequestType.HOT_PREFIX_RELEASE
        return _Future(True)

    monkeypatch.setattr(adapter_module, "send_lmcache_request", send)

    assert adapter.hotprefix_acquire(b"namespace", candidate) is None
    assert not adapter._health_events["server-b"].is_set()
    assert calls == [
        ("server-a", RequestType.HOT_PREFIX_ACQUIRE),
        ("server-b", RequestType.HOT_PREFIX_ACQUIRE),
        ("server-a", RequestType.HOT_PREFIX_RELEASE),
        ("server-b", RequestType.HOT_PREFIX_RELEASE),
    ]


def test_request_finished_cleans_access_only_hotprefix_state() -> None:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = MagicMock(hotprefix_enabled=True)
    request = SimpleNamespace(request_id="access-only", kv_transfer_params=None)

    assert connector.request_finished(request, []) == (False, None)

    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        "access-only"
    )
    connector.scheduler_adapter.end_session.assert_called_once_with("access-only")
