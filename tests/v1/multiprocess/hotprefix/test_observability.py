# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.hotprefix_metrics import HotPrefixKVConnectorStats
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules.hotprefix import HotPrefixModule
import lmcache.v1.multiprocess.modules.hotprefix as hotprefix_module


def test_control_stats_aggregate_and_reduce() -> None:
    first = HotPrefixKVConnectorStats(
        data={
            "control_observations": [
                {
                    "method": "access",
                    "outcome": "success",
                    "duration_seconds": 0.001,
                    "request_bytes": 100,
                    "response_bytes": 20,
                }
            ]
        }
    )
    second = HotPrefixKVConnectorStats(
        data={
            "control_observations": [
                {
                    "method": "admit",
                    "outcome": "timeout",
                    "duration_seconds": 0.01,
                    "request_bytes": 40,
                    "response_bytes": 0,
                }
            ]
        }
    )

    first.aggregate(second)
    reduced = first.reduce()

    assert reduced["HotPrefix control RPCs"] == 2
    assert reduced["HotPrefix control p95 (ms)"] == 10
    assert reduced["HotPrefix control bytes"] == 160


def test_access_emits_one_start_decision_and_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_bus = MagicMock()
    monkeypatch.setattr(hotprefix_module, "get_event_bus", lambda: event_bus)
    module = HotPrefixModule(object(), frequency_threshold=1)  # type: ignore[arg-type]

    module.access(1, 1, b"namespace", [1, 2, 3, 4], 0)

    events = [call.args[0] for call in event_bus.publish.call_args_list]
    assert [event.event_type for event in events] == [
        EventType.HOTPREFIX_CONTROL_START,
        EventType.HOTPREFIX_DECISION,
        EventType.HOTPREFIX_CONTROL_END,
    ]
    assert events[-1].metadata["outcome"] == "success"
    assert events[-1].metadata["duration_ns"] >= events[-1].metadata["lock_wait_ns"]


def test_access_with_hotprefix_observability_off_publishes_no_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_bus = MagicMock()
    event_bus.has_subscribers.return_value = False
    monkeypatch.setattr(hotprefix_module, "get_event_bus", lambda: event_bus)
    module = HotPrefixModule(object(), frequency_threshold=1)  # type: ignore[arg-type]

    def fail_timing() -> int:
        raise AssertionError("off mode must use an untimed handler lock")

    monkeypatch.setattr(hotprefix_module.time, "monotonic_ns", fail_timing)

    module.access(1, 1, b"namespace", [1, 2, 3, 4], 0)

    event_bus.publish.assert_not_called()


def test_access_preserves_client_control_operation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_bus = MagicMock()
    event_bus.has_subscribers.return_value = True
    monkeypatch.setattr(hotprefix_module, "get_event_bus", lambda: event_bus)
    module = HotPrefixModule(object(), frequency_threshold=1)  # type: ignore[arg-type]

    module.access(1, 1, b"namespace", [1, 2, 3, 4], 0, "shared-operation")

    control_events = [
        call.args[0]
        for call in event_bus.publish.call_args_list
        if call.args[0].event_type
        in {EventType.HOTPREFIX_CONTROL_START, EventType.HOTPREFIX_CONTROL_END}
    ]
    assert [event.session_id for event in control_events] == [
        "shared-operation",
        "shared-operation",
    ]


def test_failed_handler_still_emits_one_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_bus = MagicMock()
    monkeypatch.setattr(hotprefix_module, "get_event_bus", lambda: event_bus)
    monkeypatch.setattr(
        hotprefix_module.GlobalHostPrefixTree,
        "observe",
        MagicMock(side_effect=RuntimeError("injected failure")),
    )
    module = HotPrefixModule(object(), frequency_threshold=1)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected failure"):
        module.access(1, 1, b"namespace", [1, 2, 3, 4], 0)

    events = [call.args[0] for call in event_bus.publish.call_args_list]
    assert [event.event_type for event in events] == [
        EventType.HOTPREFIX_CONTROL_START,
        EventType.HOTPREFIX_CONTROL_END,
    ]
    assert events[-1].metadata["outcome"] == "failure"
