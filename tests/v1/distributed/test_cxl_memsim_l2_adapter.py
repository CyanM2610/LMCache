# SPDX-License-Identifier: Apache-2.0
"""Tests for the CXLMemSim MP L2 adapter."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.cxl_memsim_client import BulkClientStats
from lmcache.v1.distributed.l2_adapters.cxl_memsim_l2_adapter import (
    CxlMemSimL2Adapter,
    CxlMemSimL2AdapterConfig,
)

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


@dataclass
class _FakeClient:
    capacity: int = 16 * 1024
    closed: bool = False

    def snapshot_stats(self) -> BulkClientStats:
        return BulkClientStats()

    def close(self) -> None:
        self.closed = True


class _ClientFactory:
    def __init__(self, *, capacity: int = 16 * 1024) -> None:
        self.capacity = capacity
        self.calls: list[dict[str, object]] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, **kwargs: object) -> _FakeClient:
        self.calls.append(kwargs)
        client = _FakeClient(capacity=self.capacity)
        self.clients.append(client)
        return client


def _config(**overrides: object) -> CxlMemSimL2AdapterConfig:
    values: dict[str, object] = {
        "client_library": "/tmp/libcxlmemsim_client.so",
        "slot_bytes": 4096,
        "num_store_workers": 1,
        "num_lookup_workers": 1,
        "num_load_workers": 1,
    }
    values.update(overrides)
    return CxlMemSimL2AdapterConfig.from_dict(values)


def _make_adapter(
    *,
    factory: _ClientFactory | None = None,
    **config_overrides: object,
) -> tuple[CxlMemSimL2Adapter, _ClientFactory]:
    client_factory = factory or _ClientFactory()
    adapter = CxlMemSimL2Adapter(
        _config(**config_overrides),
        client_factory=client_factory,
    )
    return adapter, client_factory


def test_cxl_memsim_config_parses_and_registers_adapter_type() -> None:
    config = CxlMemSimL2AdapterConfig.from_dict(
        {
            "type": "cxl_memsim",
            "client_library": " /opt/cxl/libcxlmemsim_client.so ",
            "slot_bytes": 8192,
            "control_name": " /lmcache_test ",
            "offset_bytes": 4096,
            "capacity_bytes": 32768,
            "timeout_ms": 7000,
            "num_store_workers": 2,
            "num_lookup_workers": 3,
            "num_load_workers": 4,
        }
    )

    assert config.client_library == "/opt/cxl/libcxlmemsim_client.so"
    assert config.control_name == "/lmcache_test"
    assert config.slot_bytes == 8192
    assert config.offset_bytes == 4096
    assert config.capacity_bytes == 32768
    assert config.timeout_ms == 7000
    assert config.num_store_workers == 2
    assert config.num_lookup_workers == 3
    assert config.num_load_workers == 4
    assert "cxl_memsim" in get_registered_l2_adapter_types()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("client_library", "", "client_library"),
        ("control_name", "", "control_name"),
        ("slot_bytes", 0, "slot_bytes"),
        ("slot_bytes", True, "slot_bytes"),
        ("offset_bytes", -1, "offset_bytes"),
        ("offset_bytes", True, "offset_bytes"),
        ("capacity_bytes", 0, "capacity_bytes"),
        ("capacity_bytes", True, "capacity_bytes"),
        ("timeout_ms", 0, "timeout_ms"),
        ("num_store_workers", 0, "num_store_workers"),
        ("num_lookup_workers", False, "num_lookup_workers"),
        ("num_load_workers", -1, "num_load_workers"),
    ],
)
def test_cxl_memsim_config_rejects_invalid_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    values: dict[str, object] = {
        "client_library": "/tmp/libcxlmemsim_client.so",
        "slot_bytes": 4096,
    }
    values[field] = value

    with pytest.raises(ValueError, match=match):
        CxlMemSimL2AdapterConfig.from_dict(values)


def test_cxl_memsim_config_requires_fields() -> None:
    with pytest.raises(ValueError, match="client_library"):
        CxlMemSimL2AdapterConfig.from_dict({"slot_bytes": 4096})
    with pytest.raises(ValueError, match="slot_bytes"):
        CxlMemSimL2AdapterConfig.from_dict(
            {"client_library": "/tmp/libcxlmemsim_client.so"}
        )


def test_cxl_memsim_adapter_validates_arena_and_opens_expected_client() -> None:
    adapter, factory = _make_adapter(offset_bytes=2048, capacity_bytes=12288)
    try:
        assert factory.calls == [
            {
                "library_path": "/tmp/libcxlmemsim_client.so",
                "control_name": "/cxlmemsim_bulk",
                "timeout_ms": 5000,
            }
        ]
        status = adapter.report_status()
        assert status["capacity_bytes"] == 12288
        assert status["max_slots"] == 3
        assert status["slot_bytes"] == 4096
        assert status["offset_bytes"] == 2048
    finally:
        adapter.close()
    assert factory.clients[0].closed


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"offset_bytes": 16 * 1024 + 1},
        {"offset_bytes": 8192, "capacity_bytes": 8193},
        {"capacity_bytes": 2048},
    ],
)
def test_cxl_memsim_adapter_rejects_out_of_bounds_or_empty_arena(
    config_overrides: dict[str, object],
) -> None:
    factory = _ClientFactory()

    with pytest.raises(ValueError, match="arena|capacity|offset|slot"):
        CxlMemSimL2Adapter(
            _config(**config_overrides),
            client_factory=factory,
        )

    assert factory.clients[0].closed


def test_cxl_memsim_adapter_has_distinct_eventfds_and_idempotent_close() -> None:
    adapter, factory = _make_adapter()
    event_fds = {
        adapter.get_store_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
        adapter.get_load_event_fd(),
    }
    assert len(event_fds) == 3

    adapter.close()
    adapter.close()

    assert factory.clients[0].closed


def test_cxl_memsim_adapter_rejects_tasks_after_close() -> None:
    adapter, _ = _make_adapter()
    adapter.close()

    with pytest.raises(RuntimeError, match="closing|closed"):
        adapter.submit_lookup_and_lock_task([], _EMPTY_LAYOUT)


def test_client_library_path_need_not_exist_until_adapter_open(tmp_path: Path) -> None:
    path = tmp_path / "missing-client.so"
    config = _config(client_library=str(path))
    assert config.client_library == str(path)


def test_client_factory_protocol_is_runtime_injectable() -> None:
    factory: Callable[..., _FakeClient] = _ClientFactory()
    adapter = CxlMemSimL2Adapter(_config(), client_factory=factory)
    adapter.close()
