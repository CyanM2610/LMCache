# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multiprocess config parsing (coordinator registration).

Covers CLI flags, ``LMCACHE_COORDINATOR_*`` env fallback, flag-over-env
precedence, and heartbeat-interval validation. This module imports only the
pure config layer (no native extensions), so it runs without a CUDA build.
"""

# Standard
import argparse
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import (
    CXLSharedTierConfig,
    CoordinatorConfig,
    MPServerConfig,
    add_coordinator_args,
    add_mp_server_args,
    parse_args_to_coordinator_config,
    parse_args_to_mp_server_config,
)

_COORD_ENV = (
    "LMCACHE_COORDINATOR_URL",
    "LMCACHE_COORDINATOR_ADVERTISE_IP",
    "LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL",
)


def _parse(argv: list[str]) -> CoordinatorConfig:
    parser = argparse.ArgumentParser()
    add_coordinator_args(parser)
    return parse_args_to_coordinator_config(parser.parse_args(argv))


@pytest.fixture(autouse=True)
def _clear_coord_env(monkeypatch):
    """Isolate each test from any coordinator env vars in the environment."""
    for name in _COORD_ENV:
        monkeypatch.delenv(name, raising=False)


def test_defaults_disable_registration():
    config = _parse([])
    assert config.url == ""  # empty url => registration disabled
    assert config.advertise_ip == ""
    assert config.heartbeat_interval == 5.0


def test_flags_are_parsed():
    config = _parse(
        [
            "--coordinator-url",
            "http://coord:9300",
            "--coordinator-advertise-ip",
            "10.0.0.5",
            "--coordinator-heartbeat-interval",
            "2.5",
        ]
    )
    assert config.url == "http://coord:9300"
    assert config.advertise_ip == "10.0.0.5"
    assert config.heartbeat_interval == 2.5


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_URL", "http://env-coord:9300")
    monkeypatch.setenv("LMCACHE_COORDINATOR_ADVERTISE_IP", "192.168.1.2")
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", "3")
    config = _parse([])
    assert config.url == "http://env-coord:9300"
    assert config.advertise_ip == "192.168.1.2"
    assert config.heartbeat_interval == 3.0


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_URL", "http://env-coord:9300")
    config = _parse(["--coordinator-url", "http://flag-coord:9300"])
    assert config.url == "http://flag-coord:9300"


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_heartbeat_rejected(interval):
    # Non-positive and non-finite (nan/inf) values are all rejected.
    with pytest.raises(ValueError, match="finite number > 0"):
        _parse(["--coordinator-heartbeat-interval", interval])


@pytest.mark.parametrize("interval", ["nan", "inf"])
def test_invalid_heartbeat_from_env_rejected(monkeypatch, interval):
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", interval)
    with pytest.raises(ValueError, match="finite number > 0"):
        _parse([])


def test_garbage_env_heartbeat_rejected(monkeypatch):
    monkeypatch.setenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL", "abc")
    with pytest.raises(ValueError, match="not a number"):
        _parse([])


def _parse_mp(argv: list[str]) -> MPServerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parse_args_to_mp_server_config(parser.parse_args(argv))


def test_instance_id_defaults_to_uuid4():
    # No --instance-id flag => a random UUID v4 is minted.
    config = _parse_mp([])
    assert uuid.UUID(config.instance_id).version == 4


def test_instance_id_flag_is_preserved():
    config = _parse_mp(["--instance-id", "mp-server-7"])
    assert config.instance_id == "mp-server-7"


def test_instance_id_defaults_are_distinct():
    # Each parse without the flag gets its own id (no shared default).
    assert _parse_mp([]).instance_id != _parse_mp([]).instance_id


def test_instance_id_dataclass_default_is_distinct():
    # Direct construction (no CLI) also mints a fresh id per instance.
    assert MPServerConfig().instance_id != MPServerConfig().instance_id


def test_cxl_shared_tier_is_disabled_without_configuration():
    config = _parse_mp([])

    assert config.cxl_shared_tier == CXLSharedTierConfig()
    assert config.cxl_shared_tier.enabled is False


def test_cxl_shared_tier_parses_complete_opt_in_configuration():
    config = _parse_mp(
        [
            "--cxl-shared-tier-enabled",
            "--cxl-shared-tier-provider",
            "posix_shm",
            "--cxl-shared-tier-shm-name",
            "/beluga-cxl",
            "--cxl-shared-tier-capacity-bytes",
            "1048576",
            "--cxl-shared-tier-alignment-bytes",
            "4096",
            "--cxl-shared-tier-layout-id",
            "packed_kv_v1",
        ]
    )

    assert config.cxl_shared_tier == CXLSharedTierConfig(
        enabled=True,
        provider="posix_shm",
        shm_name="/beluga-cxl",
        capacity_bytes=1048576,
        alignment_bytes=4096,
        layout_id="packed_kv_v1",
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--cxl-shared-tier-enabled"], "shm_name"),
        (
            [
                "--cxl-shared-tier-enabled",
                "--cxl-shared-tier-shm-name",
                "/beluga-cxl",
            ],
            "capacity_bytes",
        ),
        (
            [
                "--cxl-shared-tier-enabled",
                "--cxl-shared-tier-shm-name",
                "/beluga-cxl",
                "--cxl-shared-tier-capacity-bytes",
                "1048576",
                "--cxl-shared-tier-alignment-bytes",
                "3000",
            ],
            "alignment_bytes",
        ),
    ],
)
def test_cxl_shared_tier_rejects_incomplete_or_invalid_opt_in(
    args: list[str], message: str
):
    with pytest.raises(ValueError, match=message):
        _parse_mp(args)


def test_cxl_shared_tier_requires_lmcache_driven_transfer_mode():
    with pytest.raises(ValueError, match="lmcache_driven"):
        _parse_mp(
            [
                "--supported-transfer-mode",
                "engine_driven",
                "--cxl-shared-tier-enabled",
                "--cxl-shared-tier-shm-name",
                "/beluga-cxl",
                "--cxl-shared-tier-capacity-bytes",
                "1048576",
            ]
        )
