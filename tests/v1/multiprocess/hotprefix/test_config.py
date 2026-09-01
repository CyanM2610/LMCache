# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import (
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.protocols.hotprefix import (
    parse_hotprefix_store_request,
)


def test_hotprefix_server_options_round_trip() -> None:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)

    config = parse_args_to_mp_server_config(
        parser.parse_args(
            [
                "--hotprefix-host-capacity-bytes",
                "4096",
                "--hotprefix-frequency-threshold",
                "3",
                "--hotprefix-aging-interval",
                "17",
                "--hotprefix-lease-ttl-seconds",
                "2.5",
            ]
        )
    )

    assert config.hotprefix_host_capacity_bytes == 4096
    assert config.hotprefix_frequency_threshold == 3
    assert config.hotprefix_aging_interval == 17
    assert config.hotprefix_lease_ttl_seconds == 2.5


def test_hotprefix_store_request_parser_is_strict() -> None:
    prefix_id = b"logical-prefix"
    request_id = f"__hotprefix_store__:42:{prefix_id.hex()}"

    assert parse_hotprefix_store_request(request_id) == (42, prefix_id)
    with pytest.raises(ValueError, match="malformed"):
        parse_hotprefix_store_request("__hotprefix_store__:0:00")
    with pytest.raises(ValueError, match="not a HotPrefix"):
        parse_hotprefix_store_request("request:42:00")
