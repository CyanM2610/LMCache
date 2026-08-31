# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse

# First Party
from lmcache.v1.multiprocess.config import (
    add_mp_server_args,
    parse_args_to_mp_server_config,
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
