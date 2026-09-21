# SPDX-License-Identifier: Apache-2.0
"""Configuration for the peer DRAM/CXLMemSim retention pools."""

# Standard
from dataclasses import dataclass
from typing import Literal
import math


@dataclass(frozen=True)
class HybridMemoryConfig:
    """Quotas exclude SHM headers; forced placement is a diagnostic hook only."""

    dram_bytes: int
    cxl_bytes: int
    bulk_control: str
    modeled_control: str
    client_library: str
    alignment: int = 4096
    policy: Literal["dram_first", "force_dram", "force_cxl"] = "dram_first"
    completion_timeout_s: float = 30.0
    poll_interval_s: float = 0.0002
    diagnostic_failure: Literal["store_partial", "load_partial"] | None = None
    diagnostic_lookup_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.dram_bytes < 0 or self.cxl_bytes <= 0:
            raise ValueError("DRAM quota must be nonnegative and CXL positive")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if any(n % self.alignment for n in (self.dram_bytes, self.cxl_bytes)):
            raise ValueError("pool quotas must be aligned")
        if self.policy not in ("dram_first", "force_dram", "force_cxl"):
            raise ValueError("unsupported hybrid placement policy")
        if self.completion_timeout_s <= 0 or self.poll_interval_s <= 0:
            raise ValueError("completion timeout and polling interval must be positive")
        if self.diagnostic_failure not in (None, "store_partial", "load_partial"):
            raise ValueError("unsupported diagnostic failure")
        if (
            not math.isfinite(self.diagnostic_lookup_delay_s)
            or not 0 <= self.diagnostic_lookup_delay_s <= 5
        ):
            raise ValueError(
                "diagnostic lookup delay must be finite and between 0 and 5s"
            )
