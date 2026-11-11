# SPDX-License-Identifier: Apache-2.0
"""Configuration for peer DRAM and modeled or physical CXL retention pools."""

# Standard
from dataclasses import dataclass
from typing import Literal
import math


@dataclass(frozen=True)
class HybridMemoryConfig:
    """Quotas exclude SHM headers; forced placement is a diagnostic hook only."""

    dram_bytes: int
    cxl_bytes: int
    bulk_control: str = ""
    modeled_control: str = ""
    client_library: str = ""
    alignment: int = 4096
    policy: Literal["dram_first", "force_dram", "force_cxl", "all_lru"] = "dram_first"
    completion_timeout_s: float = 30.0
    poll_interval_s: float = 0.0002
    diagnostic_failure: Literal["store_partial", "load_partial"] | None = None
    diagnostic_lookup_delay_s: float = 0.0
    cxl_backend: Literal["modeled", "devdax"] = "modeled"
    dax_path: str | None = None
    dax_offset_bytes: int = 0
    dax_region_bytes: int = 0

    def __post_init__(self) -> None:
        if any(
            type(n) is not int
            for n in (self.dram_bytes, self.cxl_bytes, self.alignment)
        ):
            raise ValueError("pool quotas and alignment must be integer bytes")
        if self.cxl_backend not in ("modeled", "devdax"):
            raise ValueError("unknown CXL backing")
        if self.cxl_backend == "devdax":
            if self.bulk_control or self.modeled_control or self.client_library:
                raise ValueError("physical DAX must not use modeled endpoints")
            if any(
                type(n) is not int or n < 0
                for n in (
                    self.dax_offset_bytes,
                    self.dax_region_bytes,
                )
            ):
                raise ValueError(
                    "DAX offset and assigned region must be nonnegative integers"
                )
            if self.cxl_bytes and (
                not self.dax_path or self.dax_region_bytes < self.cxl_bytes
            ):
                raise ValueError(
                    "physical CXL requires a device and an assigned region "
                    "covering its quota"
                )
        if min(self.dram_bytes, self.cxl_bytes) < 0 or (
            self.dram_bytes + self.cxl_bytes == 0
        ):
            raise ValueError("pool quotas must be nonnegative with positive total")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if any(n % self.alignment for n in (self.dram_bytes, self.cxl_bytes)):
            raise ValueError("pool quotas must be aligned")
        if self.policy not in ("dram_first", "force_dram", "force_cxl", "all_lru"):
            raise ValueError("unsupported hybrid placement policy")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.completion_timeout_s,
                self.poll_interval_s,
            )
        ):
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
