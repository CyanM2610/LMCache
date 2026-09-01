# SPDX-License-Identifier: Apache-2.0
"""vLLM connector metrics for client-observed HotPrefix control RPCs."""

# Standard
from dataclasses import dataclass
from typing import Any
import math

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)


@dataclass
class HotPrefixKVConnectorStats(KVConnectorStats):
    """Serializable interval observations for HotPrefix control fanout."""

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        """Reset observations for a new interval."""
        self.data: dict[str, Any] = {"control_observations": []}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Append another interval's observations."""
        self.data.setdefault("control_observations", []).extend(
            other.data.get("control_observations", [])
        )
        return self

    def reduce(self) -> dict[str, int | float]:
        """Return a compact logging summary."""
        observations = self.data.get("control_observations", [])
        durations = sorted(float(item["duration_seconds"]) for item in observations)
        total_bytes = sum(
            int(item["request_bytes"]) + int(item["response_bytes"])
            for item in observations
        )
        p95_index = max(0, math.ceil(0.95 * len(durations)) - 1)
        return {
            "HotPrefix control RPCs": len(observations),
            "HotPrefix control p95 (ms)": (
                1000 * durations[p95_index] if durations else 0.0
            ),
            "HotPrefix control bytes": total_bytes,
        }

    def is_empty(self) -> bool:
        """Return whether the interval contains no control observations."""
        return not self.data.get("control_observations")


class HotPrefixPromMetrics(KVConnectorPromMetrics):
    """Prometheus adapter for HotPrefix connector stats."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._duration = self._histogram_cls(
            name="vllm:hotprefix_control_rpc_seconds",
            documentation="Client-observed HotPrefix control fanout latency.",
            labelnames=labelnames + ["method", "outcome"],
            buckets=(
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
            ),
        )
        self._bytes = self._counter_cls(
            name="vllm:hotprefix_control_rpc_bytes",
            documentation="Estimated HotPrefix control fanout payload bytes.",
            labelnames=labelnames + ["method", "direction"],
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record one scheduler stats interval."""
        labelvalues = self.per_engine_labelvalues[engine_idx]
        for item in transfer_stats_data.get("control_observations", []):
            method = str(item["method"])
            outcome = str(item["outcome"])
            self._duration.labels(*labelvalues, method, outcome).observe(
                float(item["duration_seconds"])
            )
            self._bytes.labels(*labelvalues, method, "request").inc(
                int(item["request_bytes"])
            )
            self._bytes.labels(*labelvalues, method, "response").inc(
                int(item["response_bytes"])
            )
