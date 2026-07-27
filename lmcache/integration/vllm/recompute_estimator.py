# SPDX-License-Identifier: Apache-2.0
"""Calibrated prefill-cost estimates for external policy requests."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from statistics import median
import re
import threading


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RecomputeCalibrationKey:
    """Bounded-cardinality dimensions for one prefill calibration series."""

    model_name: str
    layout_fingerprint: str
    prompt_tokens_bucket: int
    batch_bucket: int
    load_bucket: int
    device: str

    def __post_init__(self) -> None:
        if not self.model_name or not self.device:
            raise ValueError("model_name and device must not be empty")
        if not _SHA256_RE.fullmatch(self.layout_fingerprint):
            raise ValueError("layout_fingerprint must be a lowercase SHA-256")
        if (
            min(
                self.prompt_tokens_bucket,
                self.batch_bucket,
                self.load_bucket,
            )
            < 0
        ):
            raise ValueError("calibration buckets must be non-negative")


class RecomputeEstimator:
    """Thread-safe rolling-median estimator with a conservative cold default."""

    calibration_version = "recompute-median-v1"

    def __init__(
        self,
        *,
        default_estimate_ns: int,
        max_samples: int = 64,
    ) -> None:
        if default_estimate_ns <= 0 or max_samples <= 0:
            raise ValueError("default estimate and max_samples must be positive")
        self._default_estimate_ns = default_estimate_ns
        self._max_samples = max_samples
        self._samples: dict[RecomputeCalibrationKey, deque[int]] = {}
        self._lock = threading.RLock()

    def record(self, key: RecomputeCalibrationKey, elapsed_ns: int) -> None:
        """Record one raw, positive prefill observation for a calibration key."""
        if elapsed_ns <= 0:
            raise ValueError("elapsed_ns must be positive")
        with self._lock:
            samples = self._samples.setdefault(key, deque(maxlen=self._max_samples))
            samples.append(elapsed_ns)

    def estimate(self, key: RecomputeCalibrationKey) -> int:
        """Return the rolling median or the cold-start default."""
        with self._lock:
            samples = self._samples.get(key)
            if not samples:
                return self._default_estimate_ns
            return max(1, int(median(samples)))

    def raw_observations(self, key: RecomputeCalibrationKey) -> tuple[int, ...]:
        """Return retained raw observations for reproducible calibration."""
        with self._lock:
            return tuple(self._samples.get(key, ()))


def calibration_bucket(value: int) -> int:
    """Round a non-negative calibration dimension up to a power of two."""
    if value < 0:
        raise ValueError("calibration value must be non-negative")
    return 0 if value == 0 else 1 << (value - 1).bit_length()
