# SPDX-License-Identifier: Apache-2.0
"""Lazy optional vLLM phase hooks; other adapter users need no vLLM import."""

# Standard
from functools import wraps
from typing import Any, Callable
import os


def observed_worker_call(name: str) -> Callable:
    """Observe a worker call only in the explicitly enabled native phase probe."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if os.environ.get("L2_HYBRID_PHASE_TIMING") != "1":
                return function(*args, **kwargs)
            # Third Party
            from vllm.v1.worker.gpu.phase_observation import observed_host_call

            return observed_host_call(name)(function)(*args, **kwargs)

        return wrapped

    return decorate
