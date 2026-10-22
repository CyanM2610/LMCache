# SPDX-License-Identifier: Apache-2.0
"""Lazy optional vLLM phase hooks; other adapter users need no vLLM import."""

# Standard
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, ContextManager
import os


def worker_observation_span(name: str) -> ContextManager:
    """Return an optional host-clock span; never change native waits or streams."""
    if os.environ.get("L2_HYBRID_PHASE_TIMING") != "1":
        return nullcontext()
    # Third Party
    from vllm.v1.worker.gpu.phase_observation import phase_span

    return phase_span(name)


def observed_retirement_call(name: str) -> Callable:
    """Enable read-only physical-lifetime witness callbacks only when requested."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if os.environ.get("L2_HYBRID_RETIREMENT_WITNESS") != "1":
                return function(*args, **kwargs)
            # Third Party
            from vllm.v1.core.retirement_observation import retirement_call

            return retirement_call(name)(function)(*args, **kwargs)

        return wrapped

    return decorate


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
