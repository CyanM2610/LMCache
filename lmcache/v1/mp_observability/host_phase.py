# SPDX-License-Identifier: Apache-2.0
"""Opt-in actual host call envelopes, not simulated CPU service costs."""

# Standard
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable
import itertools
import os
import threading
import time

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

_parent: ContextVar[str | None] = ContextVar("l2_host_phase_parent", default=None)
_ids = itertools.count()
_errors: list[str] = []


def phase_publish(event: Event) -> None:
    """Publish a diagnostic event; retain failures without changing native work."""
    try:
        get_event_bus().publish(event)
    except Exception as error:
        _errors.append(repr(error))


def host_phase_errors() -> list[str]:
    """Return process-local observer failures for terminal evidence accounting."""
    return list(_errors)


def observed_mp_call(name: str) -> Callable:
    """Bracket a real host method when phase timing is explicitly enabled.

    Args:
        name: Stable phase label. Nested calls retain the parent span identity.
    Returns:
        A decorator preserving native return values and exceptions. Durations
        include native waits and nested work; wall/thread/process CPU are separate.
    """

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if os.environ.get("L2_HYBRID_PHASE_TIMING") != "1":
                return function(*args, **kwargs)
            key = args[1] if len(args) > 1 else kwargs.get("key")
            request_id = getattr(key, "request_id", "")
            span = f"{os.getpid()}:{next(_ids)}"
            parent = _parent.get()
            token = _parent.set(span)
            before = (
                time.monotonic_ns(),
                time.thread_time_ns(),
                time.process_time_ns(),
            )
            error = None
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                after = (
                    time.monotonic_ns(),
                    time.thread_time_ns(),
                    time.process_time_ns(),
                )
                _parent.reset(token)
                phase_publish(
                    Event(
                        event_type=EventType.MP_HOST_PHASE,
                        session_id=request_id,
                        metadata=dict(
                            name=name,
                            span_id=span,
                            parent_span_id=parent,
                            request_id=request_id,
                            token_start=getattr(key, "start", None),
                            token_end=getattr(key, "end", None),
                            begin_wall_ns=before[0],
                            end_wall_ns=after[0],
                            begin_thread_cpu_ns=before[1],
                            end_thread_cpu_ns=after[1],
                            begin_process_cpu_ns=before[2],
                            end_process_cpu_ns=after[2],
                            thread_id=threading.get_ident(),
                            error=error,
                        ),
                    )
                )

        return wrapped

    return decorate
