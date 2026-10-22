# SPDX-License-Identifier: Apache-2.0
"""Optional model-call host boundary for explanatory native005 profiles only."""

# Standard
from typing import Any

# Third Party
from vllm.v1.worker.gpu.phase_observation import phase_span
import torch


class ModelDispatchObserver:
    """Observe the public module call without changing inputs, output or waits."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Install paired host-only hooks on the actual instantiated model.

        Args:
            model: Existing native model; no replacement or redispatch occurs.
        Notes:
            Phase clocks enclose profiler-range overhead as well as model dispatch.
            Hooks return None and preserve model exceptions. Observation failures
            are retained in errors for evidence rejection, never raised to model.
        """
        self.errors: list[str] = []
        self.pending: list[list[Any]] = []
        self.handles = [
            model.register_forward_pre_hook(self._begin),
            model.register_forward_hook(self._end, always_call=True),
        ]

    def close(self) -> None:
        """Remove these observation hooks without calling or waiting on the model."""
        for handle in self.handles:
            handle.remove()

    def status(self) -> dict[str, Any]:
        """Return pending boundaries and errors; nonzero values invalidate evidence."""
        return dict(pending=len(self.pending), errors=list(self.errors))

    def _begin(self, module: Any, args: Any) -> None:
        contexts: list[Any] = []
        self.pending.append(contexts)
        try:
            for context in (
                phase_span("model_dispatch"),
                torch.profiler.record_function("native005.model_dispatch"),
            ):
                context.__enter__()
                contexts.append(context)
        except Exception as error:
            self.errors.append(repr(error))

    def _end(self, module: Any, args: Any, output: Any) -> None:
        if not self.pending:
            self.errors.append("model exit without entry")
            return
        for context in reversed(self.pending.pop()):
            try:
                context.__exit__(None, None, None)
            except Exception as error:
                self.errors.append(repr(error))
