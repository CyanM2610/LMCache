# SPDX-License-Identifier: Apache-2.0
"""Read process environment outside measurement through the existing worker RPC."""

# Standard
from pathlib import Path
from typing import Any
import os
import threading
import time

# Third Party
import torch


def read_optional(path: Path) -> dict[str, Any]:
    """Read accessible environment metadata, preserving the reason when unavailable."""
    try:
        return dict(value=path.read_text().strip(), error=None)
    except OSError as error:
        return dict(value=None, error=repr(error))


def process_environment() -> dict[str, Any]:
    """Capture exact Torch threads and Linux affinity without changing either.

    Returns:
        JSON-compatible clocks, Torch thread counts, selected thread-related
        environment, all current thread affinities and NUMA/CPU frequency data.
    Notes:
        This runs before/after the workload, never inside a measured worker call.
        No device query, synchronization, affinity or thread-setting call is used.
    """
    before = time.monotonic_ns()
    threads = []
    for path in sorted(Path("/proc/self/task").iterdir()):
        tid = int(path.name)
        try:
            affinity = sorted(os.sched_getaffinity(tid))
            error = None
        except OSError as exc:
            affinity = None
            error = repr(exc)
        threads.append(
            dict(
                tid=tid,
                name=read_optional(path / "comm"),
                affinity=affinity,
                error=error,
            )
        )
    names = (
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_WAIT_POLICY",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "VLLM_BATCH_INVARIANT",
    )
    affinity = sorted(os.sched_getaffinity(0))
    cpus = {}
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        cpus[str(cpu)] = {
            name: read_optional(base / "cpufreq" / name)
            for name in (
                "scaling_cur_freq",
                "scaling_governor",
                "scaling_min_freq",
                "scaling_max_freq",
            )
        }
    return dict(
        pid=os.getpid(),
        native_tid=threading.get_native_id(),
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        process_affinity=sorted(os.sched_getaffinity(os.getpid())),
        calling_thread_affinity=affinity,
        threads=threads,
        environment={name: os.environ.get(name) for name in names},
        proc_status=read_optional(Path("/proc/self/status")),
        proc_numa_maps=read_optional(Path("/proc/self/numa_maps")),
        cpu_frequency=cpus,
        begin_ns=before,
        end_ns=time.monotonic_ns(),
    )


def normalization_environment(worker: Any) -> dict[str, Any]:
    """Read instantiated CustomOp dispatch and configured IR priority, without calls.

    Args:
        worker: Initialized native worker providing get_model and vllm_config.
    Returns:
        Actual stored callable identities and resolved compilation/kernel settings.
        IR priority is labeled configuration; heavy traces provide executed kernels.
    """
    # Third Party
    from vllm import envs, ir
    from vllm.model_executor.layers.layernorm import RMSNorm

    config = worker.vllm_config
    modules = [
        dict(
            name=name,
            **module.get_dispatch_info(),
            dtype=str(module.weight.dtype),
            device=str(module.weight.device),
            epsilon=module.variance_epsilon,
        )
        for name, module in worker.get_model().named_modules()
        if isinstance(module, RMSNorm)
    ]
    return dict(
        instantiated_rmsnorm=modules,
        compilation_mode=str(config.compilation_config.mode),
        custom_ops=list(config.compilation_config.custom_ops),
        enabled_custom_ops=sorted(config.compilation_config.enabled_custom_ops),
        disabled_custom_ops=sorted(config.compilation_config.disabled_custom_ops),
        VLLM_BATCH_INVARIANT_environment=os.environ.get("VLLM_BATCH_INVARIANT"),
        VLLM_BATCH_INVARIANT_resolved=envs.VLLM_BATCH_INVARIANT,
        ir_priority={
            "rms_norm": ir.ops.rms_norm.get_priority(),
            "fused_add_rms_norm": ir.ops.fused_add_rms_norm.get_priority(),
        },
        ir_note="Configured priority is not an executed-provider observation. "
        "Stored forward_cuda with batch-invariant enabled routes through "
        "rms_norm_batch_invariant; heavy traces must verify kernel names.",
    )


class EnvironmentExtension:
    """Metadata-only worker RPC extension; no execute/sample/scheduler override."""

    def capture_host_environment(self) -> dict[str, Any]:
        """Return this worker's environment outside the measurement window."""
        observation = None
        if os.environ.get("L2_HYBRID_MODEL_DISPATCH") == "1":
            # First Party
            from lmcache.integration.vllm.model_dispatch_observer import (
                ModelDispatchObserver,
            )

            if not hasattr(self, "model_dispatch_observer"):
                worker: Any = self
                self.model_dispatch_observer = ModelDispatchObserver(worker.get_model())
            observation = self.model_dispatch_observer.status()
        return dict(
            **process_environment(),
            normalization=normalization_environment(self),
            model_dispatch_observation=observation,
        )
