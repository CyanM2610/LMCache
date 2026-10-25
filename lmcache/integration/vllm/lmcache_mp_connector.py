# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections import OrderedDict
from collections.abc import Callable, Iterable
from itertools import islice
from typing import TYPE_CHECKING, Any
import math
import sys

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

try:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
except ImportError:
    # Older vLLM builds do not expose HMA. They cannot route per-group
    # request-finished calls, but keeping the class importable preserves
    # legacy single-group behavior.
    class SupportsHMA:  # type: ignore[no-redef]
        pass


# Third Party
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_wait_trace import emit as wait_emit
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
import torch
import zmq

# First Party
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.hybrid_promotion import PromotionScheduler
from lmcache.integration.vllm.kv_cache_group_edits import (
    apply_kv_cache_group_edits,
    validate_kv_cache_groups,
)
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    get_tokens_per_block,
)
from lmcache.integration.vllm.lazy_offload_pending_store import (
    LazyOffloadPendingStore,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.utils import (
    extract_request_configs_from_request,
    mla_only,
    vllm_layout_hints,
)
from lmcache.utils import init_logger as lmcache_init_logger

try:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
    )

    try:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (  # type: ignore[attr-defined]
            RequestAllocationRecord,
        )
    except ImportError:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            BlockAllocationRecord as RequestAllocationRecord,
        )
except ImportError:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (  # type: ignore[assignment,no-redef]
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
    )

    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        BlockAllocationRecord as RequestAllocationRecord,
    )

if TYPE_CHECKING:
    # Third Party
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.block_pool import ExternalPrefix
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = lmcache_init_logger(__name__)

_DCP_LAYOUT_NAMESPACE = "##lmcache-dcp-layout-v1-"
_MAX_LCM_EXPANSION_FACTOR = 4


# Helper functions
def _has_preemption_reqs(scheduler_output: SchedulerOutput) -> bool:
    """Return whether the scheduler output contains preemption-related requests.

    Checks for the presence of resumed or preempted requests in the
    scheduler output.

    A preemption is detected if:
    - ``scheduled_cached_reqs.resumed_req_ids``: Requests resumed from
      preemption this step.
    - ``scheduler_output.preempted_req_ids``: Requests preempted this step.

    Args:
        scheduler_output: The vLLM scheduler output for this step.

    Returns:
        True if preemption-related requests exist, False otherwise.
    """
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)

    # Primary signal: requests resumed from preemption this step.
    resumed_ids = getattr(cached_reqs, "resumed_req_ids", None)
    if resumed_ids:
        logger.warning("<preempted> by resumed requests: %s", resumed_ids)
        return True

    # Primary signal: requests preempted this step.
    preempted_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_ids:
        logger.warning("<preempted> by preempted requests: %s", preempted_ids)
        return True

    return False


def _iter_kv_cache_specs(kv_cache_config: "KVCacheConfig | None") -> Iterable[Any]:
    """Yield leaf KV cache specs without depending on a vLLM helper API."""
    if kv_cache_config is None:
        return
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        per_layer_specs = getattr(group_spec, "kv_cache_specs", None)
        if isinstance(per_layer_specs, dict):
            yield from per_layer_specs.values()
        else:
            yield group_spec


def get_group_tokens_per_block(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> list[int]:
    """Return each KV cache group's effective block span in tokens.

    Attention pages are local DCP shards and therefore cover
    ``spec.block_size * dcp_size`` global tokens. Recurrent-state pages are
    replicated and retain their physical ``spec.block_size`` span. When vLLM
    does not provide group metadata, preserve the legacy single-group rule.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The effective token span of each KV cache group.
    """
    dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    return [
        get_tokens_per_block(group.kv_cache_spec, dcp_size) for group in groups
    ] or [vllm_config.cache_config.block_size * dcp_size]


def get_vllm_scheduler_block_size(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> int:
    """Return vLLM's scheduler block size for the resolved cache groups.

    The scheduler boundary must align with every cache group, so vLLM uses the
    least common multiple of their effective block spans.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The scheduler block size in tokens.
    """
    group_spans = get_group_tokens_per_block(vllm_config, kv_cache_config)
    scheduler_block_size = math.lcm(*group_spans)
    largest_group_span = max(group_spans)
    if scheduler_block_size > largest_group_span * _MAX_LCM_EXPANSION_FACTOR:
        logger.warning(
            "LMCache resolved a scheduler block of %d tokens from group spans "
            "%s (%0.1fx the largest span). Near-coprime group geometry can "
            "make cache chunks too coarse to be useful.",
            scheduler_block_size,
            group_spans,
            scheduler_block_size / largest_group_span,
        )
    return scheduler_block_size


def get_dcp_decorated_model_name(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> str:
    """Decorate the model name with its non-trivial DCP interleave layout.

    Embedding the DCP interleave size in the LMCache model name prevents a
    changed interleave from hitting objects with an incompatible byte layout.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The decorated cache model name, or the original model name when DCP or
        interleaving is inactive.
    """
    model_name = vllm_config.model_config.model
    parallel_config = vllm_config.parallel_config
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    interleave = original = getattr(parallel_config, "cp_kv_cache_interleave_size", 1)
    # adjust_dcp_kv_cache_interleave_size runs per worker, so the scheduler's
    # config keeps the pre-adjustment value. Delegate to vLLM (idempotent, and
    # gated differently across versions), then restore what it owns.
    adjust = getattr(vllm_config, "adjust_dcp_kv_cache_interleave_size", None)
    if adjust is not None and kv_cache_config is not None:
        try:
            adjust(kv_cache_config)
            interleave = getattr(
                parallel_config, "cp_kv_cache_interleave_size", original
            )
        finally:
            parallel_config.cp_kv_cache_interleave_size = original
    if dcp_size <= 1 or interleave == 1:
        return model_name
    return f"{model_name}{_DCP_LAYOUT_NAMESPACE}d{dcp_size}-interleave{interleave}"


def get_resolved_attention_block_sizes(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> set[int]:
    """Return physical block sizes for resolved attention cache groups.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The resolved attention block sizes. Falls back to vLLM's cache block
        size when group metadata is unavailable.
    """
    block_sizes = {
        spec.block_size
        for spec in _iter_kv_cache_specs(kv_cache_config)
        if any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)
    }
    if block_sizes:
        return block_sizes
    return {vllm_config.cache_config.block_size}


def validate_mamba_step_alignment(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> None:
    """Reject scheduler configs whose steps cannot advance a whole Mamba block.

    In ``mamba_cache_mode="align"`` vLLM snapshots the recurrent state only at
    the end of each scheduler step, on the last block the step advanced. A step
    advancing more than one block fills the skipped block-table positions with
    the null block (``MambaManager.allocate_new_blocks``); LMCache handles those
    safely -- ``store`` never commits an all-null-block chunk and ``retrieve``
    loads only each object group's sliding-window suffix -- so
    ``max_num_batched_tokens`` may exceed ``2 * block_size`` (with
    ``--separate-object-groups``). Only the lower bound remains: a step must
    advance at least one full block, or vLLM's block-aligned splitting
    (``Scheduler._mamba_block_aligned_split``) yields empty chunks and prefill
    cannot progress.

    Args:
        vllm_config: The vLLM config; only Mamba-hybrid models in ``align``
            cache mode are constrained, others pass.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        None.

    Raises:
        ValueError: If Mamba groups have inconsistent block sizes or
            ``max_num_batched_tokens < block_size``.
    """
    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    mamba_block_sizes = {
        spec.block_size
        for spec in _iter_kv_cache_specs(kv_cache_config)
        if any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)
    }
    if len(mamba_block_sizes) > 1:
        raise ValueError(
            "All Mamba KV cache groups must use one physical block size; got "
            f"{sorted(mamba_block_sizes)}."
        )
    block_size = (
        next(iter(mamba_block_sizes))
        if mamba_block_sizes
        else vllm_config.cache_config.block_size
    )
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
    if max_batched < block_size:
        raise ValueError(
            f"Mamba-hybrid models with LMCache require "
            f"max_num_batched_tokens >= block_size so every prefill step "
            f"advances at least one full block; got "
            f"max_num_batched_tokens={max_batched}, block_size={block_size}. "
            f"Set --max-num-batched-tokens to at least {block_size}."
        )


def validate_dcp_support(
    vllm_config: VllmConfig,
    n_servers: int,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> None:
    """Reject decode-context-parallel topologies this connector cannot serve.

    These would silently store wrong or incomplete KV, so fail at startup
    instead. ``dcp > tp`` is not re-checked; vLLM's ``ParallelConfig``
    already rejects it.

    Args:
        vllm_config: The active vLLM configuration.
        n_servers: The number of LMCache servers backing the deployment.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        None.

    Raises:
        ValueError: On an unsupported DCP topology (``dcp_size == 1``
            always passes).
    """
    pc = vllm_config.parallel_config
    dcp_size = getattr(pc, "decode_context_parallel_size", 1)
    if dcp_size <= 1:
        return

    pcp_size = getattr(pc, "prefill_context_parallel_size", 1)
    if pcp_size > 1:
        raise ValueError(
            "LMCacheMPConnector does not support prefill-context parallelism "
            f"together with DCP (got pcp={pcp_size}, dcp={dcp_size})."
        )

    interleave = getattr(pc, "cp_kv_cache_interleave_size", 1)
    attention_block_sizes = get_resolved_attention_block_sizes(
        vllm_config, kv_cache_config
    )
    incompatible_block_sizes = sorted(
        block_size
        for block_size in attention_block_sizes
        if interleave > block_size or block_size % interleave != 0
    )
    if incompatible_block_sizes:
        raise ValueError(
            f"cp_kv_cache_interleave_size ({interleave}) must be no greater "
            "than and evenly divide every resolved attention block size; "
            f"incompatible sizes: {incompatible_block_sizes}."
        )

    ranks_per_server = pc.world_size // n_servers
    if ranks_per_server < dcp_size:
        raise ValueError(
            f"Each LMCache server needs at least decode_context_parallel_size "
            f"({dcp_size}) ranks to hold a complete set of shards, but "
            f"{n_servers} server(s) leave only {ranks_per_server} rank(s) "
            "each. Use fewer servers or a smaller DCP size."
        )


def build_parallel_strategy_from_vllm_config(
    vllm_config: "VllmConfig",
    n_servers: int,
) -> ParallelStrategy:
    """Build a ParallelStrategy from a vLLM config.

    Centralises the (vllm_config -> KV parallel geometry) mapping.

    Args:
        vllm_config: The vLLM configuration object.
        n_servers: Number of LMCache servers backing this deployment.

    Returns:
        The constructed ParallelStrategy.
    """
    pc = vllm_config.parallel_config
    return ParallelStrategy(
        mla_only=mla_only(vllm_config.model_config),
        vllm_world_size=pc.world_size,
        vllm_worker_id=pc.rank,
        tp_size=pc.tensor_parallel_size,
        pp_size=pc.pipeline_parallel_size,
        n_servers=n_servers,
        dcp_size=getattr(pc, "decode_context_parallel_size", 1),
    )


def _ensure_transport_scheme(server_url: str) -> str:
    """Ensure a multiprocess server URL carries a transport scheme.

    Preserve explicit schemes so the request client factory can select the
    transport. Bare endpoints retain the legacy ZMQ TCP behavior.

    Args:
        server_url: A server URL, with or without a ``<scheme>://`` prefix.

    Returns:
        The URL with a transport scheme, defaulting to ``tcp://``.
    """
    if "://" in server_url:
        return server_url
    return f"tcp://{server_url}"


class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA):
    """
    The connector for LMCache multi-process mode.

    Extra configs (kv_transfer_config.extra_config):

    Multi-server deployment:
    - lmcache.mp.server_urls: server URL list or comma-separated string,
      e.g. "tcp://host1:6667,tcp://host2:6667".

    Single-server deployment:
    - lmcache.mp.host: the host of the LMCache server.
    - lmcache.mp.port: the port of the LMCache server.

    - lmcache.mp.mq_timeout: timeout (seconds) for message queue requests.
    - lmcache.mp.heartbeat_interval: interval (seconds) between server
      heartbeat pings.
    - lmcache.mp.eager_prefetch: submit the LMCache lookup when a request
      enters vLLM's waiting queue. Disabled by default.
    - lmcache.mp.promotion_control: enable independent V2 worker controls for
      terminal TP1, single dense-group prefixes. Disabled by default.
    """

    # No controller exists until an explicitly enabled scheduler binds its pool.
    _promotion: PromotionScheduler | None = None
    _promotion_enabled: bool = False
    decoupled_control_enabled: bool = False
    prepare_lookup_operations: bool = False
    prepare_load_operations: bool = True  # Absent key preserves enabled v4 semantics.
    decoupled_store_enabled: bool = False
    operation_step_limit: int = 32
    store_operation_decider: Callable[[dict[str, Any]], str] | None = None

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        # Older supported vLLM releases allow connectors to omit this value,
        # while current vLLM's type declaration requires it.
        super().__init__(vllm_config, role, kv_cache_config)  # type: ignore[arg-type]

        # Fail fast, before the server handshake below.
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        validate_mamba_step_alignment(vllm_config, kv_cache_config)
        validate_kv_cache_groups(kv_cache_config)

        group_tokens_per_block = get_group_tokens_per_block(
            vllm_config, kv_cache_config
        )
        scheduler_block_size = get_vllm_scheduler_block_size(
            vllm_config, kv_cache_config
        )
        cache_model_name = get_dcp_decorated_model_name(vllm_config, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None

        self._require_terminal_completion = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.require_terminal_completion", False
            )
        )
        self._promotion_enabled = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.promotion_control", False
            )
        )

        self._eager_prefetch: bool = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.eager_prefetch", False
            )
        )
        operation_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        for key in (
            "decoupled_control", "prepare_lookup_operations", "prepare_load_operations", "decoupled_store", "action_feedback", "global_plan", "plan_new_lookup"
        ):
            if type(operation_config.get(f"lmcache.mp.{key}", False)) is not bool:
                raise ValueError(f"lmcache.mp.{key} must be a boolean")
        self.decoupled_control_enabled = operation_config.get(
            "lmcache.mp.decoupled_control", False
        ) is True
        # Only the EXTRA restoration entry is gated. Native WAITING and
        # completion adoption are unaffected. Omission preserves v4.
        self.prepare_load_operations = operation_config.get(
            "lmcache.mp.prepare_load_operations", self.decoupled_control_enabled
        ) is True
        if self.prepare_load_operations and not self.decoupled_control_enabled:
            raise ValueError("extra LOAD preparation requires decoupled_control")
        self.prepare_lookup_operations = operation_config.get(
            "lmcache.mp.prepare_lookup_operations", False
        ) is True
        self.plan_lookup_enabled = (operation_config.get("lmcache.mp.global_plan", False)
                                    and operation_config.get("lmcache.mp.plan_new_lookup", False))
        self.decoupled_store_enabled = operation_config.get(
            "lmcache.mp.decoupled_store", False
        ) is True
        self.operation_step_limit = operation_config.get(
            "lmcache.mp.operation_step_limit", 32
        )
        if type(self.operation_step_limit) is not int or self.operation_step_limit <= 0:
            raise ValueError("operation_step_limit must be a positive integer")
        if self.prepare_lookup_operations and not self.decoupled_control_enabled:
            raise ValueError("LOOKUP preparation requires decoupled_control")
        self._store_operation_sequence = 0
        self._store_operation_views: dict[str, dict[str, Any]] = {}
        self._store_submissions: list[str] = []
        self._retained_store_candidates: OrderedDict[str, None] = OrderedDict()
        self._control_step_id: int | None = None

        # Multi-server: prefer lmcache.mp.server_urls (list or comma-separated
        # string) over the single-server lmcache.mp.host / lmcache.mp.port.
        server_urls_cfg = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.server_urls", None
        )
        if server_urls_cfg:
            if isinstance(server_urls_cfg, list):
                server_urls = [u.strip() for u in server_urls_cfg if u.strip()]
            else:
                server_urls = [
                    u.strip() for u in server_urls_cfg.split(",") if u.strip()
                ]
        else:
            # Legacy single-server fallback.
            server_host = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.host", "tcp://localhost"
            )
            server_port = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.port", 5555
            )
            server_urls = [f"{server_host}:{server_port}"]

        # Preserve explicit transport schemes and default bare host:port
        # endpoints to the legacy ZMQ TCP transport.
        server_urls = [_ensure_transport_scheme(u) for u in server_urls]

        # The server count is derived from lmcache.mp.server_urls.
        n_servers = len(server_urls)

        validate_dcp_support(vllm_config, n_servers, kv_cache_config)

        logger.info(
            "Resolved LMCache MP geometry: group_tokens_per_block=%s, "
            "scheduler_block_size=%d, cache_model_name=%s",
            group_tokens_per_block,
            scheduler_block_size,
            cache_model_name,
        )

        assert vllm_config.parallel_config.world_size % n_servers == 0, (
            f"world_size ({vllm_config.parallel_config.world_size}) must be "
            f"divisible by n_servers ({n_servers})"
        )

        # Multi-server + DP is not supported yet.
        dp_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
        if n_servers > 1 and dp_size > 1:
            raise ValueError(
                "LMCacheMPConnector multi-server mode (n_servers > 1) does not "
                f"support data parallelism yet; got dp_size={dp_size}. "
                "DP across multiple LMCache servers will be "
                "supported in a follow-up PR."
            )

        # Multi-server + MLA: only TP is supported (no PP).
        # PP splits layers across nodes, which would cause per-piece
        # reader counts to vary per (server, pp_stage) pair and break
        # the single-``tp_size`` LOOKUP / FREE_LOOKUP_LOCKS protocol.
        # Non-MLA mode is not affected by this restriction.
        if n_servers > 1:
            pp_size = vllm_config.parallel_config.pipeline_parallel_size
            if pp_size > 1:
                raise ValueError(
                    "LMCacheMPConnector multi-server mode only supports "
                    "tensor parallelism (TP), not pipeline parallelism (PP). "
                    f"Got pp_size={pp_size}."
                )

        zmq_context = zmq.Context.instance()
        parallel_strategy = build_parallel_strategy_from_vllm_config(
            vllm_config, n_servers
        )

        self.dispatcher = None

        dcp_size = parallel_strategy.dcp_size
        self._dcp_size = dcp_size

        # Lazy offload configuration: when enabled, store operations are
        # deferred until some threshold is reached, rather than submitted at every step
        self.lazy_offload = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.lazy_offload", False
        )
        if self.decoupled_store_enabled and (
            not self._require_terminal_completion
            or not vllm_config.use_v2_model_runner
            or vllm_config.parallel_config.world_size != 1
            or dp_size != 1 or n_servers != 1 or dcp_size != 1
            or self.lazy_offload or self.transfer_intermediate_tensors
            or kv_cache_config is None
            or len(kv_cache_config.kv_cache_groups) != 1
            or any(type(spec).__name__ != "FullAttentionSpec"
                   for spec in _iter_kv_cache_specs(kv_cache_config))
        ):
            raise ValueError(
                "decoupled STORE requires V2, terminal TP1, one dense group "
                "and non-lazy offload without intermediate tensors"
            )
        if self._promotion_enabled and (
            not self._require_terminal_completion
            or not vllm_config.use_v2_model_runner
            or vllm_config.parallel_config.world_size != 1
            or dp_size != 1
            or n_servers != 1
            or dcp_size != 1
            or self.lazy_offload
            or self.transfer_intermediate_tensors
            or not vllm_config.cache_config.enable_prefix_caching
            or kv_cache_config is None
            or len(kv_cache_config.kv_cache_groups) != 1
            or any(
                type(spec).__name__ != "FullAttentionSpec"
                for spec in _iter_kv_cache_specs(kv_cache_config)
            )
        ):
            raise ValueError(
                "promotion_control requires V2 runner, terminal TP1, "
                "a single dense group, "
                "prefix caching and non-lazy offload"
            )

        if self.role == KVConnectorRole.SCHEDULER:
            # Banner from the scheduler role only, so tensor-parallel
            # deployments print it once rather than once per worker.
            print_banner_once(sys.stderr)
            self.scheduler_adapter = LMCacheMPSchedulerAdapter(
                server_urls=server_urls,
                context=zmq_context,
                model_name=cache_model_name,
                vllm_block_size=scheduler_block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}

            # GPU block pool reference
            self._gpu_block_pool: "BlockPool | None" = None

            # Initialize pending store for lazy offload mode
            if self.lazy_offload:
                self._pending_store = LazyOffloadPendingStore(
                    vllm_config.kv_transfer_config.kv_connector_extra_config
                )
        elif self.role == KVConnectorRole.WORKER:
            # Node routing: a worker connects only to its local LMCache server.
            # Global ranks are assigned to nodes in contiguous blocks:
            #   node 0 → ranks [0, ranks_per_node),
            #   node 1 → [ranks_per_node, 2 * ranks_per_node), ...
            ranks_per_node = parallel_strategy.vllm_world_size // n_servers
            local_server_url = server_urls[
                parallel_strategy.vllm_worker_id // ranks_per_node
            ]
            self.worker_adapter = LMCacheMPWorkerAdapter(
                server_url=local_server_url,
                context=zmq_context,
                model_name=cache_model_name,
                vllm_block_size=scheduler_block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
                kv_load_failure_policy=getattr(
                    vllm_config.kv_transfer_config,
                    "kv_load_failure_policy",
                    "recompute",
                ),
            )
            if self.transfer_intermediate_tensors:
                # First Party
                from lmcache.integration.vllm.experimental import (
                    FeatureContext,
                    init_dispatcher,
                )
                from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY

                ctx = FeatureContext(
                    worker_adapter=self.worker_adapter,
                )
                requested = (
                    {TRANSFER_QUERY} if self.transfer_intermediate_tensors else set()
                )
                self.dispatcher = init_dispatcher(ctx, requested)
                self.worker_adapter.dispatcher = self.dispatcher
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        # Tokens covered by one paged chunk (one block ID) of each engine
        # group, from the group's KV cache spec. Hybrid models can mix
        # different values (e.g. gemma-4: sliding-window groups 32,
        # full-attention groups 16; DeepSeek V4: 256/64/8/4). Falls back to
        # the engine's base block size when no group metadata is available
        # (single non-hybrid group).
        self._group_tokens_per_block = group_tokens_per_block
        for engine_group_idx, tokens_per_block in enumerate(
            self._group_tokens_per_block
        ):
            if tokens_per_block <= 0:
                raise ValueError(
                    f"group {engine_group_idx} tokens_per_block "
                    f"{tokens_per_block} must be positive"
                )
        # Smallest token count aligned to every group's paged-chunk
        # boundary; used to round down vLLM APC hit counts.
        self._hit_alignment_tokens = math.lcm(*self._group_tokens_per_block)
        if self.role == KVConnectorRole.SCHEDULER:
            # Chunk boundaries must land on every group's paged-chunk
            # boundary so per-group block-id slicing stays aligned.
            lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
            for engine_group_idx, tokens_per_block in enumerate(
                self._group_tokens_per_block
            ):
                if lmcache_tokens_per_chunk % tokens_per_block != 0:
                    raise ValueError(
                        f"LMCache chunk size {lmcache_tokens_per_chunk} must be "
                        f"a multiple of group {engine_group_idx} "
                        f"tokens_per_block {tokens_per_block}"
                    )

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    @property
    def transfer_intermediate_tensors(self) -> bool:
        cfg = self._vllm_config.kv_transfer_config
        if cfg is None:
            return False
        vllm_transfer = cfg.get_from_extra_config(
            "lmcache.mp.transfer_intermediate_tensors", False
        )
        return vllm_transfer

    # ==============================
    # Worker-side methods
    # ==============================

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        logger.info("Registering kv caches!")
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        # Must precede both group-info creation and transfer registration so
        # they see the same edited views.
        layout_hints = vllm_layout_hints(self._vllm_config)
        kv_caches = apply_kv_cache_group_edits(
            kv_cache_config, kv_caches, layout_hints=layout_hints
        )
        engine_group_infos = create_engine_group_infos_from_vllm(
            kv_cache_config,
            kv_caches,
            layout_hints=layout_hints,
            dcp_size=self._dcp_size,
        )
        self.worker_adapter.register_kv_caches(
            kv_caches,
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
        )
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "register",
                kv_caches=kv_caches,
                kv_cache_config=kv_cache_config,
                vllm_config=self._vllm_config,
            )
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []
        request_configs_list = []

        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)
            request_configs_list.append(meta.request_configs)

        if len(request_ids) == 0:
            return

        event = self.worker_adapter.create_recorded_event()

        self.worker_adapter.batched_submit_retrieve_requests(
            request_ids,
            ops,
            event,
            cache_salts=cache_salts,
            request_configs_list=request_configs_list,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "save_kv_layer",
                layer_name=layer_name,
                metadata=self._get_connector_metadata(),
                attn_metadata=attn_metadata,
                **kwargs,
            )
        return

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []
        request_configs_list = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)
            request_configs_list.append(meta.request_configs)

        if len(request_ids) == 0:
            if self.dispatcher is not None:
                dispatch(self.dispatcher, "wait_for_save", event=None)
            return

        event = self.worker_adapter.create_recorded_event()

        if self.decoupled_store_enabled:
            for meta in metadata.requests:
                if meta.direction != "STORE":
                    continue
                self.worker_adapter.retain_store_request(
                    meta.operation_id, meta.request_id, meta.op, event,
                    cache_salt=meta.cache_salt,
                    request_configs=meta.request_configs,
                )
                if meta.operation_id not in metadata.defer_store_ids:
                    self.worker_adapter.submit_retained_stores((meta.operation_id,))
            return

        self.worker_adapter.batched_submit_store_requests(
            request_ids,
            ops,
            event,
            cache_salts=cache_salts,
            request_configs_list=request_configs_list,
        )
        if self.dispatcher is not None:
            dispatch(self.dispatcher, "wait_for_save", event=event)

    # TODO: How does lmcache driven path handle preemption?
    # NOTE1: handle_preemptions is called by vllm each step regardless
    #        preemption really happens or not.
    # NOTE2: preemption hint is managed by KVConnectorRole.SCHEDULER,
    #        that's why here we have to judge preemption by
    #        need_flush_before_forward flag which is set by SCHEDULER.
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Flush async engine-driven stores only when scheduler metadata requests it.

        Args:
            kv_connector_metadata: Connector metadata produced by the scheduler;
                only acts when it is a :class:`LMCacheMPConnectorMetadata` with
                ``need_flush_before_forward=True``.
        """
        worker_adapter = getattr(self, "worker_adapter", None)
        if self.role != KVConnectorRole.WORKER or worker_adapter is None:
            return
        need_flush_before_forward = (
            isinstance(kv_connector_metadata, LMCacheMPConnectorMetadata)
            and kv_connector_metadata.need_flush_before_forward
        )
        if (
            isinstance(kv_connector_metadata, LMCacheMPConnectorMetadata)
            and worker_adapter.require_terminal_completion
        ):
            worker_adapter.wait_for_kv_access(
                kv_connector_metadata.kv_read_blocks,
                kv_connector_metadata.kv_write_blocks,
            )
        worker_adapter.handle_preemptions(need_flush_before_forward)

    def prepare_kv_cache_writes(self, scheduler_output: SchedulerOutput) -> None:
        """Protect retained STORE sources before native zeroing or CoW writes.

        Args:
            scheduler_output: Native block-zero and block-copy operations.
        Notes:
            Only the supported single dense engine group is enabled. This hook
            precedes the writes and does not move any producer event.
        """
        if not self.decoupled_store_enabled:
            return
        copies = scheduler_output.kv_cache_block_copies or ()
        reads = [pair[0] for pair in copies]
        writes = list(scheduler_output.new_block_ids_to_zero or ())
        writes.extend(pair[1] for pair in copies)
        if reads or writes:
            self.worker_adapter.wait_for_kv_access({0: reads}, {0: writes})

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        if self.lazy_offload:
            val = self.worker_adapter.get_finished_with_lazy_offload()
        else:
            val = self.worker_adapter.get_finished(finished_req_ids)
        # logger.error("Finished req ids: %s, %s", val[0], val[1])
        return val

    def build_connector_worker_meta(self):
        completed_store_requests = (
            self.worker_adapter.get_completed_store_requests()
            if self.lazy_offload
            else {}
        )
        receipts = (
            self.worker_adapter.get_promotion_receipts()
            if self._promotion_enabled
            else ()
        )
        store_receipts = (
            self.worker_adapter.get_store_operation_receipts()
            if self.decoupled_store_enabled else ()
        )
        if completed_store_requests or receipts or store_receipts:
            return LMCacheMPWorkerMetadata(
                completed_store_requests=completed_store_requests or {},
                promotion_receipts=receipts,
                store_receipts=store_receipts,
            )
        else:
            return None

    def submit_control(self) -> None:
        """Submit independent Promotion LOADs in normal or empty worker steps.

        Completion and publication remain separated by the native output FIFO.
        Raises the underlying adapter error without releasing HBM ownership.
        """
        if not self._promotion_enabled and not self.decoupled_store_enabled:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        if self.decoupled_store_enabled:
            for item in metadata.requests:
                if (
                    item.direction == "STORE"
                    and not self.worker_adapter.has_store_commitment(item.operation_id)
                ):
                    raise RuntimeError(
                        "new STORE requires its original post-forward producer event"
                    )
            self.worker_adapter.submit_retained_stores(metadata.submit_store_ids)
        if not self._promotion_enabled:
            return
        if metadata.promotions:
            event = self.worker_adapter.create_recorded_event()
            for command in metadata.promotions:
                self.worker_adapter.submit_promotion(command, event)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        return self.worker_adapter.get_block_ids_with_load_errors()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        if self._promotion is not None and self._promotion.has_pending_work:
            raise RuntimeError(
                "promotion controls must drain before scheduler shutdown"
            )
        if hasattr(self, "worker_adapter"):
            self.worker_adapter.shutdown()
        if hasattr(self, "scheduler_adapter"):
            self.scheduler_adapter.shutdown()
        return None

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        if self.role == KVConnectorRole.SCHEDULER:
            logger.info("Bind GPU block pool in LMCacheMPConnector scheduler")
            self._gpu_block_pool = gpu_block_pool
            if self._promotion_enabled:
                if self._group_tokens_per_block != [gpu_block_pool.hash_block_size]:
                    raise ValueError(
                        "promotion requires native single-group hash geometry"
                    )
                self._promotion = PromotionScheduler(
                    gpu_block_pool, self.scheduler_adapter
                )
            if self.lazy_offload:
                self._pending_store.bind_gpu_block_pool(gpu_block_pool)

    def snapshot_promotion_prefix(
        self, request: "Request", num_tokens: int
    ) -> "ExternalPrefix":
        """Snapshot known text/hashes; the native frontend Request is not retained.

        Args:
            request: Already observed native request, not a synthetic control.
            num_tokens: Whole-block known prefix length.
        Returns:
            Immutable prefix descriptor for later policy selection.
        Raises:
            ValueError: Disabled control, unbound pool, or unsupported text input.
        """
        if self._promotion is None or self._gpu_block_pool is None:
            raise ValueError("promotion control is not enabled and bound")
        if extract_request_configs_from_request(request):
            raise ValueError(
                "promotion does not yet support per-request LMCache configs"
            )
        return self._gpu_block_pool.snapshot_external_prefix(request, num_tokens)

    def queue_promotion(
        self, prefix: "ExternalPrefix", max_new_blocks: int, *, min_free_blocks: int = 0
    ) -> str | None:
        """Queue one observed prefix; runtime owns capacity and source leases.

        Returns a control ID or None for duplicate/queue rejection. Raises
        ValueError for disabled control or invalid geometry/budget.
        """
        if self._promotion is None:
            raise ValueError("promotion control is not enabled and bound")
        return self._promotion.enqueue(
            prefix, max_new_blocks, min_free_blocks=min_free_blocks
        )

    def cancel_promotion(self, control_id: str) -> None:
        """Cancel a control while retaining captured copies until terminal output."""
        if self._promotion is None:
            raise ValueError("promotion control is not enabled and bound")
        self._promotion.cancel(control_id)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._get_num_new_matched_tokens(
            request, num_computed_tokens, allow_lookup_submit=True,
            commit_match=True,
            # Typed actions may defer after matching even when control is OFF.
            # A repeated match observes the same absolute stored prefix; only
            # actual STORE commitments advance the incremental suffix watermark.
            idempotent_prefix_accounting=(self.decoupled_control_enabled or
                getattr(self, "_operation_feedback_observer", None) is not None),
        )

    def begin_prefill_control_step(self, step_id: int) -> None:
        """Observe existing LOOKUP operations once at a scheduler-step boundary.

        Args:
            step_id: Native scheduler step identity.
        Raises:
            ValueError: The same step tries to advance control twice.
        """
        if not self.decoupled_control_enabled:
            return
        if self._control_step_id == step_id:
            raise ValueError("prefill control already advanced in this step")
        self._control_step_id = step_id
        self.scheduler_adapter.progress_pending_lookups()

    def lookup_operation_state(self, request_id: str) -> dict[str, Any]:
        """Read native lookup state without querying, matching or taking resources."""
        state = self.scheduler_adapter.lookup_operation_state(request_id)
        tracker = self.request_trackers.get(request_id)
        if tracker is not None and tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE:
            state["native_fallback_reason"] = "native_tracker_bypass"
        elif tracker is not None and tracker.recompute_lower_prefix_tokens is not None:
            # Query records are consumed at initial adoption. Native re-entry
            # of the committed local path still needs no new query; this says
            # nothing about physical readiness or successful HBM allocation.
            state["native_fallback_reason"] = "committed_recompute_path"
        return state

    def begin_first_match_step(self, step_id: int) -> None:
        """Share a nonblocking observation epoch without submitting or polling."""
        self.scheduler_adapter.begin_lookup_observation_step(step_id)

    def lookup_observation_counters(self) -> dict[str, int]:
        return self.scheduler_adapter.lookup_observation_counters()

    def prepare_lookup_operation(self, request: "Request", *, planned=False) -> dict[str, Any]:
        """Submit one native LOOKUP only when separate preparation is enabled.

        Args:
            request: Existing arrived request; no future tokens are introduced.
        Returns:
            Existing protocol state. This does not perform HBM matching.
        Raises:
            ValueError: Independent new-query preparation is disabled.
        """
        authorized = self.plan_lookup_enabled if planned else self.prepare_lookup_operations
        if not self.decoupled_control_enabled or not authorized:
            raise ValueError("independent LOOKUP preparation is disabled")
        tracker = self._get_or_create_request_tracker(request)
        if (request.status != RequestStatus.PREEMPTED
                and tracker.state == LMCacheMPRequestState.PREFETCHING):
            self.scheduler_adapter.maybe_submit_lookup_request(
                request.request_id, token_ids=tracker.get_token_ids(),
                cache_salt=tracker.cache_salt, request_configs=tracker.request_configs,
            )
        return self.lookup_operation_state(request.request_id)

    def get_operation_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return bounded, non-exhaustive operation views without native references."""
        lookups = self.scheduler_adapter.lookup_operation_snapshot()
        stores = tuple(
            dict(view)
            for view in islice(
                self._store_operation_views.values(), self.operation_step_limit
            )
        )
        return tuple(lookups) + stores

    def get_global_operation_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Complete scalar read; no result polling or retained-STORE rotation."""
        lookups = self.scheduler_adapter.lookup_operation_snapshot(complete=True)
        positions = {identity: index for index, identity in enumerate(self._retained_store_candidates)}
        stores = tuple(dict(view, retained_store=identity in positions,
                            queue="retained_store" if identity in positions else "committed_store",
                            queue_position=positions.get(identity, index))
                       for index, (identity, view) in enumerate(self._store_operation_views.items()))
        return tuple(lookups) + stores

    def set_operation_decider(
        self, decider: Callable[[dict[str, Any]], str] | None
    ) -> None:
        """Bind a scheduler-owned STORE choice callback; None restores native submit.

        Args:
            decider: Receives a resource-free candidate view and returns submit/defer.
        """
        self.store_operation_decider = decider

    def set_operation_feedback_observer(self, observer) -> None:
        """Native-boundary observations only; no new transport or polling."""
        self._operation_feedback_observer = observer
        self._feedback_store_receipts = {}
        self._feedback_ended_stores = OrderedDict()
        self.scheduler_adapter._operation_feedback_observer = observer

    def _notify_operation_feedback(self, view, result, **fields):
        observer = getattr(self, "_operation_feedback_observer", None)
        if observer is not None:
            observer(dict(view), result, **fields)

    def get_retained_store_candidates(
        self, limit: int, preferred_ids: Iterable[str] = (),
        excluded_ids: Iterable[str] = (),
    ) -> tuple[dict[str, Any], ...]:
        """Visit committed but unsubmitted STOREs in bounded fair round-robin order.

        Args:
            limit: Maximum visits, independently capped by operation_step_limit.
            preferred_ids: Explicit submission choices visited before round-robin.
            excluded_ids: Plan-held operations; no visit budget or queue rotation.
        Returns:
            Immutable-valued snapshots, not native source block references.
        Raises:
            ValueError: The visit bound is not a positive integer.
        """
        if type(limit) is not int or limit <= 0:
            raise ValueError("STORE candidate limit must be a positive integer")
        count = min(
            limit, self.operation_step_limit, len(self._retained_store_candidates)
        )
        result = []
        visited = set()
        excluded = set(excluded_ids)
        for identity in preferred_ids:
            if len(result) >= count:
                break
            if (identity in self._retained_store_candidates
                    and identity not in visited and identity not in excluded):
                visited.add(identity)
                self._retained_store_candidates.move_to_end(identity)
                result.append(dict(self._store_operation_views[identity]))
        for operation_id in tuple(self._retained_store_candidates):
            if len(result) >= count:
                break
            if operation_id in visited or operation_id in excluded:
                continue
            self._retained_store_candidates.move_to_end(operation_id)
            visited.add(operation_id)
            result.append(dict(self._store_operation_views[operation_id]))
        return tuple(result)

    def decide_store_operation(self, candidate: dict[str, Any]) -> str:
        """Return baseline immediate submission; future policies may return defer.

        Args:
            candidate: Ephemeral current STORE range with no resource acquisition.
        Returns:
            ``submit`` by default. ``defer`` still commits and protects the range.
        """
        decider = getattr(self, "store_operation_decider", None)
        return "submit" if decider is None else decider(dict(candidate))

    def request_store_submission(self, operation_ids: Iterable[str]) -> None:
        """Queue explicit existing STORE commitments for the next worker step.

        Args:
            operation_ids: Known commitments retained with their producer events.
        Raises:
            ValueError: Disabled mode or an unknown operation identity.
        """
        if not self.decoupled_store_enabled:
            raise ValueError("decoupled STORE is disabled")
        for operation_id in operation_ids:
            if operation_id not in self._store_operation_views:
                raise ValueError("unknown retained STORE operation")
            if self._store_operation_views[operation_id]["native_phase"] == "submitted":
                continue
            if operation_id not in self._store_submissions:
                self._store_submissions.append(operation_id)
            self._retained_store_candidates.pop(operation_id, None)

    def observe_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Progress one native future without committing match accounting.

        The reader lease and future are native-owned. Preparation observes the
        result but has no HBM ownership and must not record a stored prefix or
        overwrite the hit boundaries used to construct STORE/RETRIEVE metadata.
        """
        return self._get_num_new_matched_tokens(
            request, num_computed_tokens, allow_lookup_submit=True,
            commit_match=False, idempotent_prefix_accounting=True,
        )

    def match_prefix_tokens(
        self, request: "Request", num_computed_tokens: int, *,
        allow_lookup_submit: bool, allow_lookup_progress: bool,
        commit_match: bool,
    ) -> tuple[int | None, bool]:
        """Reconcile a local hit with explicitly authorized query side effects.

        Preparation passes commit_match=False. Cached-only callers neither
        submit nor poll; a missing result stays unknown unless the native
        failure/bypass path already permits local recomputation. This does not
        adopt HBM blocks, reserve a target, or release source ownership.
        """
        if allow_lookup_submit and not allow_lookup_progress:
            raise ValueError("LOOKUP submission requires query progression")
        if not commit_match and request.status == RequestStatus.PREEMPTED:
            return 0, False
        if not allow_lookup_progress:
            state = self.lookup_operation_state(request.request_id)
            if not state["result_known"] and not state.get("native_fallback_reason"):
                return None, True
        return self._get_num_new_matched_tokens(
            request, num_computed_tokens,
            allow_lookup_submit=allow_lookup_submit,
            allow_lookup_progress=allow_lookup_progress,
            commit_match=commit_match, idempotent_prefix_accounting=True,
        )

    def get_num_new_matched_tokens_from_existing_lookup(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Reconcile fresh HBM against the same native future, never submit.

        Admission commits current match boundaries. Repeated allocation
        attempts must not count the same absolute lower prefix twice. STORE
        increments remain additive in LMCacheMPRequestTracker. PREEMPTED and
        failed-load bypass cases use native rules.
        """
        return self._get_num_new_matched_tokens(
            request, num_computed_tokens, allow_lookup_submit=False,
            commit_match=True, idempotent_prefix_accounting=True,
        )

    def _get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
        *,
        allow_lookup_submit: bool,
        commit_match: bool,
        idempotent_prefix_accounting: bool,
        allow_lookup_progress: bool = True,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        tracker = self._get_or_create_request_tracker(request)
        # TODO: support loading KV for preempted requests in the future
        if request.status == RequestStatus.PREEMPTED:
            if commit_match and tracker.recompute_lower_prefix_tokens is not None:
                # Native re-admission refreshes the actual local prefix. The
                # original lower saved scalar survives; no new query is needed.
                tracker.local_compute_start_tokens = num_computed_tokens
                tracker.num_vllm_hit_tokens = (
                    num_computed_tokens // self._hit_alignment_tokens
                    * self._hit_alignment_tokens
                )
            return 0, False

        if (
            tracker.recompute_lower_prefix_tokens is not None
            and tracker.state == LMCacheMPRequestState.READY
        ):
            # A committed local path is not a failed remote load, even when its
            # first compute allocation has not yet advanced vLLM's counters.
            # Never recreate a query after its original result was consumed.
            return 0, False

        # A failed asynchronous load is bypassed until vLLM admits the request
        # for local computation via update_state_after_alloc().  The scheduler
        # may poll this method repeatedly before that admission; do not submit
        # another lookup or re-enter WAITING_FOR_REMOTE_KVS in the meantime.
        if tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE:
            return 0, False

        # A completed async load normally leaves num_computed_tokens > 0, so
        # the scheduler does not call this method again.  If vLLM reset the
        # request to zero after the worker reported invalid blocks, however,
        # the existing tracker is still READY and describes the failed load.
        # Reusing it would report another external hit without transitioning
        # back through WAITING_FOR_LOAD, leaving the request stuck forever in
        # WAITING_FOR_REMOTE_KVS.  Fail closed for this request instead: drop
        # the stale local lookup/tracker state and let vLLM recompute locally.
        # The server has already released each failed worker's reader share.
        if (
            tracker.state == LMCacheMPRequestState.READY
            and request.num_computed_tokens == 0
            and tracker.num_lmcache_hit_tokens > 0
        ):
            if not commit_match:
                # A read-only estimate can report native recomputation without
                # consuming failure cleanup; admission/cancellation still owns
                # those state changes. Never start another LOOKUP for it.
                return 0, False
            logger.warning(
                "Bypassing LMCache for request %s after a failed async KV load; "
                "the prompt will be recomputed locally.",
                request.request_id,
            )
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
            tracker.allocated_block_ids.clear()
            tracker.num_stored_tokens = 0
            tracker.num_vllm_hit_tokens = 0
            tracker.num_lmcache_hit_tokens = 0
            tracker.state = LMCacheMPRequestState.BYPASS_LMCACHE
            return 0, False

        if allow_lookup_submit:
            self.scheduler_adapter.maybe_submit_lookup_request(
                request.request_id,
                token_ids=tracker.get_token_ids(),
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )

        if allow_lookup_progress:
            ret = self.scheduler_adapter.check_lookup_result(request.request_id)
        else:
            state = self.lookup_operation_state(request.request_id)
            ret = (state["result_tokens"] if state["result_known"] else
                   0 if state.get("native_fallback_reason") else None)
        if ret is None:
            return None, True

        if ret == 0:
            if commit_match:
                # A lower-cache miss can still reuse an HBM prefix. STORE
                # progress must include it; read-only preparation must not
                # commit this boundary before native admission.
                tracker.num_vllm_hit_tokens = (
                    num_computed_tokens
                    // self._hit_alignment_tokens
                    * self._hit_alignment_tokens
                )
            return 0, False

        assert ret % self.scheduler_adapter.lmcache_tokens_per_chunk == 0

        if commit_match:
            # Update num stored tokens for the tracker
            if idempotent_prefix_accounting:
                # ret is an absolute already-stored prefix, unlike the incremental
                # STORE commits that use the same += method. Preparation never
                # commits it; admission retries may observe it repeatedly.
                tracker.increase_num_stored_tokens(max(0, ret - tracker.num_stored_tokens))
            else:
                # Preserve the frozen Native entry point's accounting semantics.
                tracker.increase_num_stored_tokens(ret)

            # Save the vllm and lmcache hit tokens. The vLLM hit count is
            # rounded down to a boundary aligned for every engine group (e.g.
            # a full-prompt APC hit reports ``num_prompt_tokens - 1``), so the
            # retrieve-skip range stays paged-chunk-aligned in all groups.
            tracker.num_vllm_hit_tokens = (
                num_computed_tokens
                // self._hit_alignment_tokens
                * self._hit_alignment_tokens
            )
            tracker.num_lmcache_hit_tokens = ret

        need_to_load = max(0, ret - num_computed_tokens)

        # In full-prompt-hit case, we need to recompute the last token.
        # Without this, num_computed_tokens would equal request.num_tokens,
        # causing num_new_tokens to be 0 and triggering the
        # `assert num_new_tokens > 0` in the scheduler.
        if ret == len(request.all_token_ids):
            need_to_load = max(0, need_to_load - 1)

        logger.debug(
            "vLLM hit is: %d, Need to load is %d", num_computed_tokens, need_to_load
        )
        return need_to_load, need_to_load > 0

    def on_new_request(self, request: "Request") -> None:
        """Submit an LMCache lookup when a request enters the waiting queue.

        Args:
            request (Request): The request object.
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return
        if not self._eager_prefetch or request.resumable:
            return

        tracker = self._get_or_create_request_tracker(request)
        self.scheduler_adapter.maybe_submit_lookup_request(
            request.request_id,
            token_ids=tracker.get_token_ids(),
            cache_salt=tracker.cache_salt,
            request_configs=tracker.request_configs,
        )

    def validate_recompute_path(
        self, request: "Request", local_compute_start_tokens: int
    ) -> None:
        """Check the explicit local path without query or ownership side effects.

        Args:
            request: Owner about to take its first HBM allocation, or a native
                preemption re-admission of that already committed local path.
                The runtime must have refreshed and committed its prefix match.
            local_compute_start_tokens: Current HBM prefix, including native
                last-token handling. This is a scalar, not a resource credential.
        Raises:
            ValueError: The request has already adopted resources, its complete
                query result is unavailable, or the refreshed match disagrees.
        Notes:
            Call this before actual native allocation in the same synchronous
            processing branch. It reads only current request/tracker/query
            scalars; it never polls, allocates, releases or advances a watermark.
        """
        tracker = self.request_trackers.get(request.request_id)
        if tracker is None:
            raise ValueError("recompute path requires a live native request tracker")
        readmission = (
            request.status == RequestStatus.PREEMPTED
            and tracker.recompute_lower_prefix_tokens is not None
        )
        if (
            request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            or (request.status == RequestStatus.PREEMPTED and not readmission)
            or request.num_computed_tokens != 0
            or tracker.state != LMCacheMPRequestState.PREFETCHING
            or tracker.num_scheduled_tokens != 0
            or any(tracker.allocated_block_ids.values())
            or (tracker.local_compute_start_tokens is not None and not readmission)
        ):
            raise ValueError("recompute path requires first resource adoption")
        if (
            type(local_compute_start_tokens) is not int
            or not 0 <= local_compute_start_tokens < len(request.prompt_token_ids)
        ):
            raise ValueError("invalid local recompute prefix")
        if not readmission:
            state = self.lookup_operation_state(request.request_id)
            if not state["result_known"]:
                raise ValueError("recompute path requires a complete valid lookup")
            if tracker.num_lmcache_hit_tokens != state["result_tokens"]:
                raise ValueError("recompute path requires the current committed match")
        if (
            tracker.num_vllm_hit_tokens
            != local_compute_start_tokens
            // self._hit_alignment_tokens
            * self._hit_alignment_tokens
        ):
            raise ValueError("recompute path requires the current committed match")

    def commit_recompute_path(
        self, request: "Request", local_compute_start_tokens: int
    ) -> None:
        """Record the checked local path after successful native allocation.

        Args:
            request: The same request validated immediately before allocation.
            local_compute_start_tokens: The same current HBM prefix used by the
                successful allocation, including native last-token handling.
        Raises:
            ValueError: The synchronous validator contract has been violated.
        Notes:
            This grants no compute quota and performs no async observation.
            Follow with update_state_after_alloc(..., num_external_tokens=0);
            that native branch releases unused source locks and keeps saved
            lower coverage. Validation alone does not commit this path.
        """
        self.validate_recompute_path(request, local_compute_start_tokens)
        tracker = self._get_request_tracker(request.request_id)
        readmission = request.status == RequestStatus.PREEMPTED
        tracker.local_compute_start_tokens = local_compute_start_tokens
        if not readmission:
            tracker.recompute_lower_prefix_tokens = tracker.num_lmcache_hit_tokens
        wait_emit(
            "recompute_path_committed", request.request_id,
            local_compute_start_tokens=local_compute_start_tokens,
            lower_hit_tokens=tracker.num_lmcache_hit_tokens,
            saved_prefix_tokens=tracker.num_stored_tokens,
            native_readmission=readmission,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        # NOTE: `blocks` comes from kv_cache_manager.get_blocks(request_id),
        # which returns ALL blocks for the request (not just newly allocated).
        # This function may be called twice for async-load requests:
        #   1st call: blocks = initial allocation (APC + fresh)
        #   2nd call: blocks = all blocks
        #  (initial + newly allocated for remaining tokens)
        # We must only append the NEW blocks beyond what's already tracked
        # to avoid duplication, which would corrupt the store path's block indexing.
        tracker = self._get_request_tracker(request.request_id)
        if tracker.local_compute_start_tokens is not None and num_external_tokens:
            raise ValueError("committed recompute path cannot allocate external tokens")
        block_ids = blocks.get_block_ids() or ()

        # Only append blocks beyond what's already tracked, per engine group.
        existing_counts = tracker.num_allocated_blocks()
        new_block_ids: list[list[int]] = []
        for engine_group_idx, group_blocks in enumerate(block_ids):
            existing = existing_counts.get(engine_group_idx, 0)
            new_block_ids.append(list(group_blocks[existing:]))
        if any(new_block_ids):
            tracker.append_block_ids(tuple(new_block_ids))

        # Update the state of the tracker
        if tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE:
            # Returning zero external tokens admitted this request for local
            # computation.  Once vLLM publishes that allocation, normal READY
            # tracking (including later stores) can resume.
            tracker.state = LMCacheMPRequestState.READY
            return

        condition = num_external_tokens > 0 and tracker.needs_retrieve()
        if tracker.state == LMCacheMPRequestState.PREFETCHING:
            # If need to retrieve, change to WAITING_FOR_LOAD
            # Otherwise, change to READY
            tracker.state = (
                LMCacheMPRequestState.WAITING_FOR_LOAD
                if condition
                else LMCacheMPRequestState.READY
            )
            # Clean up lookup future in scheduler adapter
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)

            # Free locks on chunks that vLLM already computed and won't
            # retrieve from LMCache.
            if tracker.num_lmcache_hit_tokens > 0 and not (
                tracker.recompute_lower_prefix_tokens is not None
                and request.status == RequestStatus.PREEMPTED
            ):
                if not condition:
                    # No retrieve needed — free ALL locked chunks
                    free_end = tracker.num_lmcache_hit_tokens
                else:
                    # Note(Roy): Boundary misalignment between vLLM blocks and LMCache
                    # blocks is handled in free_lookup_locks. It makes sure that if
                    # the last vLLM computed block ends in the middle of a LMCache
                    # block, the end LMCache block is not freed (i.e., floor division)
                    # since it will still be needed by vLLM and such block's lock will
                    # be freed by vLLM's retrieve.
                    free_end = tracker.num_vllm_hit_tokens

                if free_end > 0:
                    self.scheduler_adapter.free_lookup_locks(
                        token_ids=tracker.get_token_ids(),
                        start=0,
                        end=free_end,
                        request_id=request.request_id,
                        cache_salt=tracker.cache_salt,
                        request_configs=tracker.request_configs,
                    )
                    if tracker.local_compute_start_tokens is not None:
                        wait_emit(
                            "recompute_source_release_requested", request.request_id,
                            start_token=0, end_token=free_end,
                            release_confirmed=False,
                            cleanup_owner="native_session_cleanup",
                        )
                    logger.debug(
                        "Free locks of tokens %d-%d since it is cached by vLLM.",
                        0,
                        free_end,
                    )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        metadata = LMCacheMPConnectorMetadata()
        metadata.need_flush_before_forward = _has_preemption_reqs(scheduler_output)

        self._process_retrieve_requests(metadata)
        self._process_new_requests(scheduler_output, metadata)
        self._process_cached_requests(scheduler_output, metadata)
        if self.decoupled_store_enabled:
            metadata.submit_store_ids = tuple(
                self._store_submissions[:self.operation_step_limit]
            )
            del self._store_submissions[:self.operation_step_limit]
            if getattr(self, "_operation_feedback_observer", None) is not None:
                for identity in metadata.submit_store_ids:
                    self._notify_operation_feedback(self._store_operation_views[identity],
                                                    "metadata_formed", native_phase="submit_control_metadata")
        if self._promotion is not None:
            metadata.promotions = self._promotion.prepare()

        if len(metadata) > 0:
            logger.debug("Final connector metadata: %s", metadata)

        # Report block allocation deltas to LMCache for observability
        self._report_block_allocation_deltas(scheduler_output)

        # Observe existing LOOKUP work only; adoption stays on a later native
        # request visit. Do not change this step's already-formed metadata.
        if not self.decoupled_control_enabled:
            self.scheduler_adapter.progress_pending_lookups()

        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, LMCacheMPWorkerMetadata):
            return
        if self.decoupled_store_enabled:
            for receipt in meta.store_receipts:
                feedback_enabled = getattr(self, "_operation_feedback_observer", None) is not None
                if feedback_enabled:
                    fingerprint = (receipt.request_id, receipt.start, receipt.end,
                                   receipt.phase, receipt.succeeded)
                    if receipt.phase not in ("committed", "submitted", "terminal"):
                        raise ValueError("unknown STORE receipt phase")
                    seen = self._feedback_store_receipts.get(receipt.operation_id)
                    ended = self._feedback_ended_stores.get(receipt.operation_id)
                    if fingerprint in (seen or ()) or fingerprint in (ended or ()):
                        continue
                    if ended is not None:
                        raise ValueError("late STORE receipt contradicts completed operation")
                view = self._store_operation_views.get(receipt.operation_id)
                if view is None:
                    raise ValueError("unknown STORE completion identity")
                if (view["request_id"] != receipt.request_id
                        or view["start_token"] != receipt.start
                        or view["end_token"] != receipt.end):
                    raise ValueError("STORE receipt range does not match commitment")
                view["stage"] = (
                    "completed_observed" if receipt.phase == "terminal"
                    else "submitted" if receipt.phase == "submitted"
                    else "dependencies_checked"
                )
                view["native_phase"] = receipt.phase
                if feedback_enabled:
                    seen = self._feedback_store_receipts.setdefault(receipt.operation_id, set())
                    seen.add(fingerprint)
                    outcome = ("failure_received" if receipt.phase == "terminal" and receipt.succeeded is False
                               else "completion_received" if receipt.phase == "terminal"
                               else "submission_observed" if receipt.phase == "submitted"
                               else "resources_acquired")
                    self._notify_operation_feedback(view, outcome, native_phase="worker_" + receipt.phase)
                if receipt.phase == "submitted":
                    self._retained_store_candidates.pop(receipt.operation_id, None)
                if receipt.phase == "terminal":
                    self._retained_store_candidates.pop(receipt.operation_id, None)
                    del self._store_operation_views[receipt.operation_id]
                    if feedback_enabled:
                        self._notify_operation_feedback(view, "completion_adopted", native_phase="store_receipt_adopted",
                                                        reason="store_failed" if receipt.succeeded is False else None)
                        self._notify_operation_feedback(view, "obligation_ended", native_phase="native_store_terminal",
                                                        reason="terminal_failure" if receipt.succeeded is False else None)
                        self._feedback_ended_stores[receipt.operation_id] = self._feedback_store_receipts.pop(receipt.operation_id)
                        while len(self._feedback_ended_stores) > self.operation_step_limit:
                            self._feedback_ended_stores.popitem(last=False)
        if meta.promotion_receipts:
            if self._promotion is None:
                raise ValueError("promotion receipt without enabled scheduler owner")
            for receipt in meta.promotion_receipts:
                self._promotion.adopt(receipt)
        if not self.lazy_offload:
            return
        if not self._gpu_block_pool:
            raise ValueError("Lazy offload is enabled but gpu block pool is not binded")
        for req_id, count in meta.completed_store_requests.items():
            if self.scheduler_adapter.update_pending_store_count(req_id, count):
                gpu_block_ids = self._pending_store.get_request_gpu_block_ids(req_id)
                self._gpu_block_pool.free_blocks(
                    [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )
                self._pending_store.remove_request_gpu_block_ids(req_id)
                self.scheduler_adapter.end_session(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        return_params: dict[str, Any] | None = {} if params is not None else None

        if (
            params is not None
            and return_params is not None
            and "cached_token_stats" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            num_vllm = request_tracker.num_vllm_hit_tokens
            num_lmcache = request_tracker.num_lmcache_hit_tokens
            return_params["cached_token_stats"] = {
                "num_vllm_cached_tokens": num_vllm,
                "num_lmcache_cached_tokens": num_lmcache,
                "num_lmcache_extra_cached_tokens": max(0, num_lmcache - num_vllm),
            }

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)

        # have not been offloaded, the touch operation in end_session is incorrect
        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)
        # Drop lookup state for a request aborted before its lookup was
        # consumed (update_state_after_alloc never ran for it).
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)

        if self.lazy_offload:
            self._pending_store.mark_req_finished(request.request_id)
            return False, (return_params or None)
        return True, (return_params or None)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA request-finished entry point; cleanup is request-id based."""
        return self.request_finished(request, block_ids[0] if block_ids else [])

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        return ()

    def has_pending_push_work(self) -> bool:
        """Return whether vLLM should keep stepping for pending push work.

        Returns:
            True when scheduler-side lazy offload has submitted stores waiting
            for worker completion. Queued stores are intentionally excluded:
            they require a model-token step for submission and cannot progress
            during a connector-only step. Non-lazy mode uses vLLM's normal
            delayed-free path and does not need this keepalive.
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return False
        if self.decoupled_store_enabled and self._store_operation_views:
            return True
        if self._promotion is not None and self._promotion.has_pending_work:
            return True
        if not self.lazy_offload:
            return False

        pending_store = getattr(self, "_pending_store", None)
        return pending_store is not None and pending_store.has_inflight_store_work()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Defer to vLLM; a connector preference is unsafe for now.

        Connector preferences outrank the default in vLLM's layout
        resolution, but backends that assume a layout without declaring it
        via ``supported_kv_cache_layouts`` (e.g. the DeepSeek V4 sparse
        attention, which hardcodes NHD) silently read a pool laid out
        differently. LMCache handles every resolved layout, so express no
        preference until such backends declare theirs.
        """
        return None

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics | None":
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return None

    ##############################
    # Helper functions
    ##############################
    def _process_retrieve_requests(
        self,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for request_tracker in self.request_trackers.values():
            if request_tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            r_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                group_tokens_per_block=self._group_tokens_per_block,
            )
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)
                if getattr(self, "_operation_feedback_observer", None) is not None:
                    self._notify_operation_feedback(dict(
                        operation_id="", request_id=r_metadata.request_id, kind="load",
                        start_token=r_metadata.op.start, end_token=r_metadata.op.end,
                        skip_first_n_tokens=r_metadata.op.skip_first_n_tokens),
                        "metadata_formed", native_phase="connector_retrieve_metadata")
            request_tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)

            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            if self._require_terminal_completion:
                metadata.add_compute_access(
                    request_tracker.allocated_block_ids,
                    self._group_tokens_per_block,
                    new_request.num_computed_tokens,
                    num_new_tokens,
                )

            r_meta = self._prepare_store_metadata(request_tracker, metadata)
            if r_meta is not None:
                # In lazy_offload mode, add to pending queue instead of immediate store
                if self.lazy_offload:
                    self._pending_store.add(r_meta)
                else:
                    metadata.add_request_metadata(r_meta)
        # if scheduler_output.total_num_scheduled_tokens is 0,
        # vllm `gpu_model_runner` will call `kv_connector_no_forward`
        # in `execute_model`, which will result in lose some store ops.
        # So we only trigger lazy offload when
        # scheduler_output.total_num_scheduled_tokens > 0
        if scheduler_output.total_num_scheduled_tokens:
            self._process_lazy_offload_store_requests(metadata)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids
            new_block_ids = cached_reqs.new_block_ids[idx] or ()
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)

            # Use the incremental num_scheduled_tokens to
            # stay consistent with _process_new_requests.
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            if self._require_terminal_completion:
                metadata.add_compute_access(
                    request_tracker.allocated_block_ids,
                    self._group_tokens_per_block,
                    cached_reqs.num_computed_tokens[idx],
                    num_new_tokens,
                )

            r_meta = self._prepare_store_metadata(request_tracker, metadata)

            if r_meta is not None:
                # In lazy_offload mode, add to pending queue instead of immediate store
                if self.lazy_offload:
                    self._pending_store.add(r_meta)
                else:
                    metadata.add_request_metadata(r_meta)
        # if scheduler_output.total_num_scheduled_tokens is 0,
        # vllm `gpu_model_runner` will call `kv_connector_no_forward`
        # in `execute_model`, which will result in lose some store ops.
        # So we only trigger lazy offload when
        # scheduler_output.total_num_scheduled_tokens > 0
        if scheduler_output.total_num_scheduled_tokens:
            self._process_lazy_offload_store_requests(metadata)

    def _prepare_store_metadata(
        self, tracker: LMCacheMPRequestTracker, metadata: LMCacheMPConnectorMetadata
    ) -> LMCacheMPRequestMetadata | None:
        """Discover and commit a native range; commitment is not physical completion."""
        chunk_size = self.scheduler_adapter.lmcache_tokens_per_chunk
        if not self.decoupled_store_enabled:
            candidate = LMCacheMPRequestMetadata.GetStoreMetadata(
                tracker, chunk_size, self._group_tokens_per_block
            )
            if candidate is not None and getattr(self, "_operation_feedback_observer", None) is not None:
                self._store_operation_sequence += 1
                view = dict(operation_id=f"{candidate.request_id}:native-store:{self._store_operation_sequence}",
                            request_id=candidate.request_id, kind="store",
                            start_token=candidate.op.start, end_token=candidate.op.end)
                # Observation identity only. Do not change native metadata or
                # turn immediate STORE into a retained worker operation.
                self._notify_operation_feedback(view, "resources_acquired", native_phase="native_store_metadata_commitment")
                self._notify_operation_feedback(view, "metadata_formed", native_phase="native_store_metadata")
            return candidate
        candidate = LMCacheMPRequestMetadata.PeekStoreMetadata(
            tracker, chunk_size, self._group_tokens_per_block
        )
        if candidate is None:
            return None
        self._store_operation_sequence += 1
        operation_id = (
            f"{candidate.request_id}:STORE:{candidate.op.start}:{candidate.op.end}:"
            f"{self._store_operation_sequence}"
        )
        view = dict(
            operation_id=operation_id, request_id=candidate.request_id, kind="store",
            start_token=candidate.op.start, end_token=candidate.op.end,
            stage="discovered", native_phase="candidate",
            dependencies=(("source_owned", "confirmed"),
                          ("producer_event", "not_recorded"),
                          ("physical_completion", "unknown")),
            block_reason=None, owner="native_request_and_worker_page_guard",
            source_ownership=True, physical_ready="unknown",
            resources=(
                ("source_pages", sum(len(b) for b in candidate.op.block_ids)),
                ("logical_tokens", candidate.op.end - candidate.op.start),
            ),
        )
        action = self.decide_store_operation(dict(view))
        if action not in ("submit", "defer"):
            raise ValueError("STORE decision must be submit or defer")
        LMCacheMPRequestMetadata.CommitStoreMetadata(
            tracker, candidate, chunk_size, self._group_tokens_per_block
        )
        candidate.operation_id = operation_id
        view["stage"] = "dependencies_checked"
        view["native_phase"] = "committed_metadata"
        self._store_operation_views[operation_id] = view
        if getattr(self, "_operation_feedback_observer", None) is not None:
            self._notify_operation_feedback(view, "resources_acquired", native_phase="store_metadata_commitment")
            self._notify_operation_feedback(view, "metadata_formed", native_phase="store_metadata")
        if action == "defer":
            metadata.defer_store_ids += (operation_id,)
            self._retained_store_candidates[operation_id] = None
        return candidate

    def _process_lazy_offload_store_requests(
        self, metadata: LMCacheMPConnectorMetadata
    ):
        if not self.lazy_offload:
            return

        if not self._gpu_block_pool:
            raise ValueError("Lazy offload is enabled but no GPU block pool is bound")

        # Each item aggregates store metadata for one request. Chunked prefill
        # or the scheduler's ``max-num-batched-tokens`` limit can schedule one
        # request multiple times, with each metadata entry containing only that
        # scheduling pass's blocks.
        for item in self._pending_store.pop_items_for_offload():
            request_id = item.request_id
            for meta, old_block_hashes in item.metadatas:
                gpu_block_ids = list(old_block_hashes.keys())
                self._gpu_block_pool.touch(
                    [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )
                new_block_hashes = {
                    bid: self._gpu_block_pool.blocks[bid].block_hash
                    for bid in gpu_block_ids
                }
                if old_block_hashes == new_block_hashes:
                    # remove block hashes and free blocks until store is done
                    metadata.add_request_metadata(meta)
                    self._pending_store.update_request_gpu_block_ids(
                        request_id, gpu_block_ids
                    )
                else:
                    logger.warning(
                        "Part block hashes mismatch for request %s, skip it",
                        request_id,
                    )
                    self._gpu_block_pool.free_blocks(
                        [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                    )
                    break

    def _report_block_allocation_deltas(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Gather per-request block allocation deltas and report to LMCache.

        For new requests: all allocated_block_ids and token_ids are new.
        For cached requests: only newly appended block_ids and token_ids.
        The L0 allocation telemetry is flat today, so HMA reports engine group 0.
        """
        records: list[RequestAllocationRecord] = []

        # New requests: send all tokens covering all allocated blocks so
        # the L0 metrics subscriber can correctly map each block to its
        # actual token content (not just the newly-scheduled slice).
        for new_request in scheduler_output.scheduled_new_reqs:
            tracker = self.request_trackers.get(new_request.req_id)
            if tracker is None:
                continue
            primary_block_ids = tracker.allocated_block_ids.get(0, [])
            num_blocks = len(primary_block_ids)
            total_tokens = num_blocks * self._group_tokens_per_block[0]
            records.append(
                RequestAllocationRecord(
                    req_id=new_request.req_id,
                    new_block_ids=list(primary_block_ids),
                    new_token_ids=tracker.get_token_ids()[:total_tokens],
                )
            )

        # Cached requests: only the newly added blocks and their full
        # token content.  We send all tokens covered by the new blocks
        # (not just the tokens scheduled this step) so the L0 subscriber
        # can correctly identify block content.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            # The L0 subscriber works on the primary (group 0) block-id list.
            new_group_block_ids = cached_reqs.new_block_ids[idx]
            new_block_ids = new_group_block_ids[0] if new_group_block_ids else []
            if not new_block_ids:
                continue
            tracker = self.request_trackers.get(request_id)
            if tracker is None:
                continue
            # The new blocks sit at the end of the request's block list.
            # Compute the token range they cover.
            total_blocks = len(tracker.allocated_block_ids.get(0, []))
            num_new_blocks = len(new_block_ids)
            tokens_per_block = self._group_tokens_per_block[0]
            start_token = (total_blocks - num_new_blocks) * tokens_per_block
            end_token = total_blocks * tokens_per_block
            new_token_ids = tracker.get_token_ids()[start_token:end_token]
            records.append(
                RequestAllocationRecord(
                    req_id=request_id,
                    new_block_ids=new_block_ids,
                    new_token_ids=new_token_ids,
                )
            )

        if records:
            self.scheduler_adapter.report_block_allocations(records)

    def _get_request_tracker(self, request_id: str) -> LMCacheMPRequestTracker:
        assert request_id in self.request_trackers, (
            f"Request tracker for request_id {request_id} not found. "
        )
        return self.request_trackers[request_id]

    def _get_or_create_request_tracker(
        self, request: "Request"
    ) -> LMCacheMPRequestTracker:
        request_id = request.request_id
        recompute_lower_prefix = None
        # Remove the old trackers that is created before the preemption
        if (
            request.status == RequestStatus.PREEMPTED
            and request_id in self.request_trackers
        ):
            tracker = self.request_trackers[request_id]

            # NOTE: since this function may be called multiple times
            # for a single request (because get_num_new_matched_tokens
            # may be called multiple times) for the same request, we
            # will only do the remove if the tracker is not in the "fresh"
            # state, i.e., PREFETCHING
            if tracker.state != LMCacheMPRequestState.PREFETCHING:
                recompute_lower_prefix = tracker.recompute_lower_prefix_tokens
                self.request_trackers.pop(request_id)

        if request_id not in self.request_trackers:
            new_tracker = LMCacheMPRequestTracker(request)
            if recompute_lower_prefix is not None:
                # Preserve only the original known L1 copy. Later STORE
                # commitments are not a physical-completion receipt, so do
                # not carry their watermark through native preemption.
                new_tracker.recompute_lower_prefix_tokens = recompute_lower_prefix
                new_tracker.num_lmcache_hit_tokens = recompute_lower_prefix
                new_tracker.num_stored_tokens = recompute_lower_prefix
                new_tracker.local_compute_start_tokens = 0
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _cleanup_request_tracker(self, request_id: str) -> None:
        """
        Clean up request tracker and associated lookup future for a request.
        This should be called when a request is finished to prevent memory leak.
        """
        # Clean up request tracker
        if self.request_trackers.pop(request_id, None):
            logger.debug(
                "[KVConnector] Cleaned up request_tracker for request %s",
                request_id,
            )
