# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Protocol
import enum
import math
import os
import threading
import time
import uuid

# Third Party
import torch
from vllm.v1.kv_wait_trace import emit as wait_emit, span as wait_span
import zmq

# First Party
from lmcache import torch_dev
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.hybrid_promotion import (
    PromotionCommand,
    PromotionCompletions,
    PromotionReceipt,
)
from lmcache.integration.vllm.hybrid_session_cleanup import ConfirmedSessionCleanup
from lmcache.integration.vllm.kv_page_ownership import PendingKVPageAccess
from lmcache.integration.vllm.phase_hooks import (
    observed_retirement_call,
    observed_worker_call,
    worker_observation_span,
)
from lmcache.integration.vllm.utils import vllm_layout_hints
from lmcache.utils import EngineType, _lmcache_nvtx_annotate, init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
)
from lmcache.v1.multiprocess.mq import MessagingFuture
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    LMCacheDrivenTransferContext,
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary
from lmcache.v1.platform.cuda.vmm_ipc import set_use_vmm_api
from lmcache.v1.platform.isolated_ipc import set_isolated_ipc

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.experimental import Dispatcher

logger = init_logger(__name__)


class ExtraConfigDefault(enum.Enum):
    """Centralized default values for extra_config keys.

    Each member's *name* is the key used in the extra_config dict, and
    its ``default`` attribute is the default value. Defaults live in an
    attribute rather than the enum value because equal enum values
    silently alias members (two ``False`` defaults would collapse into
    one member); each member's ``_value_`` is a unique ordinal instead.
    """

    #: The default value for this key (annotation only -- enum members
    #: are created from the assigned literals below, not from this).
    default: Any

    def __new__(cls, default: Any) -> "ExtraConfigDefault":
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.default = default
        return obj

    # Timeout (seconds) for blocking MQ requests: initial
    # chunk-size query, KV cache registration/unregistration,
    # and other synchronous operations.
    mq_timeout = 300.0
    # Interval (seconds) between periodic heartbeat pings
    # to the server.
    heartbeat_interval = 10.0
    # Poll status replies without blocking the scheduler by default.
    nonblocking_lookup_status = True
    # Routing mode for ``create_transfer_context``: ``auto`` keeps the
    # historical CUDA -> lmcache_driven / others -> engine_driven dispatch;
    # ``lmcache_driven`` forces the IPC / SHM zero-copy path where the
    # LMCache server pulls data via device handles;
    # ``engine_driven`` forces the worker-side gather/scatter copy path.
    # Mirrors the ``LMCACHE_MP_TRANSFER_MODE`` env var; this extra_config
    # key wins when both are set.
    mp_transfer_mode = "auto"
    # Whether IPC mechanisms must work across isolated containers (no
    # shared host IPC namespace or /dev/shm); see
    # lmcache/v1/platform/isolated_ipc.py. Must match the LMCache server's
    # ``--isolated-ipc`` setting.
    isolated_ipc = False
    # Whether the engine allocates its KV cache through the CUDA VMM API
    # (vLLM's ``--enable-cumem-allocator``), so KV registration must use
    # VMM IPC instead of legacy CUDA IPC handles; see
    # lmcache/v1/platform/cuda/vmm_ipc.py.
    use_vmm_api = False
    # Hybrid server replies prove both CUDA and model completion. Heartbeat
    # loss cannot replace that proof or authorize GPU block reuse.
    require_terminal_completion = False


# Backward-compatible aliases for the legacy `lmcache_mp_connector_0180`
# entry point, which still passes these as positional/keyword args.
DEFAULT_MQ_TIMEOUT: float = ExtraConfigDefault.mq_timeout.default
DEFAULT_HEARTBEAT_INTERVAL: float = ExtraConfigDefault.heartbeat_interval.default

_EXTRA_CONFIG_KEY_PREFIX = "lmcache.mp."

# Floor (seconds) of the MP server's worker reap timeout. It only covers the
# default 10 s heartbeat interval (3 x 10 s); the adapter warns at startup
# when 3 x heartbeat_interval exceeds it (server timeout must be raised too).
_SERVER_REAP_TIMEOUT_FLOOR_SECONDS: float = 30.0


def _coerce_extra_config_value(default: Any, raw: Any) -> Any:
    """Coerce a user-provided extra_config value to its default's type.

    Args:
        default: The :class:`ExtraConfigDefault` member value, defining the
            target type.
        raw: The user-provided value (typed when the connector config came
            from JSON, a string when it came from a CLI passthrough).

    Returns:
        ``raw`` coerced to ``type(default)``. Booleans accept the usual
        string spellings (``1/true/yes/on``, case-insensitive) because
        ``bool("false")`` is ``True``.
    """
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return type(default)(raw)


def _resolve_extra_config(
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve extra_config against :class:`ExtraConfigDefault`.

    Keys in *extra_config* are expected to carry the
    ``lmcache.mp.`` prefix (e.g. ``lmcache.mp.mq_timeout``).
    The prefix is stripped before matching against
    :class:`ExtraConfigDefault` members.

    All ``lmcache.mp.*`` entries are logged.  Entries whose
    value differs from the default are additionally marked as
    *overridden*.

    Args:
        extra_config: User-supplied config dict (may be *None*).

    Returns:
        A dict keyed by :pyattr:`ExtraConfigDefault` member names.
    """
    stripped: dict[str, Any] = {}
    if extra_config is not None:
        for k, v in extra_config.items():
            if k.startswith(_EXTRA_CONFIG_KEY_PREFIX):
                short = k[len(_EXTRA_CONFIG_KEY_PREFIX) :]
                stripped[short] = v

    resolved: dict[str, Any] = {}
    for item in ExtraConfigDefault:
        default = item.default
        raw = stripped.get(item.name)
        value = _coerce_extra_config_value(default, raw) if raw is not None else default
        if value != default:
            logger.info(
                "%s%s = %s (overridden, default: %s)",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
                default,
            )
        else:
            logger.info(
                "%s%s = %s",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
            )
        resolved[item.name] = value
    return resolved


class _IpcEvent(Protocol):
    def wait(self, stream: Any = None) -> None: ...


@dataclass
class _LookupAck:
    """LOOKUP requests whose server acknowledgement has not been observed yet.

    ``LMCacheMPSchedulerAdapter.maybe_submit_lookup_request`` sends one LOOKUP
    per server and returns without waiting for the reply.  The futures are
    kept here so that ``check_lookup_result`` can confirm every server has
    registered the prefetch job before it sends QUERY_PREFETCH_STATUS, and so
    a reply that never arrives can be detected via ``submitted_at``.
    """

    futures: dict[str, MessagingFuture[None]]
    """Outstanding LOOKUP futures keyed by server URL; acked ones are removed."""

    submitted_at: float
    """``time.monotonic()`` timestamp taken when the LOOKUPs were sent."""


def get_lmcache_chunk_size(
    req_client: RequestClient,
    timeout: float = DEFAULT_MQ_TIMEOUT,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        req_client: The LMCache multiprocess request client.
        timeout: Timeout in seconds for the blocking request.

    Returns:
        An integer representing the LMCache chunk size
    """
    future = req_client.get_chunk_size()
    lmcache_tokens_per_chunk = future.result(timeout=timeout)
    return lmcache_tokens_per_chunk


def get_experimental(
    req_client: RequestClient,
    timeout: float = DEFAULT_MQ_TIMEOUT,
) -> set[str]:
    """Query the experimental capabilities a server advertises.

    Args:
        req_client: The LMCache multiprocess request client.
        timeout: Seconds to wait for the server's response.

    Returns:
        Experimental features built into the server (now `transfer_query`).
    """
    future = req_client.get_experimental()
    try:
        return set(future.result(timeout=timeout))
    except TimeoutError:
        logger.warning(
            "LMCache server did not answer GET_EXPERIMENTAL within %ss; "
            "treating it as advertising no experimental capabilities.",
            timeout,
        )
        return set()


def _raise_server_unreachable(server_url: str, timeout: float) -> NoReturn:
    """Raise a verbose ConnectionError when the LMCache MP server is
    unreachable.

    The message intentionally spells out the most common cause (the
    standalone ``lmcache server`` process is not running), the URL that
    was being dialed, and the exact command to start it -- so that users
    landing here via ``vllm serve --kv-offloading-backend lmcache`` are
    not left guessing.
    """
    hint = (
        "Cannot reach the LMCache MP server at "
        f"'{server_url}' within {timeout}s.\n"
        "This usually means the standalone LMCache server is not "
        "running, or it is listening on a different host/port.\n"
        "To start one locally with the default port (5555):\n"
        "    lmcache server --l1-size-gb 20 --eviction-policy LRU\n"
        "To target a different host/port, override via "
        "kv_connector_extra_config (lmcache.mp.host / lmcache.mp.port), "
        "e.g.:\n"
        '    --kv-transfer-config \'{"kv_connector":'
        '"LMCacheMPConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"lmcache.mp.host":'
        '"tcp://localhost","lmcache.mp.port":5555}}\'\n'
        "See https://docs.lmcache.ai/mp/quickstart.html for details."
    )
    logger.warning(hint)
    raise ConnectionError(hint) from None


def send_ping(
    req_client: RequestClient,
    timeout: float,
    instance_id: int | None = None,
) -> bool:
    """Send a PING request and return the result.

    Args:
        req_client: The request client.
        timeout: Seconds to wait for the server's response.
        instance_id: The worker's instance ID so the server can refresh its
            liveness, or None for an untracked prober (scheduler adapter).

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = req_client.ping(instance_id)
        return future.result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("Ping failed with exception", exc_info=True)
        return False


@dataclass
class ParallelStrategy:
    mla_only: bool
    """
    True only for pure-MLA models. This flag indicates whether the KV cache
    tensor is replicated across TP ranks.
    """

    vllm_world_size: int
    """Number of workers managed by one vLLM scheduler (TP × PP; excludes DP).

    Mirrors ``vllm.parallel_config.world_size``.
    """

    vllm_worker_id: int
    """This worker's rank within its scheduler group."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""

    n_servers: int
    """Number of LMCache servers backing this deployment"""

    dcp_size: int = 1
    """Decode-context-parallel size: each chunk's KV is split into
    ``dcp_size`` token shards instead of one replicated copy."""

    @property
    def kv_world_size(self) -> int:
        """Number of pieces a single token chunk's KV cache is split into
        on the LMCache server storage."""
        if self.mla_only:
            # MLA replicates KV across TP; distinct bytes exist only per
            # pipeline stage and per DCP token shard.
            pp_stages = self.vllm_world_size // self.tp_size
            return pp_stages * self.dcp_size
        return self.vllm_world_size // self.n_servers

    @property
    def kv_worker_id(self) -> int:
        """Index of the piece of a single token chunk's KV cache
        that the current worker is responsible for,
        in ``[0, kv_world_size)``."""
        if self.mla_only:
            # Row-major over (pipeline stage, token shard).
            pp_stage = self.vllm_worker_id // self.tp_size
            return pp_stage * self.dcp_size + self._get_dcp_rank()
        return self.vllm_worker_id % (self.vllm_world_size // self.n_servers)

    def _get_dcp_rank(self) -> int:
        """Token shard index in ``[0, dcp_size)``: vLLM builds DCP groups as
        contiguous runs within each TP group, so ``tp_rank % dcp_size``."""
        return (self.vllm_worker_id % self.tp_size) % self.dcp_size

    @property
    def num_kv_readers(self) -> int:
        """Number of workers that retrieve one stored object.

        Non-MLA: 1. MLA: the same-shard TP ranks of one stage on one
        server, ``ceil(kv_tp_size / dcp_size)`` -- rounded up so an uneven
        split never under-reserves (the server takes one read lock per
        declared reader).
        """
        if not self.mla_only:
            return 1
        return math.ceil(self.kv_tp_size / self.dcp_size)

    @property
    def kv_tp_size(self) -> int:
        """Tensor-parallel size as seen from a single LMCache server."""
        return self.tp_size // self.n_servers

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank is responsible for storing KV."""
        if not self.mla_only:
            return True
        if self.dcp_size > 1:
            # One writer per token shard within every server and every
            # pipeline stage; both are contiguous rank runs, so the first
            # dcp_size ranks of the smaller run cover each shard once.
            ranks_per_server = self.vllm_world_size // self.n_servers
            block = min(ranks_per_server, self.tp_size)
            return (self.vllm_worker_id % block) < self.dcp_size
        # MLA-only: only first rank per node is a writer.
        return self.vllm_worker_id % (self.tp_size // self.n_servers) == 0


def _normalize_adapter_init_args(
    vllm_block_size: int,
    parallel_strategy: ParallelStrategy | int,
    legacy_block_size: int | None,
    mq_timeout: float,
) -> tuple[int, ParallelStrategy, float]:
    """Normalize adapter constructor args from old and new vLLM connectors.

    Args:
        vllm_block_size: The vLLM block size for the current connector API, or
            the legacy KV world size when ``parallel_strategy`` is an int.
        parallel_strategy: The current ``ParallelStrategy`` object, or the
            legacy KV worker id from older vLLM MP connectors.
        legacy_block_size: The legacy vLLM block size passed positionally by
            older vLLM MP connectors.
        mq_timeout: Timeout in seconds for synchronous message queue requests.

    Returns:
        A tuple of normalized ``(vllm_block_size, parallel_strategy,
        mq_timeout)``.

    Raises:
        TypeError: If the connector argument shape is not supported.
    """
    if isinstance(parallel_strategy, ParallelStrategy):
        return vllm_block_size, parallel_strategy, mq_timeout
    if not isinstance(parallel_strategy, int) or legacy_block_size is None:
        raise TypeError(
            "parallel_strategy must be ParallelStrategy, or legacy "
            "(kv_world_size, kv_worker_id, block_size) arguments"
        )

    kv_world_size = int(vllm_block_size)
    kv_worker_id = int(parallel_strategy)
    strategy = ParallelStrategy(
        mla_only=False,
        vllm_world_size=kv_world_size,
        vllm_worker_id=kv_worker_id,
        tp_size=kv_world_size,
        pp_size=1,
        n_servers=1,
    )
    return int(legacy_block_size), strategy, mq_timeout


class HeartbeatThread(PeriodicThread):
    """Periodically checks server health via PING.

    Manages a threading.Event that adapters use to gate operations.
    When unhealthy, the adapter enters degraded mode; if the server
    recovers, the adapter automatically resumes normal operation.
    """

    def __init__(
        self,
        req_client: RequestClient,
        health_event: threading.Event,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        instance_id: int | None = None,
    ):
        """
        Args:
            req_client: The request client used to send PING requests.
            health_event: A threading.Event shared with the adapter.
                Set when the server is healthy, cleared when unhealthy.
                Adapters check this event to decide whether to proceed
                with operations or enter degraded mode.
            interval: Seconds between heartbeat pings and ping timeout.
            instance_id: The worker's instance ID sent with each PING so the
                server can refresh its liveness, or None for an untracked
                prober (the scheduler adapter).
        """
        super().__init__(
            name="lmcache-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._req_client = req_client
        self._health_event = health_event
        self._interval = interval
        self._instance_id = instance_id

        # Optional callback invoked on the unhealthy->healthy edge,
        # before the health event is set. See register_recover_callback.
        def noop() -> bool:
            return True

        self._recover_callback: Callable[[], bool] = noop

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callback fired on the unhealthy->healthy transition.

        The callback runs **before** the health event is set. It must
        return ``True`` on success (event will be set) or ``False`` on
        failure (event will stay cleared, and the next heartbeat will
        invoke the callback again on the next successful PING).

        The callback function should NEVER raise exceptions.

        Intended for setup work that must complete before downstream
        callers observe the recovery — for example, re-registering KV
        caches with a server that just restarted.

        Should be called before :meth:`start`. Only one callback is
        supported; a second call replaces the first.

        Args:
            callback: Zero-arg callable returning a success bool.
        """
        self._recover_callback = callback

    def _execute(self) -> ThreadRunSummary:
        """Run one heartbeat cycle: ping, recover callback, event update.

        A cycle that observes a stop request returns without firing the
        callback or touching the event — a straggler success after
        UNREGISTER must not re-register a ghost context.
        """
        was_healthy = self._health_event.is_set()
        healthy = send_ping(
            self._req_client, timeout=self._interval, instance_id=self._instance_id
        )

        if self.stop_requested:
            return ThreadRunSummary(
                success=True,
                message="stop requested; skipping health update",
            )

        need_trigger_recover = (
            healthy and not was_healthy and self._recover_callback is not None
        )

        # Try to call recover callback
        if need_trigger_recover:
            logger.warning(
                "LMCache server is healthy again, triggering recovery callback"
            )
            # If the callback fails, it should not become healthy
            healthy = self._recover_callback()

        if healthy:
            self._health_event.set()
            if not was_healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        else:
            self._health_event.clear()
            if was_healthy:
                logger.warning("LMCache server is unhealthy — entering degraded mode")

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    """Token IDs for the load/store operation"""

    block_ids: list[list[int]]
    """Block IDs for the load/store operation, indexed by engine KV cache
    group (one inner list per engine group). Worker submit paths expand
    this to LMCache KV group order before sending requests to the server.
    """

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    @property
    def flat_block_ids(self) -> list[int]:
        """Return all block IDs flattened for group-blind error paths.

        Handles both the normal ``list[list[int]]`` format and the
        IPC-flattened ``list[int]`` format that vLLM v0.19.0 produces when
        ``SchedulerOutput`` serializes single-element nested lists across
        process boundaries (e.g. ``[[20, 21]]`` → ``[20, 21]``).
        Returns an empty list when ``block_ids`` is empty.
        """
        if not self.block_ids:
            return []
        # Defend against IPC serialization flattening [[20, 21, …]] → [20, 21, …]
        if isinstance(self.block_ids[0], int):
            return list(self.block_ids)
        return [
            block_id
            for group_block_ids in self.block_ids
            for block_id in group_block_ids
        ]


StoreResult = bool
RetrieveResult = bool
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_urls: list[str],
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        extra_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            server_urls: LMCache multiprocess request server URLs.
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `mla_only`,
                `kv_world_size`, `kv_worker_id` and so on. Older vLLM
                connectors pass the KV worker id here.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
                Ignored when ``extra_config`` is provided.
            heartbeat_interval: Interval in seconds between heartbeat pings.
                Ignored when ``extra_config`` is provided.
            extra_config: Optional dict with keys starting with
                ``lmcache.mp.`` (e.g., ``lmcache.mp.mq_timeout``). When
                provided, it overrides ``mq_timeout`` / ``heartbeat_interval``.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        assert len(server_urls) >= 1, "At least one server url required"
        self._server_urls: list[str] = list(server_urls)
        self.req_clients: dict[str, RequestClient] = {
            url: RequestClientFactory.create(url, context=context)
            for url in self._server_urls
        }
        self._nonblocking_lookup_status = (
            ExtraConfigDefault.nonblocking_lookup_status.default
        )
        confirmed_cleanup = False
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
            self._nonblocking_lookup_status = cfg[
                ExtraConfigDefault.nonblocking_lookup_status.name
            ]
            confirmed_cleanup = cfg[ExtraConfigDefault.require_terminal_completion.name]
        self._mq_timeout = mq_timeout

        # Lookup state tracking:
        # - _pending_lookups: request_ids submitted but not yet resolved
        # - _unacked_lookups: LOOKUP futures not yet acknowledged by every
        #   server. check_lookup_result never sends QUERY_PREFETCH_STATUS for
        #   a request until its LOOKUP has been acked (see _LookupAck).
        # - _finished_lookup_results: cached chunk count keyed by request_id,
        #   so that repeated calls to check_lookup_result return the same value
        #   even after the server has already popped the job (exactly-once).
        # - _per_server_hits: {request_id: {server_url: hit_chunks}}.
        #   Per-server hit counts, used to detect disagreement and free tail locks.
        self._pending_lookups: set[str] = set()
        self._unacked_lookups: dict[str, _LookupAck] = {}
        self._finished_lookup_results: dict[str, int] = {}
        self._per_server_hits: dict[str, dict[str, int]] = {}
        # request_id -> server URL -> (in-flight status future, submission time).
        self._lookup_status: dict[
            str, dict[str, tuple[MessagingFuture[Any], float]]
        ] = {}
        self._lookup_params: dict[
            str, tuple[list[int], str, dict[str, Any] | None]
        ] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        chunk_sizes: dict[str, int] = {}
        for url, client in self.req_clients.items():
            try:
                chunk_sizes[url] = get_lmcache_chunk_size(
                    client, timeout=self._mq_timeout
                )
            except TimeoutError:
                for c in self.req_clients.values():
                    c.close()
                _raise_server_unreachable(url, self._mq_timeout)

        # All servers must share chunk_size, otherwise the min() aggregation
        # over per-server hits would mix different granularities.
        unique_sizes = set(chunk_sizes.values())
        if len(unique_sizes) != 1:
            raise ValueError(
                f"All LMCache servers must share the same chunk_size, got {chunk_sizes}"
            )
        self.lmcache_tokens_per_chunk = unique_sizes.pop()

        if self.lmcache_tokens_per_chunk % vllm_block_size != 0:
            raise ValueError(
                f"LMCache chunk size {self.lmcache_tokens_per_chunk} must be a "
                f"multiple of {vllm_block_size} (the vLLM block size scaled by "
                f"decode_context_parallel_size). Set --chunk-size to a multiple "
                f"of {vllm_block_size}."
            )
        self.blocks_in_chunk = self.lmcache_tokens_per_chunk // vllm_block_size

        # Health state: one Event per server. The adapter is considered healthy
        # only if ALL per-server events are set (any unhealthy server taints
        # the whole adapter, matching the min() semantics used for lookups).
        self._health_events: dict[str, threading.Event] = {}
        for url in self._server_urls:
            ev = threading.Event()
            ev.set()
            self._health_events[url] = ev

        self._owned_lookup_ids: set[str] = set()
        self._session_cleanup: ConfirmedSessionCleanup | None = None
        if confirmed_cleanup:
            try:
                self._session_cleanup = ConfirmedSessionCleanup(
                    self.req_clients, self._health_events, self._mq_timeout
                )
            except BaseException:
                for client in self.req_clients.values():
                    client.close()
                raise

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first lookup
        # request, by which time vLLM is fully ready.
        self._heartbeat_interval = heartbeat_interval
        self._heartbeats: dict[str, HeartbeatThread] = {}
        self._heartbeat_lock = threading.Lock()

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = parallel_strategy.vllm_world_size
        self._store_request_pending_counts: dict[str, int] = {}

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        return self.parallel_strategy.kv_tp_size

    @property
    def is_healthy(self) -> bool:
        """True iff every backing LMCache server is healthy."""
        return all(ev.is_set() for ev in self._health_events.values())

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeats:
            return
        with self._heartbeat_lock:
            if self._heartbeats:
                return
            for url, client in self.req_clients.items():
                hb = HeartbeatThread(
                    req_client=client,
                    health_event=self._health_events[url],
                    interval=self._heartbeat_interval,
                )
                owner = self._session_cleanup
                if owner is not None:
                    hb.register_recover_callback(partial(owner.recover, url))
                self._heartbeats[url] = hb
                hb.start()

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends LOOKUP to every server without waiting for the ack, so the
        scheduler thread is not blocked here.  ``check_lookup_result``
        polls the acks and then queries the prefetch result.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache
            cache_salt: Per-user isolation salt. Requests with different
                cache_salt values produce separate cache entries.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (
            len(token_ids) // self.lmcache_tokens_per_chunk
        ) * self.lmcache_tokens_per_chunk

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        ).no_worker_id_version()

        if self._session_cleanup is not None:
            self._owned_lookup_ids.add(request_id)
        futures: dict[str, MessagingFuture[Any]] = {
            url: (
                self.req_clients[url].lookup_guarded(
                    key, self.tp_size, self._session_cleanup.generations[url]
                )
                if self._session_cleanup is not None
                else self.req_clients[url].lookup(key, self.tp_size)
            )
            for url in self._server_urls
        }

        # Do not wait for the acknowledgement here: that is a full server
        # round trip on the scheduler thread. check_lookup_result resolves
        # the futures lazily, one non-blocking poll per call.
        self._unacked_lookups[request_id] = _LookupAck(
            futures=futures, submitted_at=time.monotonic()
        )
        wait_emit("lookup_submitted", request_id, tokens=len(token_ids))
        self._pending_lookups.add(request_id)
        self._lookup_params[request_id] = (token_ids, cache_salt, request_configs)

    def _free_inconsistent_lookup_locks(
        self,
        request_id: str,
        per_server: dict[str, int],
        min_chunks: int,
    ) -> None:
        """Release over-hit tail locks on servers that reported more than min.

        When servers disagree on hit chunk counts, servers reporting more
        than min_chunks have locked chunks in the range
        [min_chunks * tokens_per_chunk, hit_chunks * tokens_per_chunk)
        that will never be retrieved. This method frees those tail locks.

        Args:
            request_id: The lookup request ID.
            per_server: Per-server hit chunk counts.
            min_chunks: Minimum hit chunk count across all servers.
        """
        token_ids_l, cs, request_configs = self._lookup_params.pop(
            request_id, (None, None, None)
        )
        if token_ids_l is not None:
            for url, hit_chunks in per_server.items():
                if hit_chunks <= min_chunks:
                    continue
                tail_end = min(
                    hit_chunks * self.lmcache_tokens_per_chunk, len(token_ids_l)
                )
                tail_key = self._create_key(
                    token_ids=token_ids_l,
                    start=min_chunks * self.lmcache_tokens_per_chunk,
                    end=tail_end,
                    request_id=request_id,
                    cache_salt=cs or "",
                    request_configs=request_configs,
                ).no_worker_id_version()
                self.req_clients[url].free_lookup_locks(tail_key, self.tp_size)

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        First polls, without blocking, whether every server has acknowledged
        the LOOKUP sent by ``maybe_submit_lookup_request``; while any ack is
        outstanding this returns None. Once all servers have acked, polls one
        outstanding QUERY_PREFETCH_STATUS future per unresolved server without
        waiting by default. Setting ``lmcache.mp.nonblocking_lookup_status`` to
        False instead waits for each reply in the current callback. Returns the
        matched token count when the prefetch is complete, or None if still
        in progress.

        A LOOKUP that is not acknowledged within the MQ timeout marks that
        server unhealthy and makes this return 0, matching the behaviour of
        a timed-out synchronous submit.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # Return the cached aggregate if any, otherwise 0.
            return self._finished_lookup_results.get(request_id, 0)

        if not self.is_healthy:
            # Server went down — give up on this lookup
            return 0

        if request_id in self._finished_lookup_results:
            # Aggregation already done; return the cached value.
            return self._finished_lookup_results[request_id]

        ack = self._unacked_lookups.get(request_id)
        if ack is not None:
            # The server registers the prefetch job while handling LOOKUP.
            # LOOKUP and QUERY_PREFETCH_STATUS run on the same server thread
            # pool and may be reordered, and a status query for an unknown
            # request_id answers 0: a spurious miss whose prefetch locks are
            # never released. So never query until every server has acked.
            waiting = {}
            for url, future in ack.futures.items():
                if not future.query():
                    waiting[url] = future
                elif self._session_cleanup is not None:
                    try:
                        accepted = future.result(timeout=0)
                    except Exception:
                        accepted = False
                    if accepted is not True:
                        self._mark_lookup_timed_out(url)
                        del self._unacked_lookups[request_id]
                        return 0
            ack.futures = waiting
            if ack.futures:
                if time.monotonic() - ack.submitted_at >= self._mq_timeout:
                    for url in ack.futures:
                        self._mark_lookup_timed_out(url)
                    del self._unacked_lookups[request_id]
                    return 0
                # Acknowledgement still in flight; poll again next step.
                return None
            wait_emit("lookup_ack_observed", request_id)
            del self._unacked_lookups[request_id]

        # Persistent accumulator for this request. A server present in
        # the dict has already handed over its final hit count and must
        # not be polled again; absence means "not yet observed".
        per_server = self._per_server_hits.setdefault(request_id, {})
        unresolved_urls = [u for u in self._server_urls if u not in per_server]

        futures = self._lookup_status.setdefault(request_id, {})
        for url in unresolved_urls:
            # Keep the same future until its reply arrives; never restart a poll.
            if url not in futures:
                futures[url] = (
                    self.req_clients[url].query_prefetch_status(request_id),
                    time.monotonic(),
                )

        for url, (fut, submitted_at) in list(futures.items()):
            if self._nonblocking_lookup_status and not fut.query():
                if time.monotonic() - submitted_at >= self._mq_timeout:
                    self._mark_lookup_timed_out(url)
                    return 0
                continue
            del futures[url]
            try:
                r = fut.result(
                    timeout=0 if self._nonblocking_lookup_status else self._mq_timeout
                )
            except TimeoutError:
                logger.warning(
                    "QUERY_PREFETCH_STATUS to %s timed out. Marking unhealthy.",
                    url,
                )
                self._health_events[url].clear()
                return 0
            if r is None:
                continue
            per_server[url] = int(r)

        if len(per_server) < len(self._server_urls):
            return None

        min_chunks = min(per_server.values())
        max_chunks = max(per_server.values())
        if min_chunks != max_chunks:
            logger.warning(
                "[req=%s] LMCache hit mismatch across servers: %s → take min=%d",
                request_id,
                dict(per_server),
                min_chunks,
            )
            self._free_inconsistent_lookup_locks(request_id, per_server, min_chunks)

        token_count = min_chunks * self.lmcache_tokens_per_chunk

        wait_emit("lookup_result", request_id, hit_tokens=token_count)
        self._finished_lookup_results[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Discard consumed lookup results while preserving request ownership.

        Args:
            request_id: Request whose result has been consumed or abandoned.
        Notes:
            Native allocation calls this before worker RETRIEVE. It must not
            retire the server lease. Request end/shutdown separately transfers
            ownership to the confirmed cleanup owner, even after these result
            records have been removed and even while the backend is unhealthy.
        """
        self._pending_lookups.discard(request_id)
        self._unacked_lookups.pop(request_id, None)
        self._lookup_status.pop(request_id, None)
        self._finished_lookup_results.pop(request_id, None)
        self._per_server_hits.pop(request_id, None)
        self._lookup_params.pop(request_id, None)

    def shutdown(self) -> None:
        """Drain confirmed logical cleanup before closing transport clients.

        Raises:
            RuntimeError: If hybrid cleanup cannot be confirmed. Clients and
                the cleanup owner remain live; backend restart is required.
        """
        with self._heartbeat_lock:
            for hb in self._heartbeats.values():
                hb.stop()
        if self._session_cleanup is not None:
            for request_id in list(self._owned_lookup_ids):
                self.end_session(request_id)
            self._session_cleanup.close(timeout=max(1.0, min(self._mq_timeout, 5.0)))
        for client in self.req_clients.values():
            client.close()

    def cleanup_status(self) -> dict[str, Any]:
        """Return confirmed hybrid ownership separately from lookup-result state."""
        if self._session_cleanup is None:
            return {"enabled": False, "pending": 0}
        return {"enabled": True, **self._session_cleanup.report_status()}

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks that
        vLLM has already computed, so they will never be retrieved.  Calling
        this prevents those chunks from holding read locks until TTL expiry.

        Or use this when a request is cancelled or aborted after lookup but
        before retrieve to avoid holding read locks until TTL expiry.

        When ``start`` or ``end`` is not aligned to the chunk size, the
        entire chunk containing start boundary is freed but not end boundary.
        It is caller's responsibility to properly align the boundaries.

        Args:
            token_ids: Token IDs for the key (same as used in lookup).
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
        """
        if not self.is_healthy:
            return

        # Free [start, end) on every server.
        base_key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        ).no_worker_id_version()
        for url in self._server_urls:
            self.req_clients[url].free_lookup_locks(base_key, self.tp_size)

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.

        Hybrid mode transfers cleanup to a generation-bound background owner;
        unhealthy state and late replies cannot discard that obligation. The
        original server must confirm END before it is retired. No DMA pin is
        released by this control path.

        For legacy unacked lookup results, this function will block waiting until
        they are acked. Already-submitted status requests are also drained
        before END_SESSION so their handlers cannot recreate the session.

        Args:
            request_id: The ID of the finished request.
        """
        if self._session_cleanup is not None:
            if request_id in self._owned_lookup_ids:
                self._session_cleanup.retire(request_id)
                self._owned_lookup_ids.remove(request_id)
            return

        if not self.is_healthy:
            return

        ack = self._unacked_lookups.pop(request_id, None)
        if ack is not None:
            remaining = max(
                0.0, self._mq_timeout - (time.monotonic() - ack.submitted_at)
            )
            for url, fut in ack.futures.items():
                try:
                    fut.result(timeout=remaining)
                except TimeoutError:
                    self._mark_lookup_timed_out(url)
                    return

        # Status polling used to complete inside check_lookup_result(). Keep
        # its ordering before END_SESSION when a request finishes early.
        for url, (future, submitted_at) in self._lookup_status.pop(
            request_id, {}
        ).items():
            remaining = max(0.0, self._mq_timeout - (time.monotonic() - submitted_at))
            if not future.wait(timeout=remaining):
                self._mark_lookup_timed_out(url)
                return

        for url in self._server_urls:
            self.req_clients[url].end_session(request_id)

    def report_block_allocations(
        self,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Report vLLM GPU block allocation deltas to LMCache server.

        Fire-and-forget: does not wait for a response. If the server
        is unhealthy the report is silently dropped.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        if not self.is_healthy or not records:
            return

        for url in self._server_urls:
            self.req_clients[url].report_block_allocation(
                os.getpid(), self.model_name, records
            )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheServerKey(
            num_kv_readers=self.parallel_strategy.num_kv_readers,
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )

    def update_pending_store_count(self, req_id: str, count: int) -> bool:
        """
        Update the pending store count for a request.

        Args:
            req_id: The request ID.
            count: The number of workers that have finished the request.

        Returns:
            True if the store request is finished, False otherwise.
        """
        total = self._store_request_pending_counts.get(req_id, 0) + count
        if total >= self._expected_worker_count:
            self._store_request_pending_counts.pop(req_id, None)
            return True
        else:
            self._store_request_pending_counts[req_id] = total
            return False

    def _mark_lookup_timed_out(self, url: str) -> None:
        """Log and mark ``url`` unhealthy after an unacknowledged LOOKUP."""
        logger.warning(
            "LOOKUP to %s timed out after %ss. Marking server as unhealthy.",
            url,
            self._mq_timeout,
        )
        self._health_events[url].clear()


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        extra_config: dict[str, Any] | None = None,
        kv_load_failure_policy: str = "recompute",
    ):
        """Initialize the worker adapter for current or legacy vLLM callers.

        Args:
            server_url: LMCache multiprocess request server URL.
            context: The ZMQ context.
            model_name: The model name used for LMCache keys.
            vllm_block_size: The block size used in vLLM, or legacy KV world
                size when ``parallel_strategy`` is an int.
            parallel_strategy: Current ``ParallelStrategy`` metadata, or the
                legacy KV worker id from older vLLM connectors.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
                Ignored when ``extra_config`` is provided.
            heartbeat_interval: Interval in seconds between heartbeat pings.
                Ignored when ``extra_config`` is provided.
            extra_config: Optional dict with keys starting with
                ``lmcache.mp.`` (e.g., ``lmcache.mp.mq_timeout``). When
                provided, it overrides ``mq_timeout`` / ``heartbeat_interval``.
            kv_load_failure_policy: vLLM's configured action on failed loads.
                ``fail`` releases the request on receive completion; ``recompute``
                keeps it alive until inference and any stores finish.

        Raises:
            TypeError: If the connector argument shape is unsupported.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
            # Only treat ``mp_transfer_mode`` as an explicit override when
            # the user actually set it in extra_config; otherwise leave it
            # as ``None`` so ``create_transfer_context`` can still consult
            # the ``LMCACHE_MP_TRANSFER_MODE`` env var.
            mp_mode_key = (
                _EXTRA_CONFIG_KEY_PREFIX + ExtraConfigDefault.mp_transfer_mode.name
            )
            if mp_mode_key in extra_config:
                self._mp_transfer_mode = cfg[ExtraConfigDefault.mp_transfer_mode.name]
            else:
                self._mp_transfer_mode = None
            set_isolated_ipc(cfg[ExtraConfigDefault.isolated_ipc.name])
            set_use_vmm_api(cfg[ExtraConfigDefault.use_vmm_api.name])
        else:
            self._mp_transfer_mode = None
        self.require_terminal_completion = (
            cfg[ExtraConfigDefault.require_terminal_completion.name]
            if extra_config is not None
            else ExtraConfigDefault.require_terminal_completion.default
        )
        self.req_client = RequestClientFactory.create(server_url, context=context)
        self._mq_timeout = mq_timeout

        # Instance id for GPU worker. uuid4-derived (OS entropy) rather
        # than os.getpid() to avoid collision in containerized deployments.
        # Masked to 63 bits to stay signed-int64-safe for any msgpack peer.
        self.instance_id: int = uuid.uuid4().int & ((1 << 63) - 1)
        logger.info(
            "LMCache MP worker adapter created with instance_id=%d",
            self.instance_id,
        )

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._layout_hints: "LayoutHints | None" = None
        self.engine_group_infos: list[EngineGroupInfo] = []
        # Transport context for transfer operations.
        self.transfer_ctx: TransferContext | None = None

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        self.page_ownership = PendingKVPageAccess()
        # Both experiment arms use this code and the same scope observation.
        # The conservative full-page behavior remains the default.
        self.load_page_scope = os.environ.get("LMCACHE_LOAD_PAGE_SCOPE", "full")
        if self.load_page_scope not in ("full", "written"):
            raise ValueError("LMCACHE_LOAD_PAGE_SCOPE must be full or written")
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}
        # The IPC handle is not enough by itself; CUDA needs the exporting
        # event object to stay alive until the consumer is done with it.
        self.store_events: dict[str, _IpcEvent] = {}
        self.retrieve_events: dict[str, _IpcEvent] = {}
        self.promotion_completions = PromotionCompletions()
        self._uncertain_promotion: tuple[_IpcEvent, LoadStoreOp] | None = None

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # Retrieve request ids dropped by the unhealthy early-return of
        # submit_retrieve_request. get_finished must still report each id
        # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
        self._dropped_retrieves: set[str] = set()
        self.kv_load_failure_policy = kv_load_failure_policy
        # Engine cancellation does not cancel an already submitted GPU copy.
        # Keep its blocks until receive completion, then release exactly once.
        self._finished_during_receive: set[str] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs whose final release was already reported to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            lmcache_tokens_per_chunk = get_lmcache_chunk_size(
                self.req_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.req_client.close()
            _raise_server_unreachable(server_url, self._mq_timeout)
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        if lmcache_tokens_per_chunk % vllm_block_size != 0:
            raise ValueError(
                f"LMCache chunk size {lmcache_tokens_per_chunk} must be a "
                f"multiple of {vllm_block_size} (the vLLM block size scaled by "
                f"decode_context_parallel_size). Set --chunk-size to a multiple "
                f"of {vllm_block_size}."
            )
        self.blocks_in_chunk = lmcache_tokens_per_chunk // vllm_block_size

        # Experimental intermediate tensor transfer
        self.experimental: set[str] = get_experimental(
            self.req_client, timeout=self._mq_timeout
        )
        self.dispatcher: "Dispatcher | None" = None

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first store or retrieve
        # request, by which time vLLM is fully ready (model loaded,
        # KV caches allocated, warmup & CUDA graph capture done).
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()
        if 3 * heartbeat_interval > _SERVER_REAP_TIMEOUT_FLOOR_SECONDS:
            logger.warning(
                "lmcache.mp.heartbeat_interval is %.1fs, so 3 x "
                "heartbeat_interval (%.1fs) exceeds the MP server's "
                "default worker reap timeout floor (%.1fs). Raise the "
                "server's worker reap timeout to at least 3 x the "
                "heartbeat interval, or the server may reap this "
                "worker between heartbeats.",
                heartbeat_interval,
                3 * heartbeat_interval,
                _SERVER_REAP_TIMEOUT_FLOOR_SECONDS,
            )

        # request telemetry, used for prefill-decode disagg
        # TODO: pass down the configuration via vLLM connector config
        # instead of env var
        self.request_telemetry = RequestTelemetryFactory.create(
            telemetry_type=os.getenv("LMCACHE_REQUEST_TELEMETRY_TYPE", "noop"),
            config={
                "endpoint": os.getenv(
                    "LMCACHE_REQUEST_TELEMETRY_ENDPOINT",
                    "http://localhost:5768/api/v1/telemetry",
                ),
            },
        )

        self.lazy_offload = (
            extra_config.get("lmcache.mp.lazy_offload", False)
            if extra_config
            else False
        )

        # Completed store requests to report via build_connector_worker_meta
        self._completed_store_requests: dict[str, int] = {}

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy.

        Reflects the most recent heartbeat result. KV cache
        re-registration on the unhealthy->healthy transition is handled
        by the heartbeat thread itself via ``register_recover_callback``,
        so this property only reads the shared event.
        """
        return self._health_event.is_set()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """Get the kv worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def is_kv_writer(self) -> bool:
        """Whether this worker is responsible for storing KV."""
        return self.parallel_strategy.is_kv_writer

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        layout_hints: "LayoutHints | None" = None,
    ) -> None:
        """
        Register the kv caches with LMCache server.

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            layout_hints: Engine layout hints forwarded to the server; built
                here from the ambient vLLM config when not provided.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
            ValueError: if the LMCache chunk size is not a multiple of an
                engine group's ``tokens_per_block`` (chunk boundaries would
                not align with that group's paged-chunk boundaries).
        """
        logger.info("Registering kv caches")
        for info in engine_group_infos:
            if (
                info.tokens_per_block > 0
                and self.lmcache_tokens_per_chunk % info.tokens_per_block
            ):
                raise ValueError(
                    f"LMCache chunk size {self.lmcache_tokens_per_chunk} must be a "
                    f"multiple of engine group {info.engine_group_id} "
                    f"tokens_per_block {info.tokens_per_block}"
                )
        self.kv_caches = kv_caches
        self.engine_group_infos = list(engine_group_infos)
        # Reused when heartbeat recovery re-registers.
        self._layout_hints = (
            layout_hints if layout_hints is not None else vllm_layout_hints()
        )
        self._send_register_kv_caches_request(kv_caches)

    def load_write_block_ids(
        self, op: LoadStoreOp
    ) -> tuple[dict[int, Sequence[int]], dict[str, Any]]:
        """Return proven LOAD write pages without changing transfer coordinates.

        Args:
            op: Original engine-group block lists already sliced to [start, end).
                skip_first_n_tokens is relative to that range, not token zero.

        Returns:
            Per-engine-group pages that may be written, and observation reasons.
            Unknown geometry/layout retains every original page. Only full
            skipped blocks of the pinned dense BF16 CUDA path are excluded;
            the kernel can overwrite the entire partially skipped block.

        The result is only an access footprint. It must never replace op's
        block lists passed to transfer, error handling or lifetime management.
        """
        full = dict(enumerate(op.block_ids))
        details: dict[str, Any] = {"reason": "dense_full_attention", "groups": {}}
        if (
            not self.require_terminal_completion
            or type(self.transfer_ctx) is not LMCacheDrivenTransferContext
            or self.parallel_strategy.vllm_world_size != 1
            or self.parallel_strategy.dcp_size != 1
            or self.parallel_strategy.mla_only
        ):
            details["reason"] = "unsupported_execution_path"
            return full, details
        skip = op.skip_first_n_tokens
        length = op.end - op.start
        if op.start < 0 or length < 0 or not 0 <= skip <= length:
            details["reason"] = "invalid_restore_range"
            return full, details
        tensors = list(self.kv_caches.values())
        result = dict(full)
        for group_id, blocks in full.items():
            infos = [info for info in self.engine_group_infos
                     if info.engine_group_id == group_id]
            group: dict[str, Any] = {"reason": "unknown_geometry", "skip_full_blocks": 0}
            details["groups"][group_id] = group
            if not infos:
                continue
            sizes = {info.tokens_per_block for info in infos}
            if len(sizes) != 1 or next(iter(sizes)) <= 0:
                continue
            block_tokens = next(iter(sizes))
            group["tokens_per_block"] = block_tokens
            if any(info.sw_size_tokens != -1 or info.recurrent_state
                   or info.extra_object_group_tag != 0 for info in infos):
                group["reason"] = "unsupported_attention_group"
                continue
            if (op.start % block_tokens or op.end % block_tokens
                    or self.lmcache_tokens_per_chunk % block_tokens
                    or len(blocks) != length // block_tokens):
                group["reason"] = "unaligned_or_inconsistent_range"
                continue
            indices = {index for info in infos for index in info.layer_indices}
            if not indices or any(index < 0 or index >= len(tensors) for index in indices):
                group["reason"] = "unknown_layer_geometry"
                continue
            # The frozen Qwen path registers per-layer [NB, BS, NH, CS].
            # Requiring physical BS == logical tokens_per_block excludes
            # compressed/recurrent layouts and avoids guessing from model names.
            try:
                supported_layout = all(
                    tensors[index].dtype == torch.bfloat16
                    and tensors[index].ndim == 4
                    and tensors[index].shape[1] == block_tokens
                    and tensors[index].is_contiguous() for index in indices
                )
            except (AttributeError, IndexError, TypeError):
                supported_layout = False
            if not supported_layout:
                group["reason"] = "unsupported_physical_layout"
                continue
            skipped_blocks = skip // block_tokens
            result[group_id] = blocks[skipped_blocks:]
            group["skip_full_blocks"] = skipped_blocks
            group["reason"] = "full_skipped_blocks_only"
        return result, details

    def _block_ids_per_group(self, op: LoadStoreOp) -> list[list[int]]:
        return expand_engine_block_ids(self.engine_group_infos, op.block_ids)

    def _send_register_kv_caches_request(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Submit a REGISTER_KV_CACHE request and wait for the response.

        Shared by the public ``register_kv_caches`` entry point and the
        heartbeat recovery path (``_reregister_kv_caches_callback``).

        Args:
            kv_caches: The KV cache dict to register.

        Raises:
            ConnectionError: if the server does not respond within
                mq_timeout.
        """
        self.kv_caches = kv_caches
        transfer_ctx = create_transfer_context(kv_caches, mode=self._mp_transfer_mode)
        layout_hints = self._layout_hints
        self.transfer_ctx = transfer_ctx
        try:
            # Register on the local, not self.transfer_ctx: a concurrent
            # shutdown() may null self.transfer_ctx between publish and this
            # call. The local is always non-None.
            transfer_ctx.register(
                self.instance_id,
                kv_caches,
                self.model_name,
                self.world_size,
                self.blocks_in_chunk,
                self.req_client,
                self._mq_timeout,
                layout_hints=layout_hints,
                engine_group_infos=self.engine_group_infos,
                engine_type=EngineType.VLLM,
            )
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first store/retrieve.

        The heartbeat starts healthy (the event was set at construction). A
        live worker pings every interval, refreshing its server-side
        ``last_seen``, so it is never reaped while alive -- no re-registration
        is needed at startup, and the first store/retrieve is not gated. The
        recover callback still re-registers on a genuine unhealthy->healthy
        edge (server restart).
        """
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            heartbeat = HeartbeatThread(
                req_client=self.req_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
                instance_id=self.instance_id,
            )
            heartbeat.register_recover_callback(self._reregister_kv_caches_callback)
            heartbeat.start()
            self._heartbeat = heartbeat

    def _heartbeat_stop_requested(self) -> bool:
        """Whether a created heartbeat thread has a stop requested.

        Returns:
            ``True`` only if a heartbeat exists and its stop was requested.
        """
        heartbeat = self._heartbeat
        return heartbeat is not None and heartbeat.stop_requested

    def _reregister_kv_caches_callback(self) -> bool:
        """Heartbeat recover callback: re-register KV caches after the
        server returns. Runs on the heartbeat thread, before the health
        event is set.

        Returns:
            ``True`` if nothing needs re-registering or registration
            succeeds; ``False`` on failure or a requested heartbeat stop
            (event stays cleared; retried on the next successful PING).
        """
        if self.require_terminal_completion and (
            self.store_futures
            or self.retrieve_futures
            or self.promotion_completions.pending
            or self._uncertain_promotion is not None
        ):
            # Retain the exporting context while a previous server may still
            # own its IPC mappings. A lost terminal response requires restart.
            return False
        if not self.kv_caches:
            # Nothing was registered yet (server flapped before the
            # very first register_kv_caches). Treat as success so the
            # health event can be set.
            return True

        # Skip the rebuild if a shutdown already requested the heartbeat stop.
        if self._heartbeat_stop_requested():
            logger.info("Heartbeat stop requested; skipping KV cache re-registration")
            return False

        try:
            self._send_register_kv_caches_request(self.kv_caches)
        except ConnectionError:
            logger.exception(
                "Failed to re-register KV caches after server recovery; "
                "will retry on next heartbeat"
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected error during KV cache re-registration; "
                "will retry on next heartbeat"
            )
            return False
        logger.warning("Finished re-registering KV caches after server recovery")

        if self.dispatcher is not None:
            if not self.dispatcher.reregister():
                logger.warning(
                    "Failed to re-register dispatcher after server recovery; "
                    "will retry on next heartbeat"
                )
                return False

        return True

    @observed_worker_call("worker_producer_event")
    def create_recorded_event(self) -> _IpcEvent | None:
        """Create the ordering event required by the active transfer context.

        Returns:
            A recorded event for contexts that need stream ordering, or
            ``None`` for synchronous engine-driven transfer or while the
            adapter is unhealthy and requests will be dropped.

        Raises:
            RuntimeError: If called before ``register_kv_caches()``.
        """
        transfer_ctx = self.transfer_ctx
        if transfer_ctx is None:
            raise RuntimeError(
                "KV caches are not registered. Call register_kv_caches() "
                "before creating transfer events."
            )
        if not self.is_healthy:
            return None
        producer_before_ns = time.monotonic_ns()
        event = transfer_ctx.create_recorded_event()
        wait_emit("producer_event_recorded",
                  producer_event_identity=str(id(event)) if event is not None else None,
                  producer_record_before_ns=producer_before_ns,
                  transfer_context=type(transfer_ctx).__name__,
                  semantics="native record call returned; not GPU event completion")
        return event

    @_lmcache_nvtx_annotate
    @observed_worker_call("worker_store_submit")
    @observed_retirement_call("worker_store")
    def submit_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent | None,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The device event recorded after the current model inference
                step, or ``None`` for synchronous engine-driven transfer.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
        """
        self._ensure_heartbeat_started()

        if not self.is_kv_writer:
            return

        if not self.is_healthy:
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting store requests."
            )
        with wait_span("store_source_fence", request_id):
            self.wait_for_kv_access(dict(enumerate(op.block_ids)), {})
        future = self.transfer_ctx.submit_store(
            request_id,
            key,
            self.instance_id,
            self.kv_caches,
            self._block_ids_per_group(op),
            event,
            self.blocks_in_chunk,
        )
        self.store_futures[request_id] = future
        if self.require_terminal_completion:
            self.page_ownership.record(
                future, dict(enumerate(op.block_ids)), writes=False,
                request_id=request_id, operation="STORE"
            )
        if event is not None:
            self.store_events[request_id] = event

    def submit_promotion(self, command: PromotionCommand, event: _IpcEvent) -> None:
        """Submit post-forward LOAD under independent physical-page ownership.

        Args:
            command: TP1/dense-group control from scheduler-owned reservations.
            event: Live producer event recorded after this worker's forward.
        Raises:
            ValueError: Invalid geometry, unsupported mode or duplicate control.
            Exception: Ambiguous submission failure; do not release the targets.
        """
        if (
            not self.require_terminal_completion
            or self.parallel_strategy.vllm_world_size != 1
            or command.chunk_size != self.lmcache_tokens_per_chunk
            or command.block_size <= 0
            or command.block_size * self.blocks_in_chunk != command.chunk_size
            or command.chunk_size % command.block_size
            or len(command.token_ids) != len(command.block_ids) * command.block_size
            or len(command.token_ids) % command.chunk_size
            or not 0 <= command.num_cached_blocks < len(command.block_ids)
            or len(set(command.block_ids)) != len(command.block_ids)
            or any(block <= 0 for block in command.block_ids)
            or self.promotion_completions.contains(command.control_id)
            or self._uncertain_promotion is not None
        ):
            raise ValueError("unsupported or duplicate promotion command")
        self._ensure_heartbeat_started()
        if not self.is_healthy:
            self.promotion_completions.record(command.control_id, None, None)
            return
        if self.transfer_ctx is None:
            raise RuntimeError("promotion requires a registered transfer context")
        cached_tokens = command.num_cached_blocks * command.block_size
        start = cached_tokens // command.chunk_size * command.chunk_size
        op = LoadStoreOp(
            token_ids=list(command.token_ids),
            block_ids=[list(command.block_ids[start // command.block_size :])],
            start=start,
            end=len(command.token_ids),
            skip_first_n_tokens=cached_tokens - start,
        )
        targets = {0: command.block_ids[command.num_cached_blocks :]}
        self.wait_for_kv_access({}, targets)
        key = self._create_key(
            op.token_ids, op.start, op.end, request_id=command.control_id
        )
        self._uncertain_promotion = (event, op)
        future = self.transfer_ctx.submit_retrieve(
            command.control_id,
            key,
            self.instance_id,
            self.kv_caches,
            self._block_ids_per_group(op),
            event,
            self.blocks_in_chunk,
            skip_first_n_tokens=op.skip_first_n_tokens,
        )
        self.page_ownership.record(future, targets, writes=True,
                                   request_id=command.control_id, operation="PROMOTION")
        self.promotion_completions.record(command.control_id, future, (event, op))
        self._uncertain_promotion = None

    def get_promotion_receipts(self) -> tuple[PromotionReceipt, ...]:
        """Poll independent terminal results without native request completion IDs.

        Returns:
            Successful/failed terminal receipts, at most once per control.
        Raises:
            Exception: Ambiguous completion failure, retaining future ownership.
        """
        return self.promotion_completions.poll()

    @_lmcache_nvtx_annotate
    @observed_worker_call("worker_load_submit")
    @observed_retirement_call("worker_load")
    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent | None,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """
        Submit a KV cache retrieve request to LMCache

        When the server is unhealthy the request is not submitted: blocks
        are flagged via ``error_block_ids`` (vLLM recomputes) and the id is
        recorded so ``get_finished`` still reports it exactly once.

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The device event recorded after the current model inference
                step, or ``None`` for synchronous engine-driven transfer.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.flat_block_ids)
            self._dropped_retrieves.add(request_id)
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting retrieve requests."
            )
        wait_emit(
            "restore_trigger",
            request_id,
            producer_event_identity=str(id(event)) if event is not None else None,
            producer_dependency="native_stream_event" if event is not None else "none",
            start_token=op.start,
            end_token=op.end,
            skip_tokens=op.skip_first_n_tokens,
            block_ids=op.block_ids,
        )
        full_blocks = dict(enumerate(op.block_ids))
        write_blocks, write_scope = self.load_write_block_ids(op)
        protected_blocks = (
            write_blocks if self.load_page_scope == "written" else full_blocks
        )
        wait_emit(
            "load_page_scope", request_id,
            mode=self.load_page_scope,
            start_token=op.start, end_token=op.end,
            skip_tokens=op.skip_first_n_tokens,
            full_block_ids=full_blocks, write_block_ids=write_blocks,
            protected_block_ids=protected_blocks, **write_scope,
        )
        with wait_span("target_store_fence", request_id):
            self.wait_for_kv_access({}, protected_blocks)
        wait_emit("restore_submit", request_id)
        future = self.transfer_ctx.submit_retrieve(
            request_id,
            key,
            self.instance_id,
            self.kv_caches,
            self._block_ids_per_group(op),
            event,
            self.blocks_in_chunk,
            skip_first_n_tokens=op.skip_first_n_tokens,
        )
        self.retrieve_futures[request_id] = (future, op.flat_block_ids)
        if self.require_terminal_completion:
            self.page_ownership.record(
                future, protected_blocks, writes=True,
                request_id=request_id, operation="LOAD"
            )
        if event is not None:
            self.retrieve_events[request_id] = event

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: _IpcEvent | None,
        cache_salts: list[str] | None = None,
        request_configs_list: list[dict[str, Any] | None] | None = None,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The device event recorded after the current model inference
                step, or ``None`` for synchronous engine-driven transfer.
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be the same as
                request_ids.
            request_configs_list: Optional LMCache request configs, one per
                request.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        if request_configs_list is None:
            request_configs_list = [None] * len(request_ids)
        if not (
            len(request_ids)
            == len(ops)
            == len(cache_salts)
            == len(request_configs_list)
        ):
            raise ValueError(
                "request_ids, ops, cache_salts, and request_configs_list "
                "must have the same length"
            )
        for request_id, op, salt, request_configs in zip(
            request_ids, ops, cache_salts, request_configs_list, strict=True
        ):
            self.submit_store_request(
                request_id,
                op,
                event,
                cache_salt=salt,
                request_configs=request_configs,
            )

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: _IpcEvent | None,
        cache_salts: list[str] | None = None,
        request_configs_list: list[dict[str, Any] | None] | None = None,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The device event recorded after the current model inference
                step, or ``None`` for synchronous engine-driven transfer.
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be same as
                request_ids.
            request_configs_list: Optional LMCache request configs, one per
                request.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        if request_configs_list is None:
            request_configs_list = [None] * len(request_ids)
        if not (
            len(request_ids)
            == len(ops)
            == len(cache_salts)
            == len(request_configs_list)
        ):
            raise ValueError(
                "request_ids, ops, cache_salts, and request_configs_list "
                "must have the same length"
            )
        for request_id, op, salt, request_configs in zip(
            request_ids, ops, cache_salts, request_configs_list, strict=True
        ):
            self.submit_retrieve_request(
                request_id,
                op,
                event,
                cache_salt=salt,
                request_configs=request_configs,
            )

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(
            finished_req_ids_from_lmcache - self._returned_finished
        )
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if (
                req_id in self._returned_finished
                or req_id in self._finished_during_receive
            ):
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
    @observed_worker_call("worker_get_finished")
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids,
                including retrieves dropped at submit time while unhealthy
                (reported exactly once; blocks already in error_block_ids).

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        if self.dispatcher is not None:
            dispatch(self.dispatcher, "reclaim")

        if self.require_terminal_completion:
            self.page_ownership.reap()
        self._finished_during_receive.update(
            finished_req_ids_from_engine
            & (self.retrieve_futures.keys() | self._dropped_retrieves)
        )

        # Legacy best-effort recovery does not prove DMA completion. Hybrid
        # mode keeps polling the actual futures even after heartbeat loss.
        if not self.is_healthy and not self.require_terminal_completion:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()
            self.store_events.clear()
            self.retrieve_events.clear()

            # Retrieves dropped at submit time still must be reported,
            # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
            # Swap-drain (not update-then-clear): a concurrent
            # submit_retrieve_request add lands in the old set (reported now)
            # or the fresh set (reported next call), never lost.
            dropped = self._dropped_retrieves
            self._dropped_retrieves = set()
            finished_retrieves.update(dropped)

            self._record_receive_releases(finished_retrieves, finished_retrieves)

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            # A request may have a pending retrieve AND appear in
            # finished_req_ids_from_engine (it ran without loading KV after
            # the server died).  The scheduler processes finished_recving
            # first and deletes the request, so we must not also report it
            # in finished_sending.
            ret_stores -= finished_retrieves
            return ret_stores, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        failed_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result(timeout=60)
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, r_block_ids) in self.retrieve_futures.items():
            wait_emit("worker_restore_poll", request_id)
            if not r_future.query():
                continue

            wait_emit("worker_restore_observed", request_id)
            r_result = r_future.result(timeout=60)
            terminal = request_id in self._finished_during_receive or (
                not r_result and self.kv_load_failure_policy == "fail"
            )
            if (
                terminal
                and request_id in self.store_futures
                and request_id not in finished_stores
            ):
                # A final receive notification frees all request blocks, so it
                # must also wait for any outstanding store reading those blocks.
                continue
            finished_retrieves.add(request_id)

            if not r_result:
                failed_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
            self.store_events.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
            self.retrieve_events.pop(request_id, None)

        # Retrieves dropped while unhealthy still must be reported exactly
        # once. Deduplicate final receive releases for fail/cancel paths too.
        # Swap-drain so a concurrent submit_retrieve_request add is never lost.
        dropped = self._dropped_retrieves
        self._dropped_retrieves = set()
        finished_retrieves.update(dropped)
        failed_retrieves.update(dropped)
        for wait_rid in finished_retrieves:
            wait_emit(
                "worker_recv_ready", wait_rid, failed=wait_rid in failed_retrieves
            )
        self._record_receive_releases(finished_retrieves, failed_retrieves)

        # Update the internal states
        ret_stores = self._process_finished_stores(
            finished_stores, finished_req_ids_from_engine
        )

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if ret_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=ret_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        return ret_stores, finished_retrieves

    # TODO(chunxiaozheng): There are some duplicated codes
    #  with `get_finished`, optimize it later.
    @_lmcache_nvtx_annotate
    def get_finished_with_lazy_offload(
        self,
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests in lazy offload mode.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids.
                In lazy offload mode, return None directly.
            - The second set contains the finished retrieve request ids,
                including retrieves dropped at submit time while unhealthy
                (reported exactly once; blocks already in error_block_ids).
        """
        if not self.lazy_offload:
            raise ValueError(
                "get_finished_with_lazy_offload is only available in lazy offload mode"
            )

        if self.require_terminal_completion:
            self.page_ownership.reap()

        if self.dispatcher is not None:
            dispatch(self.dispatcher, "reclaim")

        # Hybrid callers must retain submitted work until its actual fence.
        if not self.is_healthy and not self.require_terminal_completion:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()
            self.store_events.clear()
            self.retrieve_events.clear()

            # Retrieves dropped at submit time still must be reported,
            # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
            # Swap-drain (not update-then-clear): a concurrent
            # submit_retrieve_request add lands in the old set (reported now)
            # or the fresh set (reported next call), never lost.
            dropped = self._dropped_retrieves
            self._dropped_retrieves = set()
            finished_retrieves.update(dropped)

            for req_id in finished_stores:
                self._completed_store_requests[req_id] = 1
            return None, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result(timeout=60)
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, r_block_ids) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result(timeout=60)
            finished_retrieves.add(request_id)

            if not r_result:
                self.error_block_ids.update(r_block_ids)
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
            self.store_events.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
            self.retrieve_events.pop(request_id, None)

        # Retrieves dropped while unhealthy still must be reported,
        # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS. No
        # finished_sending dedup is needed (unlike the unhealthy branch): a
        # dropped retrieve's request is parked in WAITING_FOR_REMOTE_KVS until
        # this report, so it cannot also be engine-finished in the same call.
        # Swap-drain so a concurrent submit_retrieve_request add is never lost.
        dropped = self._dropped_retrieves
        self._dropped_retrieves = set()
        finished_retrieves.update(dropped)

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if finished_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=finished_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        for req_id in finished_stores:
            self._completed_store_requests[req_id] = 1
        return None, finished_retrieves

    def get_completed_store_requests(self) -> dict[str, int] | None:
        """Return completed store requests since the last call."""
        if not self._completed_store_requests:
            return None
        completed_store_requests = self._completed_store_requests
        self._completed_store_requests = {}
        return completed_store_requests

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Returns the block IDs that failed due to retrieve timeout,
        then clears the internal set.
        """
        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors

    @observed_retirement_call("worker_preempt")
    @observed_worker_call("worker_preempt")
    def handle_preemptions(self, need_flush_before_forward: bool) -> None:
        """Handle worker-side preemption hints from connector metadata.

        When ``need_flush_before_forward`` is true, synchronize deferred
        engine-driven gather work before the next forward pass can overwrite
        paged KV blocks.

        Args:
            need_flush_before_forward: When True, flush in-flight gather
                operations on the transfer context. When False, this is a no-op.
        """
        if not need_flush_before_forward:
            return
        if not self.is_healthy or self.transfer_ctx is None:
            return
        with worker_observation_span("worker_preempt_flush"):
            self.transfer_ctx.flush_inflight_stores()
        # Force device sync here, compare to preemption, perf panelty is trivial
        with worker_observation_span("worker_preempt_device_sync"):
            torch_dev.synchronize()

    def wait_for_kv_access(
        self,
        read_blocks: Mapping[int, Sequence[int]],
        write_blocks: Mapping[int, Sequence[int]],
    ) -> None:
        """Wait for conflicting MP physical accesses, even without a preempt hint.

        Args:
            read_blocks: Actual upcoming reads indexed by engine group.
            write_blocks: Actual upcoming writes indexed by engine group.
        Raises:
            TimeoutError: Remote completion is unavailable; do not access pages.
            Exception: An ambiguous transport failure; protection is retained.
        """
        if self.require_terminal_completion:
            with worker_observation_span("worker_page_ownership_wait"):
                self.page_ownership.reap()
                self.page_ownership.wait(
                    read_blocks, write_blocks, timeout=self._mq_timeout
                )

    def shutdown(self) -> None:
        """
        Shutdown the LMCache MP worker adapter.

        Stops the heartbeat (if started) before UNREGISTER: no new ping
        on the closing request client, and a straggler in-flight cycle cannot
        re-register or flip the health event after unregistration.
        """
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                self._heartbeat.stop()

        logger.info("Unregistering kv caches")
        try:
            if isinstance(self.transfer_ctx, EngineDrivenTransferContext):
                future = self.req_client.unregister_kv_cache_engine_driven_context(
                    self.instance_id
                )
            else:
                future = self.req_client.unregister_kv_cache(self.instance_id)
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            if self.require_terminal_completion and (
                self.store_futures
                or self.retrieve_futures
                or self.promotion_completions.pending
                or self._uncertain_promotion is not None
            ):
                raise RuntimeError(
                    "cannot close hybrid GPU context without a terminal drain reply"
                ) from None
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        if self.dispatcher is not None:
            dispatch(self.dispatcher, "shutdown")

        if self.transfer_ctx is not None:
            self.transfer_ctx.close()
            self.transfer_ctx = None

        self.req_client.close()
        self.request_telemetry.close()

    # Helper functions
    def _record_receive_releases(
        self, finished_retrieves: set[str], failed_retrieves: set[str]
    ) -> None:
        """Deduplicate final releases performed by vLLM's receive handler."""
        terminal = finished_retrieves & self._finished_during_receive
        if self.kv_load_failure_policy == "fail":
            terminal.update(failed_retrieves)
        self._returned_finished.update(terminal)
        self._finished_during_receive.difference_update(terminal)
        self.finished_stores.difference_update(terminal)
        self.previously_finished.difference_update(terminal)

    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey(
            num_kv_readers=self.parallel_strategy.num_kv_readers,
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
