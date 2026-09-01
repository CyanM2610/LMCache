# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Protocol, cast
import enum
import math
import os
import threading
import time
import uuid

# Third Party
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.integration.vllm.experimental import dispatch
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
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.protocols.hotprefix import (
    HotPrefixAccessResponse,
    HotPrefixAdmissionResponse,
    HotPrefixHostCandidate,
    HotPrefixTransferTicket,
    is_hotprefix_promotion_request,
    is_hotprefix_store_request,
)
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.isolated_ipc import set_isolated_ipc

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.experimental import Dispatcher

logger = init_logger(__name__)


@dataclass(frozen=True, eq=False)
class _ExtraConfigDefaultValue:
    """Identity-distinct wrapper for one extra-config default value."""

    default: Any


class ExtraConfigDefault(enum.Enum):
    """Centralized default values for extra_config keys.

    Each member's *name* is the key used in the extra_config dict,
    and its :attr:`default` is the user-facing default. The wrapper values are
    identity-distinct so equal defaults such as two ``False`` booleans do not
    become Enum aliases.
    """

    # Timeout (seconds) for blocking MQ requests: initial
    # chunk-size query, KV cache registration/unregistration,
    # and other synchronous operations.
    mq_timeout = _ExtraConfigDefaultValue(300.0)
    # Interval (seconds) between periodic heartbeat pings
    # to the server.
    heartbeat_interval = _ExtraConfigDefaultValue(10.0)
    # Routing mode for ``create_transfer_context``: ``auto`` keeps the
    # historical CUDA -> lmcache_driven / others -> engine_driven dispatch;
    # ``lmcache_driven`` forces the IPC / SHM zero-copy path where the
    # LMCache server pulls data via device handles;
    # ``engine_driven`` forces the worker-side gather/scatter copy path.
    # Mirrors the ``LMCACHE_MP_TRANSFER_MODE`` env var; this extra_config
    # key wins when both are set.
    mp_transfer_mode = _ExtraConfigDefaultValue("auto")
    # Whether IPC mechanisms must work across isolated containers (no
    # shared host IPC namespace or /dev/shm); see
    # lmcache/v1/platform/isolated_ipc.py. Must match the LMCache server's
    # ``--isolated-ipc`` setting.
    isolated_ipc = _ExtraConfigDefaultValue(False)
    # Enable one fleet-wide HotPrefix access observation per initial lookup.
    hotprefix_enabled = _ExtraConfigDefaultValue(False)
    # Stable scheduler identity. Negative values request a generated identity.
    hotprefix_instance_id = _ExtraConfigDefaultValue(-1)
    # Maximum Host-to-HBM promotion bytes submitted by one scheduler step.
    hotprefix_promotion_budget_bytes = _ExtraConfigDefaultValue(64 << 20)

    @property
    def default(self) -> Any:
        """Return the user-facing default value for this config key."""
        return self.value.default


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
        default: The :class:`ExtraConfigDefault` member default, defining the
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


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
    timeout: float = DEFAULT_MQ_TIMEOUT,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        timeout: Timeout in seconds for the blocking request.

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    lmcache_tokens_per_chunk = future.result(timeout=timeout)
    return lmcache_tokens_per_chunk


def get_experimental(
    mq_client: MessageQueueClient,
    timeout: float = DEFAULT_MQ_TIMEOUT,
) -> set[str]:
    """Query the experimental capabilities a server advertises.

    Args:
        mq_client: The LMCache multiprocess mode message queue client.
        timeout: Seconds to wait for the server's response.

    Returns:
        Experimental features built into the server (now `transfer_query`).
    """
    future = send_lmcache_request(mq_client, RequestType.GET_EXPERIMENTAL, [])
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
    mq_client: MessageQueueClient,
    timeout: float,
    instance_id: int | None = None,
) -> bool:
    """Send a PING request and return the result.

    Args:
        mq_client: The message queue client.
        timeout: Seconds to wait for the server's response.
        instance_id: The worker's instance ID so the server can refresh its
            liveness, or None for an untracked prober (scheduler adapter).

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = send_lmcache_request(mq_client, RequestType.PING, [instance_id])
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
        mq_client: MessageQueueClient,
        health_event: threading.Event,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        instance_id: int | None = None,
    ):
        """
        Args:
            mq_client: The message queue client used to send PING requests.
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
        self._mq_client = mq_client
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
            self._mq_client, timeout=self._interval, instance_id=self._instance_id
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
        hotprefix_namespace_prefix: bytes | None = None,
    ) -> None:
        """
        Args:
            server_urls: The servers URL for the LMCache message queue
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
            hotprefix_namespace_prefix: Canonical model/revision/layout identity
                before the per-request cache salt is appended.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        assert len(server_urls) >= 1, "At least one server url required"
        self._server_urls: list[str] = list(server_urls)
        self.mq_clients: dict[str, MessageQueueClient] = {
            url: MessageQueueClient(url, context) for url in self._server_urls
        }
        cfg = _resolve_extra_config(extra_config)
        mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
        heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self._mq_timeout = mq_timeout

        # Lookup state tracking:
        # - _pending_lookups: request_ids submitted but not yet resolved
        # - _finished_lookup_results: cached chunk count keyed by request_id,
        #   so that repeated calls to check_lookup_result return the same value
        #   even after the server has already popped the job (exactly-once).
        # - _per_server_hits: {request_id: {server_url: hit_chunks}}.
        #   Per-server hit counts, used to detect disagreement and free tail locks.
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}
        self._per_server_hits: dict[str, dict[str, int]] = {}
        self._lookup_params: dict[str, tuple[list[int], str]] = {}
        self._hotprefix_enabled = bool(cfg[ExtraConfigDefault.hotprefix_enabled.name])
        self._hotprefix_promotion_budget_bytes = int(
            cfg[ExtraConfigDefault.hotprefix_promotion_budget_bytes.name]
        )
        if self._hotprefix_promotion_budget_bytes <= 0:
            raise ValueError("hotprefix promotion budget must be positive")
        self._hotprefix_namespace_prefix = (
            hotprefix_namespace_prefix
            if hotprefix_namespace_prefix is not None
            else f"{model_name}\0".encode()
        )
        configured_instance = int(cfg[ExtraConfigDefault.hotprefix_instance_id.name])
        self._hotprefix_instance_id = (
            configured_instance
            if configured_instance >= 0
            else uuid.uuid4().int & ((1 << 63) - 1)
        )
        self._hotprefix_event_seq = 0
        self._hotprefix_request_sequences: dict[str, int] = {}
        self._hotprefix_results: dict[str, dict[str, HotPrefixAccessResponse]] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        chunk_sizes: dict[str, int] = {}
        for url, client in self.mq_clients.items():
            try:
                chunk_sizes[url] = get_lmcache_chunk_size(
                    client, timeout=self._mq_timeout
                )
            except TimeoutError:
                for c in self.mq_clients.values():
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
        self._hotprefix_control_lock = threading.Lock()
        self._hotprefix_control_observations: list[dict[str, int | float | str]] = []

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def hotprefix_enabled(self) -> bool:
        """Whether canonical eviction-triggered HotPrefix mode is active."""
        return self._hotprefix_enabled

    @property
    def hotprefix_promotion_budget_bytes(self) -> int:
        """Return the configured per-step promotion transfer budget."""
        return self._hotprefix_promotion_budget_bytes

    def drain_hotprefix_control_stats(self) -> dict[str, Any]:
        """Return and reset client-observed HotPrefix control RPC deltas."""
        with self._hotprefix_control_lock:
            observations = self._hotprefix_control_observations
            self._hotprefix_control_observations = []
        return {"control_observations": observations}

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
            for url, client in self.mq_clients.items():
                hb = HeartbeatThread(
                    mq_client=client,
                    health_event=self._health_events[url],
                    interval=self._heartbeat_interval,
                )
                hb.start()
                self._heartbeats[url] = hb

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        local_matched_tokens: int = 0,
    ) -> None:
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends a LOOKUP request to the server and blocks until a prefetch
        job ID is returned.  The actual prefetch result can then be polled
        via ``check_lookup_result``.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache
            cache_salt: Per-user isolation salt. Requests with different
                cache_salt values produce separate cache entries.
            local_matched_tokens: Tokens reused by vLLM native APC.

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

        if not self._submit_hotprefix_access(
            request_id,
            token_ids,
            cache_salt,
            local_matched_tokens,
        ):
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
        ).no_worker_id_version()

        futures: dict[str, MessagingFuture[Any]] = {
            url: send_lmcache_request(
                self.mq_clients[url],
                RequestType.LOOKUP,
                [key, self.tp_size],
            )
            for url in self._server_urls
        }

        # Any one server failure means the whole lookup fails.
        for url, fut in futures.items():
            try:
                fut.result(timeout=self._mq_timeout)
            except TimeoutError:
                logger.warning(
                    "LOOKUP to %s timed out after %ss. Marking server as unhealthy.",
                    url,
                    self._mq_timeout,
                )
                self._health_events[url].clear()
                return

        self._pending_lookups.add(request_id)
        self._lookup_params[request_id] = (token_ids, cache_salt)

    def _submit_hotprefix_access(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str,
        local_matched_tokens: int,
    ) -> bool:
        if not self._hotprefix_enabled:
            return True
        if request_id in self._hotprefix_request_sequences:
            return True
        aligned_end = (
            len(token_ids) // self.lmcache_tokens_per_chunk
        ) * self.lmcache_tokens_per_chunk
        if aligned_end == 0:
            self._hotprefix_request_sequences[request_id] = self._hotprefix_event_seq
            return True
        cacheable_token_ids = token_ids[:aligned_end]
        cacheable_matched_tokens = min(local_matched_tokens, aligned_end)
        self._hotprefix_event_seq += 1
        event_seq = self._hotprefix_event_seq
        namespace = self._hotprefix_namespace_prefix + cache_salt.encode()
        started_ns = time.monotonic_ns()
        request_bytes = len(self._server_urls) * (
            24 + len(namespace) + 8 * len(cacheable_token_ids)
        )
        futures: dict[str, MessagingFuture[Any]] = {
            url: send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_ACCESS,
                [
                    self._hotprefix_instance_id,
                    event_seq,
                    namespace,
                    cacheable_token_ids,
                    cacheable_matched_tokens,
                ],
            )
            for url in self._server_urls
        }
        responses: dict[str, HotPrefixAccessResponse] = {}
        for url, future in futures.items():
            try:
                response = future.result(timeout=self._mq_timeout)
            except TimeoutError:
                logger.warning(
                    "HOT_PREFIX_ACCESS to %s timed out after %ss.",
                    url,
                    self._mq_timeout,
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "access", "timeout", started_ns, request_bytes
                )
                return False
            if not isinstance(response, HotPrefixAccessResponse):
                logger.warning(
                    "HOT_PREFIX_ACCESS from %s returned %s", url, type(response)
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "access", "failure", started_ns, request_bytes
                )
                return False
            responses[url] = response
        self._hotprefix_request_sequences[request_id] = event_seq
        self._hotprefix_results[request_id] = responses
        response_bytes = sum(
            16 + sum(len(prefix_id) for prefix_id in response.path)
            for response in responses.values()
        )
        self._record_hotprefix_control(
            "access", "success", started_ns, request_bytes, response_bytes
        )
        return True

    def submit_hotprefix_access(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        local_matched_tokens: int = 0,
    ) -> bool:
        """Submit only the Global ACCESS observation for cost ablations.

        Args:
            request_id: Stable request identifier.
            token_ids: Complete request tokens.
            cache_salt: Per-tenant cache namespace.
            local_matched_tokens: Tokens matched by native APC.

        Returns:
            Whether every configured server accepted the observation.
        """
        self._ensure_heartbeat_started()
        if not self.is_healthy:
            return False
        return self._submit_hotprefix_access(
            request_id,
            token_ids,
            cache_salt,
            local_matched_tokens,
        )

    def prepare_hotprefix_retrieve(
        self,
        request_id: str,
        token_ids: list[int],
        start: int,
        end: int,
        cache_salt: str = "",
    ) -> bool:
        """Acquire standard L1 read locks for one promotion transfer range.

        A HotPrefix generation lease prevents policy replacement, but the
        existing RETRIEVE executor still requires the same per-object read
        locks as an on-demand fetch. Background promotions do not pass through
        ``get_num_new_matched_tokens``, so they must perform this range-scoped
        LOOKUP explicitly before worker metadata is published.

        Returns:
            ``True`` only when every server retained every requested chunk.
            Partial locks and lookup sessions are released on failure.
        """
        self._ensure_heartbeat_started()
        if not self._hotprefix_enabled or not self.is_healthy:
            return False
        if (
            start < 0
            or end <= start
            or end > len(token_ids)
            or start % self.lmcache_tokens_per_chunk != 0
            or end % self.lmcache_tokens_per_chunk != 0
        ):
            raise ValueError("HotPrefix retrieve range must be non-empty and aligned")

        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()
        lookup_futures = {
            url: send_lmcache_request(
                self.mq_clients[url],
                RequestType.LOOKUP,
                [key, self.tp_size],
            )
            for url in self._server_urls
        }
        for url, future in lookup_futures.items():
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                logger.warning(
                    "HotPrefix promotion LOOKUP failed on %s", url, exc_info=True
                )
                self._health_events[url].clear()
                self._cleanup_hotprefix_lookup(key)
                return False

        wait_futures = {
            url: send_lmcache_request(
                self.mq_clients[url],
                RequestType.WAIT_PREFETCH_STATUS,
                [request_id, float(self._mq_timeout)],
            )
            for url in self._server_urls
        }
        expected_chunks = (end - start) // self.lmcache_tokens_per_chunk
        hit_chunks: dict[str, int] = {}
        for url, future in wait_futures.items():
            try:
                response = future.result(timeout=self._mq_timeout + 1.0)
            except Exception:
                logger.warning(
                    "HotPrefix promotion prefetch wait failed on %s",
                    url,
                    exc_info=True,
                )
                self._health_events[url].clear()
                self._cleanup_hotprefix_lookup(key)
                return False
            if type(response) is not int:
                self._cleanup_hotprefix_lookup(key)
                return False
            hit_chunks[url] = response

        if any(count != expected_chunks for count in hit_chunks.values()):
            logger.warning(
                "HotPrefix promotion range retained %s chunks; expected %d on "
                "every server",
                hit_chunks,
                expected_chunks,
            )
            self._cleanup_hotprefix_lookup(key)
            return False
        return True

    def _cleanup_hotprefix_lookup(self, key: IPCCacheServerKey) -> None:
        """Synchronously release a failed promotion lookup and its session."""
        for url in self._server_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.FREE_LOOKUP_LOCKS,
                [key, self.tp_size],
            )
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                self._health_events[url].clear()
        for url in self._server_urls:
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.END_SESSION,
                [key.request_id],
            )

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
        token_ids_l, cs = self._lookup_params.pop(request_id, (None, None))
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
                ).no_worker_id_version()
                send_lmcache_request(
                    self.mq_clients[url],
                    RequestType.FREE_LOOKUP_LOCKS,
                    [tail_key, self.tp_size],
                )

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Sends a QUERY_PREFETCH_STATUS request to the servers and blocks
        until the server responds.  Returns the matched token count
        when the prefetch is complete, or None if still in progress.

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

        # Persistent accumulator for this request. A server present in
        # the dict has already handed over its final hit count and must
        # not be polled again; absence means "not yet observed".
        per_server = self._per_server_hits.setdefault(request_id, {})
        unresolved_urls = [u for u in self._server_urls if u not in per_server]

        futures: dict[str, MessagingFuture[Any]] = {
            url: send_lmcache_request(
                self.mq_clients[url],
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            )
            for url in unresolved_urls
        }

        for url, fut in futures.items():
            try:
                r = fut.result(timeout=self._mq_timeout)
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

        self._finished_lookup_results[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup state for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookup_results.pop(request_id, None)
        self._per_server_hits.pop(request_id, None)
        self._lookup_params.pop(request_id, None)
        self._hotprefix_request_sequences.pop(request_id, None)
        self._hotprefix_results.pop(request_id, None)

    def hotprefix_admit(
        self,
        namespace: bytes,
        prefix_id: bytes,
        size_bytes: int,
    ) -> HotPrefixAdmissionResponse | None:
        """Reserve shared Host capacity for an HBM eviction candidate.

        Args:
            namespace: Canonical LogicalPrefix namespace.
            prefix_id: Actual HBM victim prefix.
            size_bytes: Complete payload size on each server.

        Returns:
            Cross-server consensus response, or ``None`` when unavailable.

        Raises:
            RuntimeError: If MP servers disagree after rollback.
        """
        if not self._hotprefix_enabled or not self.is_healthy:
            return None
        started_ns = time.monotonic_ns()
        request_bytes = len(self._server_urls) * (len(namespace) + len(prefix_id) + 16)
        proposed_generation = uuid.uuid4().int & ((1 << 63) - 1)
        responses: list[HotPrefixAdmissionResponse] = []
        accepted_urls: list[str] = []
        for url in self._server_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_ADMIT,
                [namespace, prefix_id, size_bytes, proposed_generation],
            )
            try:
                response = future.result(timeout=self._mq_timeout)
            except TimeoutError:
                self._rollback_hotprefix_admission(
                    namespace, prefix_id, [*accepted_urls, url]
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "admit", "timeout", started_ns, request_bytes
                )
                return None
            if not isinstance(response, HotPrefixAdmissionResponse):
                self._rollback_hotprefix_admission(
                    namespace, prefix_id, [*accepted_urls, url]
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "admit", "failure", started_ns, request_bytes
                )
                return None
            responses.append(response)
            if response.action == "accept":
                accepted_urls.append(url)
        if not responses:
            self._record_hotprefix_control(
                "admit", "failure", started_ns, request_bytes
            )
            return None
        first = responses[0]
        if any(response != first for response in responses[1:]):
            self._rollback_hotprefix_admission(namespace, prefix_id, accepted_urls)
            self._record_hotprefix_control(
                "admit", "failure", started_ns, request_bytes
            )
            raise RuntimeError("HotPrefix admission disagreed across LMCache servers")
        response_bytes = sum(
            16
            + len(response.action)
            + len(response.reason)
            + sum(len(item) for item in response.evict_prefixes)
            for response in responses
        )
        self._record_hotprefix_control(
            "admit", "success", started_ns, request_bytes, response_bytes
        )
        return first

    def _rollback_hotprefix_admission(
        self,
        namespace: bytes,
        prefix_id: bytes,
        accepted_urls: list[str],
    ) -> None:
        for url in accepted_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_ABORT,
                [namespace, prefix_id],
            )
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                self._health_events[url].clear()

    def hotprefix_candidates(
        self,
        namespace: bytes,
        prefix_ids: list[bytes],
    ) -> list[HotPrefixHostCandidate]:
        """Return shared Host sources that are READY on every MP server.

        A multi-server LMCache deployment may shard a model's KV tensors across
        servers.  A promotion is executable only when each shard can provide the
        same logical residency generation, so this method deliberately returns
        the intersection rather than the union of server responses.

        Args:
            namespace: Canonical LogicalPrefix namespace.
            prefix_ids: Target-local missing prefixes in desired order.

        Returns:
            READY candidates present with identical generation/size everywhere.
        """
        if not self._hotprefix_enabled or not self.is_healthy or not prefix_ids:
            return []
        started_ns = time.monotonic_ns()
        request_bytes = len(self._server_urls) * (
            len(namespace) + sum(len(prefix_id) for prefix_id in prefix_ids)
        )

        responses: list[list[HotPrefixHostCandidate]] = []
        for url in self._server_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_CANDIDATES,
                [namespace, prefix_ids],
            )
            try:
                response = future.result(timeout=self._mq_timeout)
            except TimeoutError:
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "candidates", "timeout", started_ns, request_bytes
                )
                return []
            if not isinstance(response, list) or any(
                not isinstance(item, HotPrefixHostCandidate) for item in response
            ):
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "candidates", "failure", started_ns, request_bytes
                )
                return []
            responses.append(response)

        if not responses:
            self._record_hotprefix_control(
                "candidates", "failure", started_ns, request_bytes
            )
            return []
        common = {
            (item.prefix_id, item.generation, item.size_bytes): item
            for item in responses[0]
        }
        for response in responses[1:]:
            available = {
                (item.prefix_id, item.generation, item.size_bytes) for item in response
            }
            common = {key: item for key, item in common.items() if key in available}

        requested_order = {
            prefix_id: index for index, prefix_id in enumerate(prefix_ids)
        }
        result = sorted(
            common.values(),
            key=lambda item: requested_order.get(item.prefix_id, len(prefix_ids)),
        )
        response_bytes = sum(
            len(item.prefix_id) + 32 for response in responses for item in response
        )
        self._record_hotprefix_control(
            "candidates", "success", started_ns, request_bytes, response_bytes
        )
        return result

    def hotprefix_acquire(
        self,
        namespace: bytes,
        candidate: HotPrefixHostCandidate,
    ) -> HotPrefixTransferTicket | None:
        """Atomically acquire the same generation from every MP server.

        Args:
            namespace: Candidate namespace.
            candidate: READY source selected by the target-local planner.

        Returns:
            A common renewable ticket, or ``None`` after rollback on failure.
        """
        if not self._hotprefix_enabled or not self.is_healthy:
            return None
        started_ns = time.monotonic_ns()
        request_bytes = len(self._server_urls) * (
            len(namespace) + len(candidate.prefix_id) + 24
        )
        ticket_id = uuid.uuid4().bytes
        acquired_urls: list[str] = []
        tickets: list[HotPrefixTransferTicket] = []
        for url in self._server_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_ACQUIRE,
                [
                    namespace,
                    candidate.prefix_id,
                    candidate.generation,
                    ticket_id,
                ],
            )
            try:
                response = future.result(timeout=self._mq_timeout)
            except Exception:
                self._rollback_hotprefix_acquire(
                    namespace, ticket_id, [*acquired_urls, url]
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "acquire", "failure", started_ns, request_bytes
                )
                return None
            if not isinstance(response, HotPrefixTransferTicket):
                self._rollback_hotprefix_acquire(
                    namespace, ticket_id, [*acquired_urls, url]
                )
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    "acquire", "failure", started_ns, request_bytes
                )
                return None
            acquired_urls.append(url)
            tickets.append(response)

        first = tickets[0] if tickets else None
        if first is None or any(ticket != first for ticket in tickets[1:]):
            self._rollback_hotprefix_acquire(namespace, ticket_id, acquired_urls)
            self._record_hotprefix_control(
                "acquire", "failure", started_ns, request_bytes
            )
            return None
        response_bytes = len(tickets) * (
            len(first.ticket_id) + len(first.prefix_id) + 16
        )
        self._record_hotprefix_control(
            "acquire", "success", started_ns, request_bytes, response_bytes
        )
        return first

    def hotprefix_release(
        self,
        namespace: bytes,
        ticket: HotPrefixTransferTicket,
    ) -> bool:
        """Release a promotion ticket on all MP servers.

        Args:
            namespace: Ticket namespace.
            ticket: Terminal promotion ticket.

        Returns:
            ``True`` only when every server acknowledges release.
        """
        return self._hotprefix_finish(
            RequestType.HOT_PREFIX_RELEASE,
            namespace,
            ticket.ticket_id,
        )

    def hotprefix_renew(
        self,
        namespace: bytes,
        ticket: HotPrefixTransferTicket,
    ) -> bool:
        """Renew a promotion ticket on all MP servers.

        Args:
            namespace: Ticket namespace.
            ticket: In-flight promotion ticket.

        Returns:
            ``True`` only when every server extends the lease.
        """
        return self._hotprefix_finish(
            RequestType.HOT_PREFIX_RENEW,
            namespace,
            ticket.ticket_id,
        )

    def _rollback_hotprefix_acquire(
        self,
        namespace: bytes,
        ticket_id: bytes,
        acquired_urls: list[str],
    ) -> None:
        for url in acquired_urls:
            future = send_lmcache_request(
                self.mq_clients[url],
                RequestType.HOT_PREFIX_RELEASE,
                [namespace, ticket_id],
            )
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                self._health_events[url].clear()

    def hotprefix_publish(self, namespace: bytes, prefix_id: bytes) -> bool:
        """Publish a completely stored HotPrefix Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Fully stored LogicalPrefix.

        Returns:
            ``True`` only when every MP server publishes it.
        """
        return self._hotprefix_finish(
            RequestType.HOT_PREFIX_PUBLISH, namespace, prefix_id
        )

    def hotprefix_abort(self, namespace: bytes, prefix_id: bytes) -> bool:
        """Abort an incomplete HotPrefix Host residency.

        Args:
            namespace: Residency namespace.
            prefix_id: Failed LogicalPrefix reservation.

        Returns:
            ``True`` only when every server completes rollback.
        """
        return self._hotprefix_finish(
            RequestType.HOT_PREFIX_ABORT, namespace, prefix_id
        )

    def hotprefix_invalidate(
        self,
        namespace: bytes,
        prefix_id: bytes,
        generation: int,
    ) -> bool:
        """Invalidate a stale physical residency on every MP server.

        Args:
            namespace: Residency namespace.
            prefix_id: LogicalPrefix whose retrieve failed.
            generation: Exact failed generation.

        Returns:
            ``True`` only when every server acknowledges invalidation.
        """
        return self._hotprefix_all_server_bool(
            RequestType.HOT_PREFIX_INVALIDATE,
            [namespace, prefix_id, generation],
        )

    def _hotprefix_finish(
        self, request_type: RequestType, namespace: bytes, prefix_id: bytes
    ) -> bool:
        return self._hotprefix_all_server_bool(
            request_type,
            [namespace, prefix_id],
        )

    def _hotprefix_all_server_bool(
        self,
        request_type: RequestType,
        payloads: list[Any],
    ) -> bool:
        if not self._hotprefix_enabled or not self.is_healthy:
            return False
        started_ns = time.monotonic_ns()
        method = request_type.name.removeprefix("HOT_PREFIX_").lower()
        payload_bytes = sum(
            len(item) if isinstance(item, bytes) else 8 for item in payloads
        )
        request_bytes = len(self._server_urls) * payload_bytes
        for url in self._server_urls:
            future = send_lmcache_request(self.mq_clients[url], request_type, payloads)
            try:
                response = future.result(timeout=self._mq_timeout)
            except TimeoutError:
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    method, "timeout", started_ns, request_bytes
                )
                return False
            if response is not True:
                self._health_events[url].clear()
                self._record_hotprefix_control(
                    method, "failure", started_ns, request_bytes
                )
                return False
        self._record_hotprefix_control(
            method,
            "success",
            started_ns,
            request_bytes,
            len(self._server_urls),
        )
        return True

    def shutdown(self) -> None:
        """Shutdown the scheduler adapter and its resources."""
        for client in self.mq_clients.values():
            client.close()
        with self._heartbeat_lock:
            for hb in self._heartbeats.values():
                hb.stop()

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
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
        ).no_worker_id_version()
        for url in self._server_urls:
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.FREE_LOOKUP_LOCKS,
                [base_key, self.tp_size],
            )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        for url in self._server_urls:
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.END_SESSION,
                [request_id],
            )

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
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.REPORT_BLOCK_ALLOCATION,
                [os.getpid(), self.model_name, records],
            )

    # Helper functions
    def _record_hotprefix_control(
        self,
        method: str,
        outcome: str,
        started_ns: int,
        request_bytes: int,
        response_bytes: int = 0,
    ) -> None:
        observation: dict[str, int | float | str] = {
            "method": method,
            "outcome": outcome,
            "duration_seconds": (time.monotonic_ns() - started_ns) / 1e9,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
        }
        with self._hotprefix_control_lock:
            self._hotprefix_control_observations.append(observation)

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

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
    ):
        """Initialize the worker adapter for current or legacy vLLM callers.

        Args:
            server_url: The server URL for the LMCache message queue.
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

        Raises:
            TypeError: If the connector argument shape is unsupported.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        self._hotprefix_enabled = False
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
            self._hotprefix_enabled = bool(
                cfg[ExtraConfigDefault.hotprefix_enabled.name]
            )
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
        else:
            self._mp_transfer_mode = None
        self.mq_client = MessageQueueClient(server_url, context)
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
        # KV-cache device and event backend, resolved once at registration
        # so per-request event creation stays off the lookup path.
        self._kv_device: torch.device | None = None
        self._event_ipc_backend: EventIPCBackend | None = None

        # Transport context for transfer operations.
        self.transfer_ctx: TransferContext | None = None

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}
        # The IPC handle is not enough by itself; CUDA needs the exporting
        # event object to stay alive until the consumer is done with it.
        self.store_events: dict[str, _IpcEvent] = {}
        self.retrieve_events: dict[str, _IpcEvent] = {}

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # Retrieve request ids dropped by the unhealthy early-return of
        # submit_retrieve_request. get_finished must still report each id
        # exactly once, or async loads hang in WAITING_FOR_REMOTE_KVS.
        self._dropped_retrieves: set[str] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs already returned as finished_sending to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            lmcache_tokens_per_chunk = get_lmcache_chunk_size(
                self.mq_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.mq_client.close()
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
            self.mq_client, timeout=self._mq_timeout
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
        self._failed_store_requests: set[str] = set()
        self._completed_promotion_requests: dict[str, int] = {}
        self._failed_promotion_requests: set[str] = set()

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
    def hotprefix_enabled(self) -> bool:
        """Whether canonical HotPrefix request bookkeeping is active."""
        return self._hotprefix_enabled

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
        self._kv_device = next(iter(kv_caches.values())).device
        self._event_ipc_backend = get_event_ipc_backend(self._kv_device)
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
                self.mq_client,
                self._mq_timeout,
                send_request=send_lmcache_request,
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
                mq_client=self.mq_client,
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

    def create_recorded_event(self) -> _IpcEvent:
        """Create an IPC event recorded on the current stream with configured backend.

        Returns:
            An event recorded on the current stream, ready for
            ``submit_store_request`` / ``submit_retrieve_request``.

        Raises:
            RuntimeError: If called before ``register_kv_caches()``.
        """
        if self._kv_device is None or self._event_ipc_backend is None:
            raise RuntimeError(
                "KV caches are not registered. Call register_kv_caches() "
                "before creating transfer events."
            )
        event = self._event_ipc_backend.create_event(self._kv_device)
        self._event_ipc_backend.record_event(event, torch_dev.current_stream())
        return cast(_IpcEvent, event)

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent,
        cache_salt: str = "",
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
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
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting store requests."
            )
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
        self.store_events[request_id] = event

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent,
        cache_salt: str = "",
    ) -> None:
        """
        Submit a KV cache retrieve request to LMCache

        When the server is unhealthy the request is not submitted: blocks
        are flagged via ``error_block_ids`` (vLLM recomputes) and the id is
        recorded so ``get_finished`` still reports it exactly once.

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            if not is_hotprefix_promotion_request(request_id):
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
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting retrieve requests."
            )
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
        self.retrieve_events[request_id] = event

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: _IpcEvent,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be the same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_store_request(request_id, op, event, cache_salt=salt)

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: _IpcEvent,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_retrieve_request(request_id, op, event, cache_salt=salt)

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(
            request_id
            for request_id in finished_req_ids_from_lmcache
            if not is_hotprefix_store_request(request_id)
        )
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self._returned_finished:
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
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

        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                if not is_hotprefix_promotion_request(request_id):
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
            self._record_hotprefix_promotion_results(
                finished_retrieves,
                failed_request_ids=set(finished_retrieves),
            )

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            for request_id in finished_stores:
                if is_hotprefix_store_request(request_id):
                    self._completed_store_requests[request_id] = 1
                    self._failed_store_requests.add(request_id)
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

            if is_hotprefix_store_request(request_id):
                self._completed_store_requests[request_id] = 1

            if not s_result:
                if is_hotprefix_store_request(request_id):
                    self._failed_store_requests.add(request_id)
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
                failed_retrieves.add(request_id)
                if not is_hotprefix_promotion_request(request_id):
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
        failed_retrieves.update(dropped)
        self._record_hotprefix_promotion_results(
            finished_retrieves,
            failed_request_ids=failed_retrieves,
        )

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

        if self.dispatcher is not None:
            dispatch(self.dispatcher, "reclaim")

        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                if not is_hotprefix_promotion_request(request_id):
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
            self._record_hotprefix_promotion_results(
                finished_retrieves,
                failed_request_ids=set(finished_retrieves),
            )

            for req_id in finished_stores:
                self._completed_store_requests[req_id] = 1
                if is_hotprefix_store_request(req_id):
                    self._failed_store_requests.add(req_id)
            return None, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        failed_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result(timeout=60)
            finished_stores.add(request_id)

            if not s_result:
                if is_hotprefix_store_request(request_id):
                    self._failed_store_requests.add(request_id)
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
                failed_retrieves.add(request_id)
                if not is_hotprefix_promotion_request(request_id):
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
        failed_retrieves.update(dropped)
        self._record_hotprefix_promotion_results(
            finished_retrieves,
            failed_request_ids=failed_retrieves,
        )

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
        """Return completed store requests since the last call.

        Returns:
            Per-request worker completion counts, or ``None`` when empty.
        """
        if not self._completed_store_requests:
            return None
        completed_store_requests = self._completed_store_requests
        self._completed_store_requests = {}
        return completed_store_requests

    def get_failed_store_requests(self) -> set[str]:
        """Return failed store requests since the last call.

        Returns:
            Store request IDs with at least one failed worker.
        """
        failed = self._failed_store_requests
        self._failed_store_requests = set()
        return failed

    def get_completed_promotion_requests(self) -> dict[str, int] | None:
        """Return completed background promotion retrieves.

        Returns:
            Per-chunk worker completion counts, or ``None`` when empty.
        """
        if not self._completed_promotion_requests:
            return None
        completed = self._completed_promotion_requests
        self._completed_promotion_requests = {}
        return completed

    def get_failed_promotion_requests(self) -> set[str]:
        """Return failed background promotion retrieves.

        Returns:
            Promotion chunk IDs with at least one failed worker.
        """
        failed = self._failed_promotion_requests
        self._failed_promotion_requests = set()
        return failed

    def _record_hotprefix_promotion_results(
        self,
        finished_request_ids: set[str],
        *,
        failed_request_ids: set[str],
    ) -> None:
        promotions = {
            request_id
            for request_id in finished_request_ids
            if is_hotprefix_promotion_request(request_id)
        }
        for request_id in promotions:
            self._completed_promotion_requests[request_id] = 1
            if request_id in failed_request_ids:
                self._failed_promotion_requests.add(request_id)
        finished_request_ids.difference_update(promotions)

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
        self.transfer_ctx.flush_inflight_stores()
        # Force device sync here, compare to preemption, perf panelty is trivial
        torch_dev.synchronize()

    def shutdown(self) -> None:
        """
        Shutdown the LMCache MP worker adapter.

        Stops the heartbeat (if started) before UNREGISTER: no new ping
        on the closing mq_client, and a straggler in-flight cycle cannot
        re-register or flip the health event after unregistration.
        """
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                self._heartbeat.stop()

        logger.info("Unregistering kv caches")
        try:
            unregister_type = (
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
                if isinstance(self.transfer_ctx, EngineDrivenTransferContext)
                else RequestType.UNREGISTER_KV_CACHE
            )
            send_lmcache_request(
                self.mq_client,
                unregister_type,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
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

        self.mq_client.close()
        self.request_telemetry.close()

    # Helper functions
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
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

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
        )
