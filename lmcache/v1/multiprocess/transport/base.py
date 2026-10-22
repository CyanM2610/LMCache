# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess request client contract."""

# Standard
from typing import Any, Protocol

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture


class RequestClient(Protocol):
    """Base class for method-oriented multiprocess request clients."""

    def register_kv_cache(
        self,
        instance_id: int,
        kv_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]: ...

    def unregister_kv_cache(self, instance_id: int) -> MessagingFuture[Any]: ...

    def register_q_cache(
        self,
        instance_id: int,
        q_cache: Any,
        model_name: str,
        world_size: int,
        engine_type: Any,
        layout_hints: Any,
        engine_group_infos: list[Any],
    ) -> MessagingFuture[Any]: ...

    def unregister_q_cache(self, instance_id: int) -> MessagingFuture[Any]: ...

    def store_q(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def retrieve(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[Any]: ...

    def lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def query_prefetch_status(self, request_id: str) -> MessagingFuture[Any]: ...

    def wait_prefetch_status(
        self, request_id: str, timeout: float
    ) -> MessagingFuture[Any]: ...

    def query_prefetch_lookup_hits(self, request_id: str) -> MessagingFuture[Any]: ...

    def free_lookup_locks(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def end_session(self, request_id: str) -> MessagingFuture[Any]: ...

    def get_server_generation(self) -> MessagingFuture[Any]:
        """Return a Future[str] identifying the original hybrid server incarnation.

        The generation is captured before LOOKUP; it must not be silently
        replaced on recovery. Waiting on an unavailable peer can time out.
        """
        ...

    def lookup_guarded(
        self, key: Any, tp_size: int, generation: str
    ) -> MessagingFuture[Any]:
        """Return Future[bool] confirming lookup acceptance by the named server.

        Args:
            key: Request identity and prefix.
            tp_size: Legacy wire field; actual reader count is in key.
            generation: Original incarnation from get_server_generation.
        Returns:
            True on acceptance; False for a changed server or retired request.
            The future may raise a transport error or time out when awaited.
        """
        ...

    def end_session_guarded(
        self, request_id: str, generation: str
    ) -> MessagingFuture[Any]:
        """Return Future[bool] confirming logical cleanup, preserving DMA pins.

        Args:
            request_id: Unique engine demand that has ended.
            generation: Original incarnation that could own its lookup lease.
        Returns:
            True only after cleanup in that incarnation; False for a mismatch.
            The future may raise a transport error or time out when awaited.
        """
        ...

    def register_kv_cache_engine_driven_context(
        self, payload: Any
    ) -> MessagingFuture[Any]: ...

    def unregister_kv_cache_engine_driven_context(
        self, instance_id: int
    ) -> MessagingFuture[Any]: ...

    def prepare_store(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def commit_store(
        self, key: Any, instance_id: int, data: bytes
    ) -> MessagingFuture[Any]: ...

    def prepare_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def commit_retrieve(self, key: Any, instance_id: int) -> MessagingFuture[Any]: ...

    def clear(self) -> MessagingFuture[Any]: ...

    def get_chunk_size(self) -> MessagingFuture[Any]: ...

    def ping(self, instance_id: int | None) -> MessagingFuture[Any]: ...

    def report_block_allocation(
        self,
        instance_id: int,
        model_name: str,
        records: list[Any],
    ) -> MessagingFuture[Any]: ...

    def noop(self) -> MessagingFuture[Any]: ...

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]: ...

    def cb_unregister_rope(self, instance_id: int) -> MessagingFuture[Any]: ...

    def cb_retrieve_pre_computed(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def cb_unified_lookup(self, key: Any, tp_size: int) -> MessagingFuture[Any]: ...

    def cb_protocol_handshake(self, client_version: int) -> MessagingFuture[Any]: ...

    def p2p_lookup_and_lock(
        self, keys: list[Any], group_layout_descs: dict[int, Any]
    ) -> MessagingFuture[Any]: ...

    def p2p_query_lookup_results(self, task_id: int) -> MessagingFuture[Any]: ...

    def p2p_unlock_objects(self, keys: list[Any]) -> MessagingFuture[Any]: ...

    def get_experimental(self) -> MessagingFuture[Any]: ...

    def cb_register_rope_v3(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[Any],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
    ) -> MessagingFuture[Any]: ...

    def cb_unregister_rope_v3(self, instance_id: int) -> MessagingFuture[Any]: ...

    def cb_retrieve_pre_computed_v3(
        self,
        key: Any,
        match_results: list[Any],
        block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> MessagingFuture[Any]: ...

    def close(self) -> None:
        """Close the client and release its transport resources."""
