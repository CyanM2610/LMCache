# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.recompute_estimator import (
    RecomputeCalibrationKey,
    RecomputeEstimator,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    ParallelStrategy,
)
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.cxl.policy_protocol import (
    GATE_E_PROTOCOL_VERSION,
    GateEConnectorTranslator,
    GateELookupResponse,
    GateERequestEnvelope,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocol import (
    get_payload_classes,
    get_response_class,
)


pytestmark = pytest.mark.no_shared_allocator


def _envelope(request_id: str = "request") -> GateERequestEnvelope:
    return GateERequestEnvelope(
        protocol_version=GATE_E_PROTOCOL_VERSION,
        request_id=request_id,
        deadline_ns=None,
        recompute_estimate_ns=10_000,
        layout_fingerprint="a" * 64,
    )


def _adapter(responses: dict[str, GateELookupResponse]) -> LMCacheMPSchedulerAdapter:
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.lmcache_tokens_per_chunk = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 2, 0, 2, 1, 1)
    adapter._server_urls = list(responses)
    adapter._health_events = {}
    adapter._mq_timeout = 30.0
    adapter._heartbeats = {}
    adapter._heartbeat_lock = threading.Lock()

    clients = {}
    for url, response in responses.items():
        event = threading.Event()
        event.set()
        adapter._health_events[url] = event
        client = MagicMock(spec=MessageQueueClient)
        future = MagicMock()
        future.result.return_value = response
        client.submit_request.return_value = future
        clients[url] = client
    adapter.mq_clients = clients
    return adapter


def test_positive_external_match_is_returned_only_with_bound_tickets() -> None:
    response = GateELookupResponse(
        GATE_E_PROTOCOL_VERSION,
        "fetch",
        512,
        ("ticket-0", "ticket-1"),
        "bound",
    )
    adapter = _adapter({"tcp://one:1": response})

    with patch.object(adapter, "_ensure_heartbeat_started"):
        result = adapter.policy_lookup(
            "request", list(range(512)), _envelope(), "tenant"
        )

    assert GateEConnectorTranslator.matched_tokens(result) == 512
    call = adapter.mq_clients["tcp://one:1"].submit_request.call_args
    assert call.args[0] is RequestType.POLICY_LOOKUP
    assert call.args[1][2] == _envelope()


def test_policy_lookup_fails_closed_and_releases_cross_server_tickets() -> None:
    adapter = _adapter(
        {
            "tcp://one:1": GateELookupResponse(
                GATE_E_PROTOCOL_VERSION,
                "fetch",
                512,
                ("ticket-0", "ticket-1"),
                "bound",
            ),
            "tcp://two:2": GateELookupResponse.recompute("policy chose recompute"),
        }
    )

    with (
        patch.object(adapter, "_ensure_heartbeat_started"),
        patch.object(adapter, "free_lookup_locks") as release,
    ):
        result = adapter.policy_lookup(
            "request", list(range(512)), _envelope(), "tenant"
        )

    assert result.status == "recompute"
    assert result.matched_tokens == 0
    release.assert_called_once_with(list(range(512)), 0, 512, "request", "tenant")


def test_recompute_estimator_is_calibrated_and_bucket_isolated() -> None:
    estimator = RecomputeEstimator(default_estimate_ns=90_000, max_samples=3)
    key = RecomputeCalibrationKey(
        "model",
        "a" * 64,
        prompt_tokens_bucket=2048,
        batch_bucket=4,
        load_bucket=1,
        device="gpu-0",
    )
    other = RecomputeCalibrationKey(
        "model",
        "a" * 64,
        prompt_tokens_bucket=8192,
        batch_bucket=4,
        load_bucket=1,
        device="gpu-0",
    )

    estimator.record(key, 30_000)
    estimator.record(key, 10_000)
    estimator.record(key, 20_000)

    assert estimator.estimate(key) == 20_000
    assert estimator.estimate(other) == 90_000
    assert estimator.calibration_version == "recompute-median-v1"
    assert estimator.raw_observations(key) == (30_000, 10_000, 20_000)


def test_policy_protocol_is_version_separate_and_ticket_precedes_response() -> None:
    assert get_payload_classes(RequestType.POLICY_LOOKUP) == [
        IPCCacheServerKey,
        int,
        GateERequestEnvelope,
    ]
    assert get_response_class(RequestType.POLICY_LOOKUP) is GateELookupResponse

    order: list[str] = []

    class DirectTier:
        def policy_bind_ready_prefix(
            self, request_id, object_keys, keys_per_chunk, envelope
        ):
            del request_id, object_keys, keys_per_chunk, envelope
            order.append("bind")
            return 1

        def get_bound_ticket_ids(self, request_id):
            del request_id
            order.append("response")
            return ("ticket",)

        def get_lookup_decision_reason(self, request_id):
            del request_id
            return "bound by policy"

        def release_lookup_tickets(self, request_id, reason, object_keys=None):
            del request_id, reason, object_keys

    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"chunk"]
    module = LookupModule(ctx, cxl_shared_tier=DirectTier())
    key = IPCCacheServerKey(
        model_name="test_model",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(16)),
        start=0,
        end=16,
        request_id="request",
    )

    response = module.policy_lookup(key, 1, _envelope())

    assert response.status == "fetch" and response.matched_tokens == 16
    assert order == ["bind", "response"]
