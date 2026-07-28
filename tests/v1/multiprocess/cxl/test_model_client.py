# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import deque

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.model_client import (
    CXLModelProtocolError,
    CXLMemSimModelClient,
    ModelCompletion,
    ModeledAccessRequest,
    RegisteredModelRegion,
)
from lmcache.v1.multiprocess.cxl.region_provider import RegionHandle


pytestmark = pytest.mark.no_shared_allocator


class _FakeTransport:
    def __init__(self) -> None:
        self.protocol_version = 1
        self.capability_names = frozenset({"gpu_direct_modeled_access_v1"})
        self.capacity = 4096
        self.alignment = 64
        self.events: list[tuple[object, ...]] = []
        self.polls: deque[ModelCompletion] = deque()
        self.disconnected = False

    def capabilities(self) -> tuple[int, frozenset[str], int, int]:
        return (
            self.protocol_version,
            self.capability_names,
            self.capacity,
            self.alignment,
        )

    def register_region(self, region_id: int, capacity: int, alignment: int) -> int:
        self.events.append(("register", region_id, capacity, alignment))
        return 17

    def begin_access(self, request: ModeledAccessRequest) -> tuple[int, int]:
        self.events.append(("begin", request))
        return 23, request.start_ns + 500

    def data_complete(
        self, access_token: int, cuda_status: str, complete_ns: int
    ) -> None:
        self.events.append(("data", access_token, cuda_status, complete_ns))

    def poll_access(self, access_token: int) -> ModelCompletion:
        if self.disconnected:
            raise CXLModelProtocolError("server disconnected")
        self.events.append(("poll", access_token))
        if self.polls:
            return self.polls.popleft()
        return ModelCompletion("op-1", "pending", 23, 10, 500, 1500, None)

    def cancel_access(self, access_token: int, reason: str) -> None:
        self.events.append(("cancel", access_token, reason))

    def close(self) -> None:
        self.events.append(("close",))


def _handle(capacity: int = 4096, alignment: int = 64) -> RegionHandle:
    return RegionHandle(
        region_id="cxlmemsim:/cxlmemsim_shared",
        shm_name="/cxlmemsim_shared",
        capacity=capacity,
        alignment=alignment,
        capabilities=frozenset({"cuda_host_register_v1", "cxlmemsim_region_v1"}),
        data_offset=4096,
    )


def _request(op_id: str = "op-1") -> ModeledAccessRequest:
    return ModeledAccessRequest(
        op_id=op_id,
        client_id=9,
        direction="store",
        server_region_token=17,
        offset=64,
        bytes=512,
        start_ns=1000,
    )


@pytest.mark.parametrize(
    ("version", "capabilities", "message"),
    [
        (2, frozenset({"gpu_direct_modeled_access_v1"}), "version"),
        (1, frozenset(), "gpu_direct_modeled_access_v1"),
    ],
)
def test_client_fails_closed_on_version_or_capability_mismatch(
    version: int, capabilities: frozenset[str], message: str
) -> None:
    transport = _FakeTransport()
    transport.protocol_version = version
    transport.capability_names = capabilities

    with pytest.raises(CXLModelProtocolError, match=message):
        CXLMemSimModelClient(transport, timeout_ns=100)


def test_client_registers_only_matching_provider_region() -> None:
    transport = _FakeTransport()
    client = CXLMemSimModelClient(transport, timeout_ns=100)

    region = client.register_region(_handle())

    assert region == RegisteredModelRegion(
        region_id="cxlmemsim:/cxlmemsim_shared",
        server_region_token=17,
        capacity=4096,
        alignment=64,
    )
    with pytest.raises(ValueError, match="capacity"):
        client.register_region(_handle(capacity=8192))
    with pytest.raises(ValueError, match="CXLMemSim"):
        client.register_region(
            RegionHandle(
                region_id="not-cxlmemsim",
                shm_name="/unrelated",
                capacity=4096,
                alignment=64,
                capabilities=frozenset({"cuda_host_register_v1"}),
            )
        )


def test_client_preserves_model_fields_and_terminal_idempotency() -> None:
    transport = _FakeTransport()
    transport.polls.extend(
        [
            ModelCompletion("op-1", "data_complete", 23, 10, 500, 1500, None),
            ModelCompletion("op-1", "ok", 23, 10, 500, 1500, None),
        ]
    )
    ticks = iter([1000, 1100, 1200])
    client = CXLMemSimModelClient(
        transport,
        timeout_ns=1000,
        clock_ns=lambda: next(ticks),
        wait=lambda: None,
    )

    pending = client.begin_access(_request())
    client.data_complete("op-1", "ok", 1200)
    client.data_complete("op-1", "ok", 1200)
    completed = client.await_completion("op-1")

    assert pending.modeled_complete_ns == 1500
    assert completed == ModelCompletion("op-1", "ok", 23, 10, 500, 1500, None)
    assert [event[0] for event in transport.events].count("data") == 1


def test_client_reports_timeout_disconnect_and_cancellation() -> None:
    transport = _FakeTransport()
    ticks = iter([1000, 1001, 1101])
    client = CXLMemSimModelClient(
        transport,
        timeout_ns=100,
        clock_ns=lambda: next(ticks),
        wait=lambda: None,
    )
    client.begin_access(_request())

    with pytest.raises(TimeoutError, match="op-1"):
        client.await_completion("op-1")

    client.cancel("op-1", "timeout")
    assert transport.events[-1] == ("cancel", 23, "timeout")

    disconnected = _FakeTransport()
    disconnected.disconnected = True
    client = CXLMemSimModelClient(disconnected, timeout_ns=100)
    client.begin_access(_request())
    with pytest.raises(CXLModelProtocolError, match="disconnected"):
        client.await_completion("op-1")
