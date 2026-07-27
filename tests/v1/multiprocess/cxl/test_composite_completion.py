# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.completion import (
    ModeledCompletionCoordinator,
    compose_completion,
)
from lmcache.v1.multiprocess.cxl.contracts import DataCompletion
from lmcache.v1.multiprocess.cxl.model_client import (
    ModelCompletion,
    RegisteredModelRegion,
)


pytestmark = pytest.mark.no_shared_allocator


class _FakeClient:
    def __init__(self, model: ModelCompletion) -> None:
        self.model = model
        self.events: list[tuple[object, ...]] = []

    def begin_access(self, request):
        self.events.append(("begin", request))
        return ModelCompletion(
            request.op_id,
            "pending",
            31,
            self.model.queue_ns,
            self.model.service_ns,
            self.model.modeled_complete_ns,
            None,
        )

    def data_complete(self, op_id: str, status: str, complete_ns: int) -> None:
        self.events.append(("data", op_id, status, complete_ns))

    def await_completion(self, op_id: str) -> ModelCompletion:
        self.events.append(("await", op_id))
        return self.model

    def cancel(self, op_id: str, reason: str) -> None:
        self.events.append(("cancel", op_id, reason))


def _data(complete_ns: int = 1200, status: str = "ok") -> DataCompletion:
    return DataCompletion(
        op_id="op-1",
        status=status,  # type: ignore[arg-type]
        complete_ns=complete_ns,
        elapsed_ns=200 if status == "ok" else None,
        error=None if status == "ok" else "CUDA failed",
    )


@pytest.mark.parametrize(
    ("cuda_ns", "model_ns", "effective_ns"),
    [(1200, 1500, 1500), (1800, 1500, 1800), (1500, 1500, 1500)],
)
def test_composite_completion_is_the_later_branch(
    cuda_ns: int, model_ns: int, effective_ns: int
) -> None:
    result = compose_completion(
        1000,
        _data(cuda_ns),
        ModelCompletion("op-1", "ok", 31, 100, 400, model_ns, None),
    )

    assert result.effective_complete_ns == effective_ns
    assert result.effective_elapsed_ns == effective_ns - 1000
    assert result.cuda_complete_ns == cuda_ns
    assert result.modeled_complete_ns == model_ns


def test_coordinator_reserves_model_before_launch_and_reports_data_after() -> None:
    client = _FakeClient(
        ModelCompletion("op-1", "ok", 31, 100, 400, 1500, None)
    )
    region = RegisteredModelRegion("region", 17, 4096, 64)
    coordinator = ModeledCompletionCoordinator(client, region, clock_ns=lambda: 1000)

    def launch() -> DataCompletion:
        client.events.append(("launch",))
        return _data()

    result = coordinator.run(
        op_id="op-1",
        instance_id=9,
        direction="store",
        offset=64,
        bytes=512,
        launch=launch,
    )

    assert [event[0] for event in client.events] == [
        "begin",
        "launch",
        "data",
        "await",
    ]
    assert result.cuda_status == "ok"
    assert result.modeled_status == "ok"


@pytest.mark.parametrize(
    ("data", "model", "cuda_status", "modeled_status"),
    [
        (
            _data(status="error"),
            ModelCompletion("op-1", "error", 31, 10, 500, 1500, "model saw CUDA error"),
            "error",
            "error",
        ),
        (
            _data(),
            ModelCompletion("op-1", "error", 31, 10, 500, 1500, "model failed"),
            "ok",
            "error",
        ),
    ],
)
def test_composite_completion_propagates_branch_errors(
    data: DataCompletion,
    model: ModelCompletion,
    cuda_status: str,
    modeled_status: str,
) -> None:
    result = compose_completion(1000, data, model)

    assert result.cuda_status == cuda_status
    assert result.modeled_status == modeled_status
    assert result.effective_complete_ns is None


def test_coordinator_cancels_model_before_or_after_one_branch() -> None:
    client = _FakeClient(
        ModelCompletion("op-1", "cancelled", 31, 10, 500, 1500, None)
    )
    coordinator = ModeledCompletionCoordinator(
        client, RegisteredModelRegion("region", 17, 4096, 64)
    )

    coordinator.cancel("op-1", "before begin")
    with pytest.raises(RuntimeError, match="cancelled"):
        coordinator.run(
            op_id="op-1",
            instance_id=9,
            direction="retrieve",
            offset=0,
            bytes=64,
            launch=_data,
        )

    other = ModeledCompletionCoordinator(
        client, RegisteredModelRegion("region", 17, 4096, 64)
    )
    other.cancel("unknown", "after one branch")
    assert client.events[-1] == ("cancel", "unknown", "after one branch")
