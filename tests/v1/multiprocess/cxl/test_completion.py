# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.cxl.completion import NoopModelClient
from lmcache.v1.multiprocess.cxl.contracts import DataCompletion


pytestmark = pytest.mark.no_shared_allocator


def test_noop_model_client_preserves_cuda_completion_without_modeled_delay() -> None:
    data = DataCompletion(
        op_id="op-1",
        status="ok",
        complete_ns=120,
        elapsed_ns=20,
        error=None,
    )

    completion = NoopModelClient().join(data)

    assert completion.op_id == "op-1"
    assert completion.cuda_status == "ok"
    assert completion.cuda_elapsed_ns == 20
    assert completion.modeled_status == "not_required"
    assert completion.modeled_queue_ns is None
    assert completion.modeled_service_ns is None
