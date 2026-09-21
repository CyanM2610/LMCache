# SPDX-License-Identifier: Apache-2.0
"""Independent checksum logic tests; real GPU execution is in the 7B probes."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.kv_content_check import KvContentCheck
import lmcache.lmcache_native as native


@pytest.mark.parametrize(
    "direction", [native.TransferDirection.H2D, native.TransferDirection.D2H]
)
def test_checker_detects_content_corruption_and_ignores_allocation_padding(direction):
    """One changed defined byte must fail despite unchanged metadata/shape."""
    # Two layers, four physical blocks, two tokens, one head, combined K/V=4.
    tensors = [
        torch.arange(32, dtype=torch.float32).reshape(4, 2, 1, 4) + i * 100
        for i in range(2)
    ]
    group = SimpleNamespace(layer_indices=[0, 1], dtype=torch.float32)
    context = SimpleNamespace(
        device=torch.device("cpu"),
        kv_tensors=tensors,
        lmcache_tokens_per_chunk=4,
        stream=SimpleNamespace(synchronize=lambda: None),
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            kernel_groups=[group],
        ),
        calculate_num_blocks=lambda _tokens, _group: 2,
        get_engine_kv_format=lambda _group: native.EngineKVFormat.NL_X_NB_BS_NH_CS,
    )
    ids = torch.tensor([3, 1])
    data = torch.stack([t.index_select(0, ids).reshape(4, 4) for t in tensors])
    defined_bytes = data.numel() * data.element_size()
    raw = torch.full((defined_bytes + 64,), 77, dtype=torch.uint8)
    raw[:defined_bytes].copy_(data.view(torch.uint8).flatten())
    obj = SimpleNamespace(
        raw_tensor=raw,
        get_size=lambda: defined_bytes,
        data_ptr=raw.data_ptr(),
        meta=SimpleNamespace(address=0, phy_size=raw.numel()),
    )
    check = KvContentCheck(context, [ids], [obj], 0, 0, direction, "request#0")
    raw[defined_bytes:] = 1
    check.verify()  # Undefined allocation padding is intentionally excluded.
    if direction == native.TransferDirection.D2H:
        raw[0] ^= 1
    else:
        tensors[0][3, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="KV content mismatch"):
        check.verify()
