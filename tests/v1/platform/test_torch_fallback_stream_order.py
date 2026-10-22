# SPDX-License-Identifier: Apache-2.0
"""The raw-pointer fallback must respect the active CUDA stream."""

import pytest
import torch

from lmcache import lmcache_native
from lmcache.v1.platform import torch_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_pointer_copy_obeys_prior_work_on_current_stream(direction):
    device = torch.device("cuda:0")
    gpu = torch.full((1 << 18,), 11, dtype=torch.uint8, device=device)
    host = torch.full((1 << 18,), 23, dtype=torch.uint8, pin_memory=True)
    observed = torch.zeros_like(gpu)
    stream = torch.cuda.Stream(device=device)
    # Load libcudart before the deliberate delay so dynamic loading cannot
    # accidentally serialize the producer and make the broken path pass.
    torch_ops.lmcache_memcpy_async(
        observed.data_ptr(),
        host.data_ptr(),
        host.numel(),
        lmcache_native.TransferDirection.H2D,
        0,
        1 << 26,
    )
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream):
        # A bounded device delay makes missing stream ordering deterministic.
        torch.cuda._sleep(100_000_000)
        if direction == "d2h":
            gpu.fill_(37)
            torch_ops.lmcache_memcpy_async(
                host.data_ptr(),
                gpu.data_ptr(),
                gpu.numel(),
                lmcache_native.TransferDirection.D2H,
                0,
                1 << 26,
            )
        else:
            observed.copy_(gpu)
            torch_ops.lmcache_memcpy_async(
                gpu.data_ptr(),
                host.data_ptr(),
                host.numel(),
                lmcache_native.TransferDirection.H2D,
                0,
                1 << 26,
            )
    stream.synchronize()
    if direction == "d2h":
        assert torch.all(host == 37), "copied before the producer filled its source"
    else:
        assert torch.all(observed == 11), (
            "overwrote staging before its previous consumer"
        )
        assert torch.all(gpu == 23)
