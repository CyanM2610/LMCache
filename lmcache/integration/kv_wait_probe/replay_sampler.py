# SPDX-License-Identifier: Apache-2.0
"""Experiment-local V2 sampler hook: native compute plus device-resident tokens.

Imported through worker_extension_cls before worker/model construction. No source
in the installed vLLM/LMCache trees is edited, and trace_decode_token_ids is not
used: native public max_tokens and EOS normalization therefore stay unchanged.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from lmcache.integration.vllm.attribution_environment import EnvironmentExtension

_STATES = []


@triton.jit
def _select_reference(sampled, mapping, slot_reference, offsets, lengths, tokens,
                      total_len, prompt_len):
    row = tl.program_id(0)
    slot = tl.load(mapping + row)
    if slot < 0:
        return
    ref = tl.load(slot_reference + slot) - 1
    if ref < 0:
        return
    step = tl.load(total_len + slot) - tl.load(prompt_len + slot)
    if step < 0:
        return
    size = tl.load(lengths + ref)
    # Async scheduling may already have queued a step beyond terminal EOS.
    # Repeat EOS for that discarded work, never fall back to free sampling.
    position = tl.minimum(step, size - 1)
    token = tl.load(tokens + tl.load(offsets + ref) + position)
    tl.store(sampled + row, token.to(tl.int64))


class DeviceReferenceState:
    def __init__(self, req_states):
        self.req_states = req_states
        path = Path(os.environ["NATIVE_TOKEN_REPLAY_CATALOG"])
        data = path.read_bytes()
        self.catalog_file_sha256 = hashlib.sha256(data).hexdigest()
        catalog = json.loads(data)
        self.ids = {rid: i for i, rid in enumerate(sorted(catalog))}
        flat, offsets, lengths = [], [], []
        for rid in sorted(catalog):
            seq = catalog[rid]
            if len(seq) != 1 or type(seq[0]) is not int or seq[0] < 0:
                raise ValueError("P catalog must contain exactly the first reference token")
            offsets.append(len(flat))
            lengths.append(len(seq))
            flat.extend(seq)
        self.tokens = torch.tensor(flat, dtype=torch.int32, device=req_states.device)
        self.offsets = torch.tensor(offsets, dtype=torch.int32, device=req_states.device)
        self.lengths = torch.tensor(lengths, dtype=torch.int32, device=req_states.device)
        self.slot_reference = StagedWriteTensor(req_states.max_num_reqs,
                                                dtype=torch.int32, device=req_states.device)
        self.registrations = set()
        self.gather_calls = 0
        _STATES.append(self)

    def add_request(self, req_idx, sampling_params):
        if sampling_params.trace_decode_token_ids is not None:
            raise ValueError("native trace params would change public stopping rules")
        rid = (sampling_params.extra_args or {}).get("native_token_replay_id")
        ref = 0
        if rid is not None:
            if sampling_params.ignore_eos or sampling_params.max_tokens != 1:
                raise ValueError("public output budget or EOS changed")
            ref = self.ids[rid] + 1
            self.registrations.add(rid)
        self.slot_reference.stage_write_elem(req_idx, ref)

    def apply_staged_writes(self):
        self.slot_reference.apply_write()

    def apply_trace(self, sampled, idx_mapping):
        self.gather_calls += 1
        _select_reference[(idx_mapping.shape[0],)](
            sampled, idx_mapping, self.slot_reference.gpu, self.offsets,
            self.lengths, self.tokens, self.req_states.total_len.gpu,
            self.req_states.prompt_len.gpu)

    def snapshot(self):
        tensors = [self.tokens, self.offsets, self.lengths, self.slot_reference.gpu]
        return dict(reference_file_sha256=self.catalog_file_sha256,
                    reference_tokens=self.tokens.numel(),
                    device_bytes=sum(t.numel()*t.element_size() for t in tensors),
                    device=str(self.tokens.device), gpu_resident=True,
                    reference_requests=len(self.ids), registered_ids=sorted(self.registrations),
                    gather_calls=self.gather_calls, public_max_tokens=1,
                    stopping="native max_tokens=1 or EOS; full reference answer is external history")


source = inspect.getsource(sampler_module.Sampler.__call__)
if source.count("self.trace_replay_state.apply_trace(sampled, idx_mapping)") != 1:
    raise RuntimeError("native sampling hook changed")
if source.index("self.trace_replay_state.apply_trace") > source.index("compute_topk_scores("):
    raise RuntimeError("replay hook must precede selected-token score gathering")
sampler_module.TraceReplayState = DeviceReferenceState


class ReplayEnvironmentExtension(EnvironmentExtension):
    def token_replay_snapshot(self):
        return [s.snapshot() for s in _STATES]
