# SPDX-License-Identifier: Apache-2.0
"""Opt-in bytewise transfer checks using independent PyTorch block selection.

This deliberately synchronizes and copies debug data to CPU. Its timings are
not normal-path performance. The first supported geometry is dense/GQA
NL x [NB, BS, NH, 2*HS], as used by the fixed Qwen2.5-7B deployment.
"""

# Standard
from collections.abc import Sequence
import hashlib

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as native


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Read only a defined tensor view as bytes, with no BF16 numeric conversion."""
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


class KvContentCheck:
    """Snapshot each defined source and compare actual destination after copy."""

    def __init__(
        self,
        context: BaseCacheContext,
        block_ids: list[torch.Tensor],
        objects: Sequence[MemoryObj | None],
        object_group_id: int,
        skip_tokens: int,
        direction: native.TransferDirection,
        transfer_key: str,
    ) -> None:
        """Capture protected source bytes after the producer stream dependency.

        Args:
            context: Actual engine cache context, including imported GPU tensors.
            block_ids: Physical block IDs on GPU, per kernel group.
            objects: Real host objects in chunk order; None means no STORE.
            object_group_id: Logical layer group index.
            skip_tokens: HBM-hit tokens that H2D must not overwrite.
            direction: Real transfer direction.
            transfer_key: Correlation with the actual MP operation.
        Raises:
            NotImplementedError: For a geometry not independently verified here.
        """
        self.context = context
        self.objects = objects
        self.direction = direction
        self.transfer_key = transfer_key
        self.object_group_id = object_group_id
        group = context.kv_layer_groups_manager.object_groups[object_group_id]
        if len(group.kernel_group_indices) != 1:
            raise NotImplementedError("content checker requires one kernel group")
        kernel_id = group.kernel_group_indices[0]
        if (
            context.get_engine_kv_format(kernel_id)
            != native.EngineKVFormat.NL_X_NB_BS_NH_CS
        ):
            raise NotImplementedError("content checker requires NL x [NB,BS,NH,CS]")
        self.group = context.kv_layer_groups_manager.kernel_groups[kernel_id]
        self.chunk_tokens = context.lmcache_tokens_per_chunk
        self.bpc = context.calculate_num_blocks(self.chunk_tokens, kernel_id)
        self.skip_tokens = (
            skip_tokens if direction == native.TransferDirection.H2D else 0
        )
        context.stream.synchronize()
        self.ids = block_ids[kernel_id].cpu().tolist()
        self.before = {
            index: self._source(index, obj)
            for index, obj in enumerate(objects)
            if obj is not None and self._skip(index) < self.chunk_tokens
        }

    def verify(self) -> None:
        """Compare destination bytes and emit auditable hashes before publication.

        Raises:
            RuntimeError: On any mismatch, so no invalid STORE can publish READY.
        """
        self.context.stream.synchronize()
        for index, expected in self.before.items():
            obj = self.objects[index]
            assert obj is not None
            actual = (
                self._host(index, obj)
                if self.direction == native.TransferDirection.D2H
                else self._gpu(index)
            )
            equal = expected == actual
            get_event_bus().publish(
                Event(
                    event_type=EventType.MP_KV_CONTENT_CHECK,
                    session_id=self.transfer_key.rsplit("#", 1)[0],
                    metadata={
                        "transfer_key": self.transfer_key,
                        "object_group_id": self.object_group_id,
                        "chunk_index": index,
                        "direction": "load"
                        if self.direction == native.TransferDirection.H2D
                        else "store",
                        "host_ptr": obj.data_ptr,
                        "host_slot_offset": obj.meta.address,
                        "physical_bytes": obj.meta.phy_size,
                        "effective_bytes_checked": len(expected),
                        "skipped_prefix_tokens": self._skip(index),
                        "block_ids": self.ids[
                            index * self.bpc : (index + 1) * self.bpc
                        ],
                        "layer_indices": self.group.layer_indices,
                        "source_sha256": hashlib.sha256(expected).hexdigest(),
                        "destination_sha256": hashlib.sha256(actual).hexdigest(),
                        "equal": equal,
                        "synchronizing_diagnostic": True,
                    },
                )
            )
            if not equal:
                raise RuntimeError(
                    f"KV content mismatch: {self.transfer_key} chunk {index}"
                )

    def _skip(self, index: int) -> int:
        return max(
            0, min(self.chunk_tokens, self.skip_tokens - index * self.chunk_tokens)
        )

    def _gpu(self, index: int) -> bytes:
        ids = self.ids[index * self.bpc : (index + 1) * self.bpc]
        indices = torch.tensor(ids, dtype=torch.long, device=self.context.device)
        layers = [
            tensor.index_select(0, indices).reshape(self.chunk_tokens, -1)
            for layer in self.group.layer_indices
            for tensor in [self.context.kv_tensors[layer]]
        ]
        return tensor_bytes(torch.stack(layers)[:, self._skip(index) :, :])

    def _host(self, index: int, obj: MemoryObj) -> bytes:
        raw = obj.raw_tensor
        assert raw is not None
        view = (
            raw[: obj.get_size()]
            .view(self.group.dtype)
            .reshape(len(self.group.layer_indices), self.chunk_tokens, -1)
        )
        return tensor_bytes(view[:, self._skip(index) :, :])

    def _source(self, index: int, obj: MemoryObj) -> bytes:
        return (
            self._gpu(index)
            if self.direction == native.TransferDirection.D2H
            else self._host(index, obj)
        )
