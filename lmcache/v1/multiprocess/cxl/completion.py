# SPDX-License-Identifier: Apache-2.0
"""Completion join surface used before modeled access is enabled."""

# Local
from .contracts import CompositeCompletion, DataCompletion


class NoopModelClient:
    """Mark modeled completion as unnecessary for Gates A-C."""

    def join(self, data_completion: DataCompletion) -> CompositeCompletion:
        """Project a CUDA completion onto the composite public contract.

        Args:
            data_completion: Physical CUDA completion to expose.

        Returns:
            A composite completion with no modeled queue or service delay.
        """
        return CompositeCompletion(
            op_id=data_completion.op_id,
            cuda_status=data_completion.status,
            modeled_status="not_required",
            cuda_elapsed_ns=data_completion.elapsed_ns,
            modeled_queue_ns=None,
            modeled_service_ns=None,
        )
