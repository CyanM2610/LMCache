# SPDX-License-Identifier: Apache-2.0
"""Benchmark direct CUDA transfers through a registered POSIX SHM region."""

# Standard
from argparse import ArgumentParser, Namespace
from multiprocessing import shared_memory
from pathlib import Path
from statistics import median
from typing import Any, Callable
import hashlib
import json
import os
import platform
import time
import uuid

# Third Party
import torch

# First Party
import lmcache.c_ops as c_ops
from lmcache.v1.multiprocess.cxl.contracts import TransferDirection
from lmcache.v1.multiprocess.cxl.region_provider import (
    REGION_HEADER_SIZE,
    pack_region_header,
)


_BLOCK_SIZE = 16
_HEAD_SIZE = 256


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="65536,1048576,16777216,67108864")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--fragments", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _perf_event_paranoid() -> int | None:
    try:
        return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
    except (OSError, ValueError):
        return None


def build_block_map(
    num_blocks: int, *, fragmented: bool, fragment_count: int
) -> tuple[int, ...]:
    """Build a deterministic physical paged-block map.

    Args:
        num_blocks: Number of logical blocks to transfer.
        fragmented: Whether to insert physical gaps between block runs.
        fragment_count: Maximum number of physical block runs.

    Returns:
        Physical block IDs in logical packed order.

    Raises:
        ValueError: If block or fragment counts are not positive.
    """
    if num_blocks <= 0 or fragment_count <= 0:
        raise ValueError("block and fragment counts must be positive")
    if not fragmented:
        return tuple(range(num_blocks))
    actual_count = min(fragment_count, num_blocks)
    base, remainder = divmod(num_blocks, actual_count)
    result: list[int] = []
    physical_block = 0
    for index in range(actual_count):
        length = base + (1 if index < remainder else 0)
        result.extend(range(physical_block, physical_block + length))
        physical_block += length
        if index + 1 < actual_count:
            physical_block += 1
    return tuple(result)


def _measure(
    operation: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
    range_name: str,
) -> tuple[list[int], float]:
    torch.cuda.nvtx.range_push(range_name)
    try:
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                operation()
            stream.synchronize()
            samples: list[int] = []
            cpu_start = time.process_time_ns()
            wall_start = time.monotonic_ns()
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                operation()
                end.record(stream)
                end.synchronize()
                samples.append(int(start.elapsed_time(end) * 1_000_000))
    finally:
        torch.cuda.nvtx.range_pop()
    wall_ns = time.monotonic_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    cpu_percent = 100.0 * cpu_ns / wall_ns if wall_ns else 0.0
    return samples, cpu_percent


def _run_case(
    registration: Any,
    *,
    size: int,
    direction: TransferDirection,
    fragmented: bool,
    fragment_count: int,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    bytes_per_block = _BLOCK_SIZE * _HEAD_SIZE
    if size % bytes_per_block:
        raise ValueError(f"size must be divisible by {bytes_per_block}")
    num_blocks = size // bytes_per_block
    block_map = build_block_map(
        num_blocks, fragmented=fragmented, fragment_count=fragment_count
    )
    physical_blocks = block_map[-1] + 1
    paged = torch.arange(
        physical_blocks * bytes_per_block,
        dtype=torch.uint8,
        device=stream.device,
    ).view(physical_blocks, _BLOCK_SIZE, _HEAD_SIZE)
    block_ids = torch.tensor(block_map, dtype=torch.int64, device=stream.device)
    expected = paged.index_select(0, block_ids).clone()
    paged_pointers = torch.tensor(
        [paged.data_ptr()], dtype=torch.int64, device=stream.device
    )
    shape = c_ops.PageBufferShapeDesc()
    shape.kv_size = 1
    shape.nl = 1
    shape.nb = physical_blocks
    shape.bs = _BLOCK_SIZE
    shape.nh = 1
    shape.hs = _HEAD_SIZE
    shape.element_size = 1
    shape.block_stride_elems = 0
    object_pointer = registration.device_address(0, size)

    def store() -> None:
        c_ops.cxl_region_block_kv_transfer(
            paged_pointers,
            [object_pointer],
            block_ids,
            stream.device,
            shape,
            c_ops.TransferDirection.D2H,
            num_blocks * _BLOCK_SIZE,
            c_ops.EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )

    def retrieve() -> None:
        c_ops.cxl_region_block_kv_transfer(
            paged_pointers,
            [object_pointer],
            block_ids,
            stream.device,
            shape,
            c_ops.TransferDirection.H2D,
            num_blocks * _BLOCK_SIZE,
            c_ops.EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )

    with torch.cuda.stream(stream):
        store()
        stream.synchronize()
        paged.index_fill_(0, block_ids, 0)
        stream.synchronize()
    if direction == "retrieve":
        operation = retrieve
    else:
        with torch.cuda.stream(stream):
            retrieve()
        stream.synchronize()
        operation = store

    samples, cpu_percent = _measure(
        operation,
        stream=stream,
        warmup=warmup,
        iterations=iterations,
        range_name=(
            f"beluga_proxy_{size}_{direction}_"
            f"{'fragmented' if fragmented else 'contiguous'}"
        ),
    )
    if direction == "store":
        with torch.cuda.stream(stream):
            paged.index_fill_(0, block_ids, 0)
            retrieve()
        stream.synchronize()

    actual = paged.index_select(0, block_ids)
    expected_checksum = hashlib.sha256(expected.cpu().numpy().tobytes()).hexdigest()
    actual_checksum = hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()
    return {
        "size_bytes": size,
        "direction": direction,
        "layout": "fragmented_block_map" if fragmented else "contiguous_block_map",
        "fragment_count": min(fragment_count, num_blocks) if fragmented else 1,
        "logical_block_count": num_blocks,
        "physical_block_count": physical_blocks,
        "block_map_sha256": hashlib.sha256(
            json.dumps(block_map, separators=(",", ":")).encode()
        ).hexdigest(),
        "block_ids_preview": list(block_map[:16]),
        "iterations": iterations,
        "warmup": warmup,
        "checksum_expected": expected_checksum,
        "checksum_actual": actual_checksum,
        "checksum_ok": expected_checksum == actual_checksum,
        "cuda_elapsed_ns": samples,
        "cuda_p50_ns": int(median(samples)),
        "cuda_p95_ns": _percentile(samples, 0.95),
        "cuda_p99_ns": _percentile(samples, 0.99),
        "cpu_utilization_percent_single_core": cpu_percent,
    }


def main() -> None:
    """Run the complete Gate A size/layout/direction matrix.

    Raises:
        ValueError: If benchmark dimensions or iteration counts are invalid.
        RuntimeError: If CUDA registration or a transfer operation fails.
        OSError: If shared memory or the output artifact cannot be accessed.
    """
    args = _parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")
    if args.warmup < 0 or args.iterations <= 0 or args.fragments <= 0:
        raise ValueError("warmup, iterations, and fragments are invalid")

    torch.cuda.set_device(args.device)
    stream = torch.cuda.Stream(device=args.device)
    capacity = max(sizes)
    name = f"beluga-gate-a-{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(
        name=name, create=True, size=REGION_HEADER_SIZE + capacity
    )
    registration = None
    try:
        header = pack_region_header(capacity, 4096)
        buffer = shm.buf
        if buffer is None:
            raise RuntimeError("POSIX shared region has no mapped buffer")
        buffer[: len(header)] = header
        registration = c_ops.CudaRegionRegistration(f"/{name}", capacity)
        directions: tuple[TransferDirection, ...] = ("store", "retrieve")
        cases = [
            _run_case(
                registration,
                size=size,
                direction=direction,
                fragmented=fragmented,
                fragment_count=args.fragments,
                stream=stream,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for size in sizes
            for fragmented in (False, True)
            for direction in directions
        ]
        result = {
            "schema_version": 1,
            "path": "cuda_registered_posix_shm_direct",
            "payload_staging": False,
            "pid": os.getpid(),
            "host": platform.node(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_index": args.device,
            "device_name": torch.cuda.get_device_name(args.device),
            "telemetry": {
                "gpu_pcie_tx_rx": {"status": "external_capture_required"},
                "cpu_memory_controller": {
                    "status": "external_capture_required",
                    "perf_event_paranoid": _perf_event_paranoid(),
                },
            },
            "cases": cases,
            "all_checksums_ok": all(case["checksum_ok"] for case in cases),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    finally:
        if registration is not None:
            registration.close()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
