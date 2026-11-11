# SPDX-License-Identifier: Apache-2.0
"""CXLMemSim metadata transport and direct payload mapping.

This module never copies payloads. Its caller owns CUDA registration, streams,
and source/destination lifetimes. Completion requires both a reported successful
CUDA copy and the simulator's terminal COMPLETE state. Modeled time is not DMA.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import mmap
import os
import secrets
import struct
import threading
import time


def shm_path(name: str) -> Path:
    """Resolve a single POSIX SHM name, rejecting traversal and empty names.

    Args:
        name: Leading slash followed by one nonempty component.
    Returns:
        The Linux shared-memory filesystem path.
    Raises:
        ValueError: If name is not a single POSIX SHM component.
    """
    if not name.startswith("/") or name[1:] in ("", ".", ".."):
        raise ValueError("invalid POSIX shared-memory name")
    if "/" in name[1:] or "\x00" in name:
        raise ValueError("invalid POSIX shared-memory name")
    return Path("/dev/shm") / name[1:]


def read_payload_descriptor(control_name: str) -> PayloadDescriptor:
    """Read and validate CXLMemSim's bulk-v3 payload descriptor.

    Args:
        control_name: POSIX name of the live bulk control object.
    Returns:
        Validated metadata; does not create or modify server objects.
    Raises:
        OSError: If the control file cannot be read.
        ValueError: If ABI, readiness, name, or bounds are invalid.
    """
    with shm_path(control_name).open("rb") as source:
        raw = source.read(256)
    if len(raw) != 256:
        raise ValueError("truncated CXLMemSim bulk header")
    magic, version, size, slots, line, ready, _flags = struct.unpack_from("<Q6I", raw)
    if (magic, version, size, slots, line, ready) != (
        0x43584C42554C4B31,
        3,
        8448,
        64,
        64,
        1,
    ):
        raise ValueError("incompatible or inactive CXLMemSim bulk header")
    generation, mapped, offset, capacity = struct.unpack_from("<4Q", raw, 32)
    name = raw[96:224].split(b"\0", 1)[0].decode("ascii")
    shm_path(name)
    if not capacity or offset % mmap.PAGESIZE or offset + capacity > mapped:
        raise ValueError("invalid CXLMemSim payload bounds or alignment")
    if shm_path(name).stat().st_size != mapped:
        raise ValueError("CXLMemSim payload file size differs from descriptor")
    return PayloadDescriptor(name, mapped, offset, capacity, generation)


class Capabilities(ct.Structure):
    """ABI mirror of cxl_modeled_capabilities_t, protocol version one."""

    _fields_ = [
        ("protocol_version", ct.c_uint32),
        ("provider_kind", ct.c_uint32),
        ("capability_bits", ct.c_uint64),
        ("server_generation", ct.c_uint64),
        ("data_capacity", ct.c_uint64),
        ("data_alignment", ct.c_uint64),
    ]


class BeginResult(ct.Structure):
    """Native access token and initial model completion estimate."""

    _fields_ = [
        ("access_token", ct.c_uint64),
        ("estimated_complete_ns", ct.c_uint64),
    ]


class PollResult(ct.Structure):
    """Native state and model times; all durations/timestamps use nanoseconds."""

    _fields_ = [
        ("state", ct.c_uint32),
        ("error_code", ct.c_uint32),
        ("queue_ns", ct.c_uint64),
        ("service_ns", ct.c_uint64),
        ("modeled_complete_ns", ct.c_uint64),
    ]


class AccessRange(ct.Structure):
    """One ordered native batch input; offsets and lengths are bytes."""

    _fields_ = [("op_id", ct.c_uint64), ("offset", ct.c_uint64), ("bytes", ct.c_uint64)]


class TimedBegin(ct.Structure):
    """A successful token plus actual native call boundaries on the CPU clock."""

    _fields_ = [
        ("result", BeginResult),
        ("started_ns", ct.c_uint64),
        ("observed_ns", ct.c_uint64),
    ]


class BatchPollResult(ct.Structure):
    """Per-token transport status, distinct from the modeled terminal state."""

    _fields_ = [
        ("call_status", ct.c_int32),
        ("reserved", ct.c_uint32),
        ("result", PollResult),
    ]


@dataclass(frozen=True)
class PayloadDescriptor:
    """Validated server-published POSIX mapping; sizes and offset are bytes."""

    name: str
    mapped_bytes: int
    data_offset: int
    capacity: int
    bulk_generation: int


class PayloadMapping:
    """Map only the server's data area; caller must drain CUDA before close."""

    def __init__(self, descriptor: PayloadDescriptor) -> None:
        """Map descriptor's payload without an intermediate payload buffer.

        Args:
            descriptor: Validated bulk descriptor.
        Raises:
            OSError: If the payload cannot be opened or mapped.
        """
        self.descriptor = descriptor
        fd = os.open(shm_path(descriptor.name), os.O_RDWR)
        try:
            self.mapping = mmap.mmap(
                fd,
                descriptor.capacity,
                offset=descriptor.data_offset,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)
        self.address = ct.addressof(ct.c_char.from_buffer(self.mapping))

    def close(self) -> None:
        """Unmap after all transfers and exported tensor views are released.

        Raises:
            BufferError: If Python still holds exported views.
        """
        self.mapping.close()


class ModeledClient:
    """Serialized, fail-closed C ABI client. Multiple accesses may be in flight."""

    def __init__(
        self, library: str, control_name: str, capacity: int, timeout_ms: int = 2000
    ) -> None:
        """Attach to a live modeled-access-v1 server and register the region.

        Args:
            library: Absolute path to libcxlmemsim_modeled_client.so.
            control_name: POSIX modeled control name.
            capacity: Actual mapped payload bytes, must match capabilities.
            timeout_ms: Bound on each IPC call.
        Raises:
            RuntimeError: If the native protocol rejects a call/capability.
            ValueError: If arguments or the advertised topology are invalid.
        """
        shm_path(control_name)
        if capacity <= 0 or timeout_ms <= 0:
            raise ValueError("capacity and timeout must be positive")
        self.library = ct.CDLL(library)
        self.lock = threading.RLock()
        self.handle = ct.c_void_p()
        self.active: set[int] = set()
        self.reported: dict[int, bool] = {}
        self.cancelled: set[int] = set()
        self.client_id = secrets.randbits(63) or 1
        self.capacity = capacity
        self._bind()
        self._call(
            "client_open", control_name.encode(), timeout_ms, ct.byref(self.handle)
        )
        try:
            self.capabilities = Capabilities()
            self._call("get_capabilities", self.handle, ct.byref(self.capabilities))
            cap = self.capabilities
            self.combined_completion = bool(cap.capability_bits & 2) and hasattr(
                self.library, "cxl_modeled_complete_and_poll"
            )
            self.batch_calls = all(
                hasattr(self.library, "cxl_modeled_" + name)
                for name in ("begin_access_batch", "completion_batch")
            )
            if (
                cap.protocol_version != 1
                or cap.provider_kind != 1
                or not cap.capability_bits & 1
                or cap.data_capacity != capacity
                or cap.data_alignment != mmap.PAGESIZE
            ):
                raise ValueError("CXLMemSim capability/capacity/alignment mismatch")
            self.region = ct.c_uint64()
            self._call(
                "register_region",
                self.handle,
                self.client_id,
                capacity,
                cap.data_alignment,
                1,
                ct.byref(self.region),
            )
        except BaseException:
            self.library.cxl_modeled_client_close(self.handle)
            self.handle = ct.c_void_p()
            raise

    def begin(self, direction: str, offset: int, size: int) -> BeginResult:
        """Start model timing before CUDA submission; never move payload.

        Args:
            direction: ``load`` (CXL to GPU) or ``store`` (GPU to CXL).
            offset: Byte offset within the registered payload.
            size: Positive effective transfer bytes.
        Returns:
            Token to retain until a terminal poll, plus an estimate.
        Raises:
            ValueError: If direction or byte range is invalid.
            RuntimeError: If closed or the native call fails.
        """
        if direction not in ("load", "store"):
            raise ValueError("direction must be load or store")
        if offset < 0 or size <= 0 or offset + size > self.capacity:
            raise ValueError("transfer outside payload")
        with self.lock:
            result = BeginResult()
            self._call(
                "begin_access",
                self.handle,
                secrets.randbits(63) or 1,
                self.client_id,
                1 if direction == "load" else 2,
                self.region,
                offset,
                size,
                time.monotonic_ns(),
                ct.byref(result),
            )
            self.active.add(result.access_token)
            return result

    def begin_many(
        self, direction: str, ranges: list[tuple[int, int]], begun: list[TimedBegin]
    ) -> None:
        """Begin ordered accesses with one Python/native boundary crossing.

        Args:
            direction: ``load`` or ``store``.
            ranges: Ordered (offset, bytes) pairs in the registered payload.
            begun: Caller-owned output; retains every successful prefix token,
                even when a later native exchange fails. Caller must drain it.
        Raises:
            ValueError: On invalid direction or any out-of-bounds input.
            RuntimeError: On native failure, after preserving successful tokens.
        Notes:
            Older libraries use individual calls. The server still processes
            each access separately, with its original queue and ownership rules.
        """
        if direction not in ("load", "store"):
            raise ValueError("direction must be load or store")
        if any(o < 0 or n <= 0 or o + n > self.capacity for o, n in ranges):
            raise ValueError("transfer outside payload")
        if not ranges:
            return
        with self.lock:
            if not self.batch_calls:
                for offset, size in ranges:
                    start = time.monotonic_ns()
                    result = self.begin(direction, offset, size)
                    begun.append(TimedBegin(result, start, time.monotonic_ns()))
                return
            requests = (AccessRange * len(ranges))(
                *(AccessRange(secrets.randbits(63) or 1, o, n) for o, n in ranges)
            )
            results = (TimedBegin * len(ranges))()
            completed = ct.c_uint32()
            try:
                self._call(
                    "begin_access_batch",
                    self.handle,
                    self.client_id,
                    1 if direction == "load" else 2,
                    self.region,
                    requests,
                    len(ranges),
                    results,
                    ct.byref(completed),
                )
            finally:
                for result in results[: completed.value]:
                    self.active.add(result.result.access_token)
                    begun.append(result)

    def poll_many(
        self, tokens: list[int], *, cuda_success: bool | None = None
    ) -> list[PollResult | RuntimeError]:
        """Poll ordered tokens, optionally reporting an observed CUDA fence.

        Args:
            tokens: Distinct active tokens in submission order.
            cuda_success: None polls only; a bool reports terminal CUDA first.
        Returns:
            One state or protocol error per token. Every successful terminal
            state retains the same per-token DMA-fence checks as ``poll``.
        Raises:
            ValueError: On duplicate, absent, or already-reported inputs.
        Notes:
            The CUDA report remains recorded locally on IPC failure, so the
            caller can fail the operation and abandon only drained reservations.
        """
        with self.lock:
            if len(set(tokens)) != len(tokens) or any(
                t not in self.active for t in tokens
            ):
                raise ValueError("duplicate or unknown transfer")
            if cuda_success is not None and any(t in self.reported for t in tokens):
                raise ValueError("already reported transfer")
            if not tokens:
                return []
            if not self.batch_calls or any(t in self.cancelled for t in tokens):
                results: list[PollResult | RuntimeError] = []
                for token in tokens:
                    try:
                        result = (
                            self.poll(token)
                            if cuda_success is None
                            else self.complete_and_poll(token, cuda_success)
                        )
                        results.append(result)
                    except RuntimeError as error:
                        results.append(error)
                return results
            if cuda_success is not None:
                self.reported.update({t: cuda_success for t in tokens})
            native = (BatchPollResult * len(tokens))()
            try:
                self._call(
                    "completion_batch",
                    self.handle,
                    (ct.c_uint64 * len(tokens))(*tokens),
                    len(tokens),
                    0 if cuda_success is None else (1 if cuda_success else 2),
                    time.monotonic_ns(),
                    native,
                )
            except RuntimeError as error:
                return [error] * len(tokens)
            results = []
            for token, entry in zip(tokens, native, strict=True):
                try:
                    if entry.call_status:
                        message = self.library.cxl_modeled_error_string(
                            entry.call_status
                        ).decode()
                        raise RuntimeError("CXLMemSim completion batch: " + message)
                    results.append(self._accept_poll(token, entry.result))
                except RuntimeError as error:
                    results.append(error)
            return results

    def data_complete(self, token: int, success: bool) -> None:
        """Report an *observed* CUDA terminal state, after the caller drains it.

        Args:
            token: Active begin token.
            success: Whether CUDA completed successfully.
        Raises:
            ValueError: If the token is absent or already reported.
            RuntimeError: If native IPC fails.
        """
        with self.lock:
            if token not in self.active or token in self.reported:
                raise ValueError("unknown or already reported transfer")
            # Record the caller's real DMA fence even if the metadata RPC fails.
            self.reported[token] = success
            # The v1 server rejects data_complete after cancellation. Retain
            # the local DMA fence independently of its terminal model state.
            if token not in self.cancelled:
                self._call(
                    "data_complete",
                    self.handle,
                    token,
                    1 if success else 2,
                    time.monotonic_ns(),
                )

    def poll(self, token: int) -> PollResult:
        """Observe completion without pretending model time is CUDA time.

        Args:
            token: Active token.
        Returns:
            Native state; COMPLETE=3, CANCELLED=4, ERROR=5.
        Raises:
            ValueError: If the token is absent.
            RuntimeError: If server claims success before CUDA or IPC fails.
        """
        with self.lock:
            if token not in self.active:
                raise ValueError("unknown transfer")
            result = PollResult()
            self._call("poll_access", self.handle, token, ct.byref(result))
            return self._accept_poll(token, result)

    def complete_and_poll(self, token: int, success: bool) -> PollResult:
        """Report observed CUDA completion and check the model in one exchange.

        Args:
            token: Active, not yet reported access token.
            success: Whether the caller observed successful terminal CUDA work.
        Returns:
            Current model state, which can still be pending. Older peers use
            separate report and poll calls with the same completion semantics.
        Raises:
            ValueError: If the token is unknown or already reported.
            RuntimeError: On protocol failure or success without a CUDA fence.
        """
        with self.lock:
            if not self.combined_completion or token in self.cancelled:
                self.data_complete(token, success)
                return self.poll(token)
            if token not in self.active or token in self.reported:
                raise ValueError("unknown or already reported transfer")
            # A metadata failure must not erase the observed real DMA fence.
            self.reported[token] = success
            result = PollResult()
            self._call(
                "complete_and_poll",
                self.handle,
                token,
                1 if success else 2,
                time.monotonic_ns(),
                ct.byref(result),
            )
            return self._accept_poll(token, result)

    def cancel(self, token: int, reason: str) -> None:
        """Cancel model demand; caller must still drain CUDA and report it.

        Args:
            token: Active token.
            reason: Short diagnostic message.
        Raises:
            ValueError: If token is absent.
            RuntimeError: If native IPC fails.
        """
        with self.lock:
            if token not in self.active:
                raise ValueError("unknown transfer")
            self._call("cancel_access", self.handle, token, reason.encode()[:63])
            self.cancelled.add(token)

    def close(self) -> None:
        """Close an idle client; reject cleanup while copies remain in flight.

        Raises:
            RuntimeError: If any transfer still owns a reservation.
        """
        with self.lock:
            if self.active:
                raise RuntimeError("drain in-flight CUDA and model accesses first")
            if self.handle:
                self.library.cxl_modeled_client_close(self.handle)
                self.handle = ct.c_void_p()

    def abandon_drained(self, token: int) -> None:
        """Drop a local reservation after DMA terminal state and protocol failure.

        The caller must invalidate the failed operation, never publish READY,
        and stop using this failed backend. This does not repair server state.
        Raises ValueError if no real data-completion report was made.
        """
        with self.lock:
            if token not in self.reported:
                raise ValueError("cannot abandon a transfer before DMA drains")
            self.active.discard(token)
            self.reported.pop(token)
            self.cancelled.discard(token)

    def _accept_poll(self, token: int, result: PollResult) -> PollResult:
        if result.state == 3 and not self.reported.get(token, False):
            raise RuntimeError("server published completion before CUDA success")
        # Cancellation alone is not a DMA fence.
        if result.state in (3, 4, 5) and token in self.reported:
            self.active.remove(token)
            del self.reported[token]
            self.cancelled.discard(token)
        return result

    def _call(self, name: str, *args: object) -> None:
        if name != "client_open" and not self.handle:
            raise RuntimeError("CXLMemSim client is closed")
        code = getattr(self.library, "cxl_modeled_" + name)(*args)
        if code:
            message = self.library.cxl_modeled_error_string(code).decode()
            raise RuntimeError(f"CXLMemSim {name}: {message} ({code})")

    def _bind(self) -> None:
        ptr, u64, u32 = ct.c_void_p, ct.c_uint64, ct.c_uint32
        signatures = {
            "client_open": [ct.c_char_p, u32, ct.POINTER(ptr)],
            "client_close": [ptr],
            "get_capabilities": [ptr, ct.POINTER(Capabilities)],
            "register_region": [ptr, u64, u64, u64, u32, ct.POINTER(u64)],
            "begin_access": [
                ptr,
                u64,
                u64,
                u32,
                u64,
                u64,
                u64,
                u64,
                ct.POINTER(BeginResult),
            ],
            "data_complete": [ptr, u64, u32, u64],
            "poll_access": [ptr, u64, ct.POINTER(PollResult)],
            "cancel_access": [ptr, u64, ct.c_char_p],
            "error_string": [ct.c_int],
        }
        for name, args in signatures.items():
            function = getattr(self.library, "cxl_modeled_" + name)
            function.argtypes = args
            function.restype = ct.c_int
        if hasattr(self.library, "cxl_modeled_complete_and_poll"):
            function = self.library.cxl_modeled_complete_and_poll
            function.argtypes = [ptr, u64, u32, u64, ct.POINTER(PollResult)]
            function.restype = ct.c_int
        optional = {
            "begin_access_batch": [
                ptr,
                u64,
                u32,
                u64,
                ct.POINTER(AccessRange),
                u32,
                ct.POINTER(TimedBegin),
                ct.POINTER(u32),
            ],
            "completion_batch": [
                ptr,
                ct.POINTER(u64),
                u32,
                u32,
                u64,
                ct.POINTER(BatchPollResult),
            ],
        }
        for name, args in optional.items():
            if hasattr(self.library, "cxl_modeled_" + name):
                function = getattr(self.library, "cxl_modeled_" + name)
                function.argtypes = args
                function.restype = ct.c_int
        self.library.cxl_modeled_client_close.restype = None
        self.library.cxl_modeled_error_string.restype = ct.c_char_p
