# SPDX-License-Identifier: Apache-2.0
"""Capability-negotiated client for CXLMemSim modeled access."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol
import ctypes
import hashlib
import threading
import time

# Local
from .region_provider import RegionHandle


_CAPABILITY = "gpu_direct_modeled_access_v1"
_PROTOCOL_VERSION = 1
_UINT64_MAX = (1 << 64) - 1

ModelState = Literal[
    "pending",
    "modeled_complete",
    "data_complete",
    "ok",
    "cancelled",
    "error",
]


class CXLModelProtocolError(RuntimeError):
    """Modeled-access negotiation or transport failure."""


@dataclass(frozen=True)
class RegisteredModelRegion:
    """Server-issued identity for one metadata-only modeled region."""

    region_id: str
    server_region_token: int
    capacity: int
    alignment: int


@dataclass(frozen=True)
class ModeledAccessRequest:
    """Pointer-free metadata for one CXL data operation."""

    op_id: str
    client_id: int
    direction: Literal["store", "retrieve"]
    server_region_token: int
    offset: int
    bytes: int
    start_ns: int

    def __post_init__(self) -> None:
        if not self.op_id:
            raise ValueError("op_id must not be empty")
        if self.client_id <= 0 or self.client_id > _UINT64_MAX:
            raise ValueError("client_id must fit positive uint64")
        if self.direction not in ("store", "retrieve"):
            raise ValueError("direction must be store or retrieve")
        if self.server_region_token <= 0:
            raise ValueError("server_region_token must be positive")
        if self.offset < 0 or self.bytes <= 0 or self.start_ns <= 0:
            raise ValueError("offset, bytes, and start_ns are invalid")
        if self.offset > _UINT64_MAX or self.bytes > _UINT64_MAX:
            raise ValueError("extent fields must fit uint64")
        if self.offset + self.bytes > _UINT64_MAX:
            raise ValueError("extent end must fit uint64")


@dataclass(frozen=True)
class ModelCompletion:
    """Unmodified completion fields returned by the modeled service."""

    op_id: str
    status: ModelState
    access_token: int
    queue_ns: int
    service_ns: int
    modeled_complete_ns: int
    error: str | None

    def __post_init__(self) -> None:
        if not self.op_id or self.access_token <= 0:
            raise ValueError("completion identity is invalid")
        if self.status not in (
            "pending",
            "modeled_complete",
            "data_complete",
            "ok",
            "cancelled",
            "error",
        ):
            raise ValueError("invalid modeled completion state")
        if min(self.queue_ns, self.service_ns, self.modeled_complete_ns) < 0:
            raise ValueError("modeled completion values must be non-negative")
        if self.status == "error" and not self.error:
            raise ValueError("modeled error requires a diagnostic")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error model completion cannot have an error")


class CXLModelClient(Protocol):
    """Completion-model boundary used by the CXL data plane."""

    def capabilities(self) -> frozenset[str]: ...

    def register_region(self, handle: RegionHandle) -> RegisteredModelRegion: ...

    def begin_access(self, request: ModeledAccessRequest) -> ModelCompletion: ...

    def data_complete(
        self, op_id: str, cuda_status: Literal["ok", "error"], complete_ns: int
    ) -> None: ...

    def await_completion(self, op_id: str) -> ModelCompletion: ...

    def cancel(self, op_id: str, reason: str) -> None: ...

    def close(self) -> None: ...


class _ModelTransport(Protocol):
    def capabilities(self) -> tuple[int, frozenset[str], int, int]: ...

    def register_region(self, region_id: int, capacity: int, alignment: int) -> int: ...

    def begin_access(self, request: ModeledAccessRequest) -> tuple[int, int]: ...

    def data_complete(
        self, access_token: int, cuda_status: str, complete_ns: int
    ) -> None: ...

    def poll_access(self, access_token: int) -> ModelCompletion: ...

    def cancel_access(self, access_token: int, reason: str) -> None: ...

    def close(self) -> None: ...


class CXLMemSimModelClient:
    """Validate capability and expose op-ID based modeled completion."""

    def __init__(
        self,
        transport: _ModelTransport,
        *,
        timeout_ns: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wait: Callable[[], None] = lambda: time.sleep(0.00005),
    ) -> None:
        if timeout_ns <= 0:
            raise ValueError("timeout_ns must be positive")
        version, capabilities, capacity, alignment = transport.capabilities()
        if version != _PROTOCOL_VERSION:
            transport.close()
            raise CXLModelProtocolError(
                f"modeled-access version {version} != {_PROTOCOL_VERSION}"
            )
        if _CAPABILITY not in capabilities:
            transport.close()
            raise CXLModelProtocolError(f"missing capability {_CAPABILITY}")
        if capacity <= 0 or alignment <= 0 or alignment & (alignment - 1):
            transport.close()
            raise CXLModelProtocolError("modeled-access provider is invalid")
        self._transport = transport
        self._capabilities = capabilities
        self._capacity = capacity
        self._alignment = alignment
        self._timeout_ns = timeout_ns
        self._clock_ns = clock_ns
        self._wait = wait
        self._operations: dict[str, ModelCompletion] = {}
        self._data_terminals: dict[str, tuple[str, int]] = {}
        self._closed = False
        self._lock = threading.RLock()

    @classmethod
    def open(
        cls,
        *,
        library_path: str,
        control_name: str,
        timeout_ms: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> CXLMemSimModelClient:
        """Open the native shared-memory transport and negotiate capability.

        Args:
            library_path: CXLMemSim modeled-client shared library.
            control_name: Modeled-access POSIX SHM control name.
            timeout_ms: Open and operation timeout in milliseconds.
            clock_ns: Monotonic clock for local timeout enforcement.

        Returns:
            A fail-closed negotiated model client.
        """
        transport = _CtypesModelTransport.open(
            library_path=library_path,
            control_name=control_name,
            timeout_ms=timeout_ms,
        )
        return cls(
            transport,
            timeout_ns=timeout_ms * 1_000_000,
            clock_ns=clock_ns,
        )

    def capabilities(self) -> frozenset[str]:
        """Return negotiated modeled-access capability names."""
        return self._capabilities

    def register_region(self, handle: RegionHandle) -> RegisteredModelRegion:
        """Register the exact CXLMemSim provider region.

        Args:
            handle: Validated process-independent backing description.

        Returns:
            Server-issued region identity.
        """
        self._require_open()
        if "cxlmemsim_region_v1" not in handle.capabilities:
            raise ValueError("modeled access requires a CXLMemSim provider region")
        if handle.capacity != self._capacity:
            raise ValueError("region capacity does not match modeled provider")
        if handle.alignment > self._alignment or self._alignment % handle.alignment:
            raise ValueError("region alignment does not match modeled provider")
        token = self._transport.register_region(
            _stable_u64(handle.region_id), handle.capacity, handle.alignment
        )
        return RegisteredModelRegion(
            region_id=handle.region_id,
            server_region_token=token,
            capacity=handle.capacity,
            alignment=handle.alignment,
        )

    def begin_access(self, request: ModeledAccessRequest) -> ModelCompletion:
        """Reserve modeled service before CUDA launch."""
        with self._lock:
            self._require_open()
            if request.op_id in self._operations:
                raise CXLModelProtocolError("duplicate modeled op_id")
            token, modeled_complete_ns = self._transport.begin_access(request)
            completion = ModelCompletion(
                request.op_id,
                "pending",
                token,
                0,
                0,
                modeled_complete_ns,
                None,
            )
            self._operations[request.op_id] = completion
            return completion

    def data_complete(
        self, op_id: str, cuda_status: Literal["ok", "error"], complete_ns: int
    ) -> None:
        """Report CUDA terminal state exactly once."""
        if cuda_status not in ("ok", "error") or complete_ns <= 0:
            raise ValueError("invalid CUDA terminal state")
        with self._lock:
            operation = self._operation(op_id)
            terminal = (cuda_status, complete_ns)
            previous = self._data_terminals.get(op_id)
            if previous is not None:
                if previous != terminal:
                    raise CXLModelProtocolError("conflicting CUDA terminal state")
                return
            self._transport.data_complete(
                operation.access_token, cuda_status, complete_ns
            )
            self._data_terminals[op_id] = terminal

    def await_completion(self, op_id: str) -> ModelCompletion:
        """Poll until the service reaches a terminal composite state."""
        with self._lock:
            operation = self._operation(op_id)
        deadline = self._clock_ns() + self._timeout_ns
        while True:
            result = self._transport.poll_access(operation.access_token)
            if result.op_id != op_id:
                result = ModelCompletion(
                    op_id,
                    result.status,
                    result.access_token,
                    result.queue_ns,
                    result.service_ns,
                    result.modeled_complete_ns,
                    result.error,
                )
            with self._lock:
                self._operations[op_id] = result
            if result.status in ("ok", "error", "cancelled"):
                return result
            if self._clock_ns() >= deadline:
                raise TimeoutError(f"modeled access {op_id} timed out")
            self._wait()

    def cancel(self, op_id: str, reason: str) -> None:
        """Cancel a known modeled operation idempotently."""
        if not reason:
            raise ValueError("cancellation reason must not be empty")
        with self._lock:
            operation = self._operations.get(op_id)
            if operation is None or operation.status == "cancelled":
                return
            self._transport.cancel_access(operation.access_token, reason)
            self._operations[op_id] = ModelCompletion(
                op_id,
                "cancelled",
                operation.access_token,
                operation.queue_ns,
                operation.service_ns,
                operation.modeled_complete_ns,
                None,
            )

    def close(self) -> None:
        """Close the native modeled transport once."""
        with self._lock:
            if not self._closed:
                self._transport.close()
                self._closed = True

    def _operation(self, op_id: str) -> ModelCompletion:
        try:
            return self._operations[op_id]
        except KeyError as error:
            raise CXLModelProtocolError(f"unknown modeled op_id {op_id}") from error

    def _require_open(self) -> None:
        if self._closed:
            raise CXLModelProtocolError("modeled-access client is closed")


def _stable_u64(value: str) -> int:
    result = int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=8).digest(), "little"
    )
    return result or 1


class _Capabilities(ctypes.Structure):
    _fields_ = [
        ("protocol_version", ctypes.c_uint32),
        ("provider_kind", ctypes.c_uint32),
        ("capability_bits", ctypes.c_uint64),
        ("server_generation", ctypes.c_uint64),
        ("data_capacity", ctypes.c_uint64),
        ("data_alignment", ctypes.c_uint64),
    ]


class _BeginResult(ctypes.Structure):
    _fields_ = [
        ("access_token", ctypes.c_uint64),
        ("estimated_complete_ns", ctypes.c_uint64),
    ]


class _PollResult(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_uint32),
        ("error_code", ctypes.c_uint32),
        ("queue_ns", ctypes.c_uint64),
        ("service_ns", ctypes.c_uint64),
        ("modeled_complete_ns", ctypes.c_uint64),
    ]


class _CtypesModelTransport:
    _STATE_NAMES: dict[int, ModelState] = {
        0: "pending",
        1: "modeled_complete",
        2: "data_complete",
        3: "ok",
        4: "cancelled",
        5: "error",
    }

    def __init__(self, library: ctypes.CDLL, client: ctypes.c_void_p) -> None:
        self._library = library
        self._client = client
        self._op_by_token: dict[int, str] = {}

    @classmethod
    def open(
        cls, *, library_path: str, control_name: str, timeout_ms: int
    ) -> _CtypesModelTransport:
        if not library_path:
            raise ValueError("modeled client library path must not be empty")
        if not control_name.startswith("/") or "/" in control_name[1:]:
            raise ValueError("modeled control name must be a POSIX SHM name")
        if timeout_ms <= 0:
            raise ValueError("modeled timeout must be positive")
        try:
            library = ctypes.CDLL(library_path)
        except OSError as error:
            raise CXLModelProtocolError(str(error)) from error
        cls._bind(library)
        client = ctypes.c_void_p()
        result = library.cxl_modeled_client_open(
            control_name.encode(), timeout_ms, ctypes.byref(client)
        )
        if result:
            raise CXLModelProtocolError(cls._error_string(library, result))
        return cls(library, client)

    @staticmethod
    def _bind(library: ctypes.CDLL) -> None:
        library.cxl_modeled_client_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.cxl_modeled_client_open.restype = ctypes.c_int
        library.cxl_modeled_client_close.argtypes = [ctypes.c_void_p]
        library.cxl_modeled_get_capabilities.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Capabilities),
        ]
        library.cxl_modeled_get_capabilities.restype = ctypes.c_int
        library.cxl_modeled_register_region.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.cxl_modeled_register_region.restype = ctypes.c_int
        library.cxl_modeled_begin_access.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(_BeginResult),
        ]
        library.cxl_modeled_begin_access.restype = ctypes.c_int
        library.cxl_modeled_data_complete.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        library.cxl_modeled_data_complete.restype = ctypes.c_int
        library.cxl_modeled_poll_access.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_PollResult),
        ]
        library.cxl_modeled_poll_access.restype = ctypes.c_int
        library.cxl_modeled_cancel_access.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
        ]
        library.cxl_modeled_cancel_access.restype = ctypes.c_int
        library.cxl_modeled_error_string.argtypes = [ctypes.c_int]
        library.cxl_modeled_error_string.restype = ctypes.c_char_p

    def capabilities(self) -> tuple[int, frozenset[str], int, int]:
        result = _Capabilities()
        self._check(
            self._library.cxl_modeled_get_capabilities(
                self._client, ctypes.byref(result)
            )
        )
        capabilities = (
            frozenset({_CAPABILITY})
            if result.provider_kind == 1 and result.capability_bits & 1
            else frozenset()
        )
        return (
            result.protocol_version,
            capabilities,
            result.data_capacity,
            result.data_alignment,
        )

    def register_region(self, region_id: int, capacity: int, alignment: int) -> int:
        token = ctypes.c_uint64()
        self._check(
            self._library.cxl_modeled_register_region(
                self._client,
                region_id,
                capacity,
                alignment,
                1,
                ctypes.byref(token),
            )
        )
        return token.value

    def begin_access(self, request: ModeledAccessRequest) -> tuple[int, int]:
        result = _BeginResult()
        direction = 2 if request.direction == "store" else 1
        self._check(
            self._library.cxl_modeled_begin_access(
                self._client,
                _stable_u64(f"{request.client_id}:{request.op_id}"),
                request.client_id,
                direction,
                request.server_region_token,
                request.offset,
                request.bytes,
                request.start_ns,
                ctypes.byref(result),
            )
        )
        self._op_by_token[result.access_token] = request.op_id
        return result.access_token, result.estimated_complete_ns

    def data_complete(
        self, access_token: int, cuda_status: str, complete_ns: int
    ) -> None:
        status = 1 if cuda_status == "ok" else 2
        self._check(
            self._library.cxl_modeled_data_complete(
                self._client, access_token, status, complete_ns
            )
        )

    def poll_access(self, access_token: int) -> ModelCompletion:
        result = _PollResult()
        self._check(
            self._library.cxl_modeled_poll_access(
                self._client, access_token, ctypes.byref(result)
            )
        )
        try:
            state = self._STATE_NAMES[result.state]
            op_id = self._op_by_token[access_token]
        except KeyError as protocol_error:
            raise CXLModelProtocolError(
                "invalid modeled-access response"
            ) from protocol_error
        diagnostic = (
            f"modeled-access error {result.error_code}"
            if state == "error"
            else None
        )
        return ModelCompletion(
            op_id,
            state,
            access_token,
            result.queue_ns,
            result.service_ns,
            result.modeled_complete_ns,
            diagnostic,
        )

    def cancel_access(self, access_token: int, reason: str) -> None:
        self._check(
            self._library.cxl_modeled_cancel_access(
                self._client, access_token, reason.encode()
            )
        )

    def close(self) -> None:
        if self._client:
            self._library.cxl_modeled_client_close(self._client)
            self._client = ctypes.c_void_p()

    def _check(self, result: int) -> None:
        if result:
            raise CXLModelProtocolError(self._error_string(self._library, result))

    @staticmethod
    def _error_string(library: ctypes.CDLL, result: int) -> str:
        value = library.cxl_modeled_error_string(result)
        return value.decode() if value else f"modeled-access error {result}"
