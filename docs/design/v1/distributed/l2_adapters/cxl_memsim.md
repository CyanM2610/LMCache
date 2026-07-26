# CXLMemSim L2 Adapter

## Purpose

The `cxl_memsim` adapter lets the LMCache multiprocess server use a
CXLMemSim `bulk-shm` arena as volatile L2 storage. LMCache remains responsible
for object identity, lifetime, locking, slot allocation, and eviction. The
simulator remains responsible for CXL cache-line timing and bandwidth
accounting.

This adapter is for experiments. Real CXL Type-3 memory exposed through
Device-DAX continues to use the existing `dax` adapter.

## Boundaries

The adapter loads CXLMemSim's installed `libcxlmemsim_client.so` through
`ctypes` and calls only the public C ABI in `include/cxl_bulk.h`:

- `cxl_bulk_client_open` and `cxl_bulk_client_close`
- `cxl_bulk_client_capacity`
- `cxl_bulk_write` and `cxl_bulk_read`
- `cxl_bulk_error_string`

LMCache sends one complete KV object per bulk call. The client performs one
host copy, while the CXLMemSim server expands the byte range into every 64-byte
cache line it touches and applies direction-specific latency and bandwidth.
LMCache must not split objects into 64-byte Python calls or import vLLM code.

## Configuration

The adapter is selected by an `--l2-adapter` JSON object:

```json
{
  "type": "cxl_memsim",
  "client_library": "/path/to/libcxlmemsim_client.so",
  "control_name": "/cxlmemsim_bulk",
  "slot_bytes": 67108864,
  "offset_bytes": 0,
  "capacity_bytes": 4294967296,
  "timeout_ms": 5000,
  "num_store_workers": 1,
  "num_lookup_workers": 1,
  "num_load_workers": 4,
  "eviction": {
    "eviction_policy": "LRU",
    "trigger_watermark": 0.9,
    "eviction_ratio": 0.1
  }
}
```

`client_library` and `slot_bytes` are required. `control_name` defaults to
`/cxlmemsim_bulk`, `offset_bytes` defaults to zero, and `capacity_bytes` may be
omitted to use the server-advertised capacity after the offset. The configured
region must fit within the advertised capacity and must contain at least one
complete slot. Worker counts and `timeout_ms` must be positive integers.

## Components

`cxl_memsim_client.py` is a typed lifecycle wrapper around the C ABI. It
validates pointer/range arguments, translates native error codes into
`CxlMemSimError`, permits concurrent transfers, prevents close from racing an
active transfer, and accumulates the timing/accounting returned by the native
client.

`cxl_memsim_l2_adapter.py` implements `L2AdapterInterface`. It owns three
distinct event notifiers, one executor per operation class, a fixed-slot arena,
an in-memory `ObjectKey` index, external lookup lock counts, active-read borrow
counts, and completed task maps. The index is intentionally volatile because
the simulator's `bulk-shm` data plane is also volatile and the server does not
persist LMCache object metadata.

The native client uses host `memcpy`, so the adapter accepts only contiguous,
CPU-backed `MemoryObj` buffers. Unsupported or undersized buffers fail their
individual store or load result without calling native code; a CUDA pointer is
never passed to the C ABI.

## State and data flow

Each slot is in one of these logical states:

1. Free: available for a new key.
2. Storing: reserved for a key but not visible to lookup.
3. Ready: indexed and readable.
4. Pending free: removed from the index but retained until active reads finish.

A store reserves a slot and publishes a storing record under the state lock,
then issues `cxl_bulk_write` without holding the state lock. Only a successful
full write publishes the key as ready. Failure removes the reservation and
returns the slot to the free list. A concurrent store of the same key waits for
the first reservation and becomes a zero-byte success only if that reservation
commits. The committed entry snapshots `cached_positions` so later reuse of the
caller-owned L1 object cannot mutate L2 metadata.

Lookup-and-lock checks only ready entries and increments an external lock count
for each hit. Load reserves each ready slot with a borrow count, issues
`cxl_bulk_read` directly into the caller-owned `MemoryObj`, restores
`cached_positions`, and releases the borrow. Unlock decrements the external
lock count synchronously.

Delete skips externally locked or storing keys. It removes an unlocked ready
entry immediately from the index and notifies listeners. Its slot is recycled
immediately when no read borrows exist, or after the final borrowed read
finishes.

## Async and error contract

Store, lookup, and load submissions allocate adapter-local task IDs. Completion
is published before the corresponding event notifier is signaled. Each lookup
or load result can be queried once. Store results report task success and the
payload bytes actually written; slot occupancy and eviction usage are reported
in `slot_bytes` units.

An individual native store or load failure is contained in its task result and
logged. Store tasks fail at task granularity as required by
`L2AdapterInterface`; lookup and load return per-key bitmaps. Construction
fails closed when the library cannot load, the server cannot be opened, the ABI
returns an error, or the configured arena exceeds the server capacity.

`close()` rejects new work, waits for submitted tasks and active native calls,
closes the native client and all three notifiers, and is idempotent.

## Observability

`report_status()` reports health, configured and advertised capacity, slot
occupancy, lock and in-flight counts, plus cumulative native read/write
requests, bytes, host-copy time, modeled latency, serialization time, and
cache-line count. These counters provide direct evidence that an LMCache hit
was served through CXLMemSim instead of only inferring it from TTFT.

## Verification

Unit tests cover configuration validation, C error translation, distinct
eventfds, store/lookup/load data integrity, publish-after-write atomicity,
capacity exhaustion, lock-aware deletion, one-shot task results, and close
lifecycle. A live integration test starts the CXLMemSim server in `bulk-shm`
mode, stores and reloads an object through the real shared library, and checks
that native byte and cache-line counters increase.
