# LMCache CXLMemSim L2 Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-shaped LMCache MP L2 adapter that stores CPU-backed
KV objects through CXLMemSim's real `bulk-shm` client ABI.

**Architecture:** A small `ctypes` wrapper owns the native bulk client and
translates its result structure into typed Python records. A separate
`L2AdapterInterface` implementation owns the volatile key index, fixed-slot
allocator, async worker pools, locking, eventfds, and listener accounting. The
adapter performs one native bulk operation per KV object; CXLMemSim retains
64-byte cache-line timing and bandwidth semantics.

**Tech Stack:** Python 3.10+, `ctypes`, `concurrent.futures`, LMCache MP L2
interfaces, PyTorch CPU `MemoryObj`, pytest, CXLMemSim C ABI v3, Conda
`cxl-lmcache` environment.

## Global Constraints

- Develop and test with `conda activate cxl-lmcache`; do not use uv.
- Do not import vLLM from LMCache.
- Load only the public ABI declared by CXLMemSim `include/cxl_bulk.h`.
- Send one complete object per native call; never issue one Python call per
  64-byte cache line.
- Pass only contiguous CPU-backed `MemoryObj` pointers to native code.
- Publish keys only after a successful full bulk write.
- Keep lookup locks distinct from active-load borrow counts.
- Return three distinct event file descriptors and signal after publishing
  each result.
- Keep the index volatile; process restart recovery is not supported.
- All new Python files begin with `# SPDX-License-Identifier: Apache-2.0` and
  use LMCache import section conventions and typed public APIs.

---

### Task 1: Native bulk client wrapper

**Files:**

- Create: `lmcache/v1/distributed/l2_adapters/cxl_memsim_client.py`
- Create: `tests/v1/distributed/test_cxl_memsim_client.py`
- Create: `tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py`

**Interfaces:**

- Produces: `CxlMemSimClient(library_path: str, control_name: str,
  timeout_ms: int)`, `BulkTransferResult`, `BulkClientStats`, and
  `CxlMemSimError`.
- Produces: `capacity`, `write_from(offset, src_ptr, size)`,
  `read_into(offset, dst_ptr, size)`, `snapshot_stats()`, and `close()`.

- [ ] **Step 1: Write failing ABI behavior tests**

  Create a complete fake CDLL surface whose callables allow `argtypes` and
  `restype` assignment. Exercise open/capacity, copy results, native error
  translation, invalid pointer/range rejection, cumulative counters, and
  idempotent close. Derive expected values as literals:

  ```python
  result = client.write_from(offset=64, src_ptr=source.data_ptr(), size=128)
  assert result == BulkTransferResult(
      bytes=128,
      host_copy_ns=11,
      model_latency_ns=22,
      serialization_ns=33,
      cacheline_count=2,
  )
  assert client.snapshot_stats().write_bytes == 128
  ```

  Also write the complete opt-in live integration scenario described in Task
  5 now, before either production module exists. This ensures the real-ABI
  round trip is observed failing for the intended missing-feature reason.

- [ ] **Step 2: Run the tests and verify RED**

  Run:

  ```bash
  source /home/zjhuang/miniforge3/etc/profile.d/conda.sh
  conda activate cxl-lmcache
  python -m pytest tests/v1/distributed/test_cxl_memsim_client.py -q
  ```

  Expected: collection fails because `cxl_memsim_client` does not exist.

- [ ] **Step 3: Implement the minimal typed wrapper**

  Define the exact native result layout and function signatures:

  ```python
  class _CxlBulkResult(ctypes.Structure):
      _fields_ = [
          ("bytes", ctypes.c_uint64),
          ("host_copy_ns", ctypes.c_uint64),
          ("model_latency_ns", ctypes.c_uint64),
          ("serialization_ns", ctypes.c_uint64),
          ("cacheline_count", ctypes.c_uint64),
      ]
  ```

  Use a condition variable to increment an active-transfer count before calling
  C, decrement it in `finally`, and make `close()` wait for zero. Treat a
  successful native call whose result byte count differs from the request as a
  protocol error. Update read/write counters under a stats lock.

- [ ] **Step 4: Run the client tests and verify GREEN**

  Run the command from Step 2 and require all tests to pass.

- [ ] **Step 5: Commit the client wrapper**

  ```bash
  git add lmcache/v1/distributed/l2_adapters/cxl_memsim_client.py \
    tests/v1/distributed/test_cxl_memsim_client.py
  git commit -m "feat(l2): wrap CXLMemSim bulk client"
  ```

### Task 2: Adapter configuration and lifecycle

**Files:**

- Create: `lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py`
- Create: `tests/v1/distributed/test_cxl_memsim_l2_adapter.py`

**Interfaces:**

- Consumes: `CxlMemSimClient` and its capacity/lifecycle API from Task 1.
- Produces: `CxlMemSimL2AdapterConfig.from_dict()` registered as
  `cxl_memsim`.
- Produces: `CxlMemSimL2Adapter` with three distinct eventfds, validated arena
  bounds, idempotent close, and registry factory construction.

- [ ] **Step 1: Write failing configuration and lifecycle tests**

  Cover required `client_library` and `slot_bytes`, positive integer fields,
  non-negative offset, optional region capacity, advertised-capacity bounds,
  lazy registry discovery, distinct eventfds, and close rejecting new tasks.
  The fake client advertises a literal capacity of 16 KiB.

- [ ] **Step 2: Run the focused tests and verify RED**

  ```bash
  python -m pytest \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py \
    -q
  ```

  Expected: collection fails because `cxl_memsim_l2_adapter` does not exist.

- [ ] **Step 3: Implement config, arena setup, task maps, and lifecycle**

  The config fields are:

  ```python
  CxlMemSimL2AdapterConfig(
      client_library: str,
      slot_bytes: int,
      control_name: str = "/cxlmemsim_bulk",
      offset_bytes: int = 0,
      capacity_bytes: int | None = None,
      timeout_ms: int = 5000,
      num_store_workers: int = 1,
      num_lookup_workers: int = 1,
      num_load_workers: int = min(4, os.cpu_count() or 1),
  )
  ```

  At adapter construction compute `available = client.capacity -
  offset_bytes`, select the configured capacity or `available`, floor it to
  complete slots, and fail if the region is empty or out of range. Allocate
  three event notifiers and three executors only after the client and arena
  validate; close partially constructed resources on any exception.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

  Run the command from Step 2 and require all tests to pass.

- [ ] **Step 5: Commit configuration and lifecycle**

  ```bash
  git add lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py
  git commit -m "feat(l2): register CXLMemSim adapter"
  ```

### Task 3: Atomic asynchronous store path

**Files:**

- Modify: `lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py`
- Modify: `tests/v1/distributed/test_cxl_memsim_l2_adapter.py`

**Interfaces:**

- Consumes: native `write_from()` from Task 1 and lifecycle state from Task 2.
- Produces: `submit_store_task()` and `pop_completed_store_tasks()`.

- [ ] **Step 1: Write failing store behavior tests**

  Cover data transfer, event signaling, one-shot completion draining, actual
  payload byte counts, slot-based usage, capacity exhaustion, host-only buffer
  rejection, failed-write rollback, keys hidden while a write is blocked, and
  a concurrent duplicate store waiting for the first write to commit.

  The atomicity assertion is observable through the public interface:

  ```python
  blocked_store = adapter.submit_store_task([key], [source])
  lookup = adapter.submit_lookup_and_lock_task([key], EMPTY_LAYOUT)
  assert wait_lookup(adapter, lookup) == [False]
  fake_client.release_write()
  assert wait_store(adapter, blocked_store).is_successful()
  assert lookup_once(adapter, key) is True
  ```

- [ ] **Step 2: Run store tests and verify RED**

  Run only test names containing `store`, `capacity`, or `publish` and confirm
  failures are missing behavior rather than fixture errors.

- [ ] **Step 3: Implement reservation, commit, and rollback**

  Store entry metadata includes slot id, logical payload size, cloned
  `cached_positions`, external lock count, read borrow count, and pending-free
  state. Reserve under a condition lock, call native code outside it, then
  either publish ready or recycle the slot and notify waiters. Only newly
  committed keys trigger `_notify_keys_stored`; use `slot_bytes` for listener
  accounting and the real payload size for `L2StoreResult`.

- [ ] **Step 4: Run store tests and verify GREEN**

  Re-run the focused store selection and then the complete adapter test file.

- [ ] **Step 5: Commit the store path**

  ```bash
  git add lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py
  git commit -m "feat(l2): store KV through CXLMemSim"
  ```

### Task 4: Lookup, load, unlock, deletion, and status

**Files:**

- Modify: `lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py`
- Modify: `tests/v1/distributed/test_cxl_memsim_l2_adapter.py`

**Interfaces:**

- Consumes: committed slot entries from Task 3 and native `read_into()` from
  Task 1.
- Produces: lookup-and-lock, load, unlock, delete, usage, and status behavior
  required by `L2AdapterInterface`.

- [ ] **Step 1: Write failing round-trip and concurrency tests**

  Cover hit/miss bitmaps, one-shot query results, payload equality, restored
  cached positions, undersized/unsupported destination failure, lock-aware
  delete, deletion deferred during a blocked load, slot recycling after the
  read finishes, listener callbacks, and native counter exposure in
  `report_status()`.

- [ ] **Step 2: Run lookup/load/delete tests and verify RED**

  Run only test names containing `lookup`, `load`, `delete`, `unlock`, or
  `status`, and confirm each failure names missing behavior.

- [ ] **Step 3: Implement read reservations and logical deletion**

  Lookup increments external lock counts only for ready keys. Load increments
  borrow counts before leaving the state lock, validates the target, reads
  directly into its pointer, restores metadata, and decrements the borrow in
  `finally`. Delete removes only unlocked ready entries; borrowed slots become
  pending-free and are recycled by the last read release. Publish task results
  before signaling their operation-specific notifier.

- [ ] **Step 4: Run the full adapter unit suite and verify GREEN**

  ```bash
  python -m pytest \
    tests/v1/distributed/test_cxl_memsim_client.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py \
    -q
  ```

- [ ] **Step 5: Commit the complete adapter behavior**

  ```bash
  git add lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py
  git commit -m "feat(l2): load and evict CXLMemSim KV"
  ```

### Task 5: Real CXLMemSim integration and user documentation

**Files:**

- Modify: `tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py`
- Create: `docs/source/mp/l2_storage/cxl_memsim.rst`
- Modify: `docs/source/mp/l2_storage/index.rst`
- Modify: `docs/source/mp/l2_storage/supported_storages.rst`

**Interfaces:**

- Consumes: the built CXLMemSim server and `libcxlmemsim_client.so` from the
  parent repository.
- Produces: a reproducible real-ABI round trip and documented LMCache/CXLMemSim
  launch commands.

- [ ] **Step 1: Re-run the opt-in live integration test written in Task 1**

  The test reads `CXLMEMSIM_SERVER` and `CXLMEMSIM_CLIENT_LIBRARY` and skips
  when either is absent. It starts a unique control object in `bulk-shm` mode
  with 16 MB capacity, waits for adapter construction, stores and reloads one
  4 KiB object, asserts byte equality and literal native deltas of one write,
  one read, 4096 bytes in each direction, and 64 cache lines per operation.
  It terminates only the child server process in `finally`.

- [ ] **Step 2: Run the live test and verify GREEN**

  ```bash
  CXLMEMSIM_SERVER=/home/zjhuang/cxl_offloading/CXLMemSim/build-vllm/cxlmemsim_server \
  CXLMEMSIM_CLIENT_LIBRARY=/home/zjhuang/cxl_offloading/CXLMemSim/build-vllm/libcxlmemsim_client.so \
  python -m pytest \
    tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py -q
  ```

  The same test was observed RED in Task 1 before implementation. It must now
  pass through the real server and client library.

- [ ] **Step 3: Add MP launch and safety documentation**

  Document the CXLMemSim server command, LMCache `--l2-adapter` JSON, volatile
  semantics, CPU-buffer requirement, meaning of native counters, and the rule
  that Device-DAX hardware uses `type: dax` instead.

- [ ] **Step 4: Run integration, regression, lint, and docs verification**

  ```bash
  python -m pytest \
    tests/v1/distributed/test_cxl_memsim_client.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py -q
  python -m pytest tests/v1/distributed/test_dax_l2_adapter.py -q
  ruff check \
    lmcache/v1/distributed/l2_adapters/cxl_memsim_client.py \
    lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_client.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py
  ruff format --check \
    lmcache/v1/distributed/l2_adapters/cxl_memsim_client.py \
    lmcache/v1/distributed/l2_adapters/cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_client.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter.py \
    tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py
  cd docs && make clean && make html
  ```

- [ ] **Step 5: Record evidence and commit**

  Save exact commands and results under
  `/home/zjhuang/cxl_offloading/results/lmcache_cxl_memsim_l2_20260726/`, then:

  ```bash
  git add tests/v1/distributed/test_cxl_memsim_l2_adapter_integration.py \
    docs/source/mp/l2_storage/cxl_memsim.rst \
    docs/source/mp/l2_storage/index.rst \
    docs/source/mp/l2_storage/supported_storages.rst
  git commit -m "test(l2): verify CXLMemSim round trip"
  ```
