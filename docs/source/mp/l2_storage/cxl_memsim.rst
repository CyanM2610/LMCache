CXLMemSim
=========

The ``cxl_memsim`` L2 adapter connects an LMCache multiprocess server to
CXLMemSim's ``bulk-shm`` transport. It is intended for controlled CXL-memory
experiments: LMCache owns the KV index and fixed-slot allocator, while
CXLMemSim performs the host copy and models 64-byte cache-line latency and
bulk read/write bandwidth.

Use the :doc:`DAX adapter <dax>` for real CXL memory exposed as a Linux
Device-DAX path. The ``cxl_memsim`` type is specific to the simulator and does
not open ``/dev/dax*``.

Build requirements
------------------

Build CXLMemSim with its bulk shared-memory server and public client library.
The commands below assume these artifacts:

- ``build-vllm/cxlmemsim_server``
- ``build-vllm/libcxlmemsim_client.so``

LMCache loads only the C ABI declared in CXLMemSim's ``include/cxl_bulk.h``.
No CXLMemSim Python package is required.

Start the simulator
-------------------

Start CXLMemSim before LMCache. ``--capacity`` is in MiB in the simulator
CLI, while the LMCache adapter fields use bytes.

.. code-block:: bash

    cd /path/to/CXLMemSim
    ./build-vllm/cxlmemsim_server \
        --comm-mode=bulk-shm \
        --bulk-shm-name=/lmcache_cxl \
        --capacity=16 \
        --default_latency=100 \
        --bulk-read-bandwidth=25 \
        --bulk-write-bandwidth=25

The control name must be unique for each concurrently running simulator. A
stale ``.lock`` shared-memory object may remain after a clean shutdown by
design; the server generation check prevents a new client from attaching to
an old instance.

Start LMCache MP
----------------

Configure the adapter with the same control name and the absolute path to the
client library. This example gives LMCache the complete 16 MiB simulator
region and divides it into four 4 MiB slots:

.. code-block:: bash

    lmcache server \
        --host 127.0.0.1 \
        --port 6555 \
        --l1-size-gb 1 \
        --eviction-policy noop \
        --l2-store-policy skip_l1 \
        --l2-prefetch-policy default \
        --l2-adapter '{
          "type": "cxl_memsim",
          "client_library": "/path/to/CXLMemSim/build-vllm/libcxlmemsim_client.so",
          "control_name": "/lmcache_cxl",
          "slot_bytes": 4194304,
          "capacity_bytes": 16777216,
          "timeout_ms": 5000,
          "num_store_workers": 1,
          "num_lookup_workers": 1,
          "num_load_workers": 4
        }'

``slot_bytes`` must be at least the largest complete LMCache KV object emitted
by the selected model, chunk size, dtype, and TP layout. Capacity is rounded
down to complete slots. ``offset_bytes`` and ``capacity_bytes`` can partition
one simulator data arena among independent adapter instances, but their
regions must not overlap.

Configuration fields
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Field
     - Default
     - Description
   * - ``client_library``
     - required
     - Absolute or loader-resolvable path to ``libcxlmemsim_client.so``.
   * - ``slot_bytes``
     - required
     - Positive fixed allocation size for one KV object.
   * - ``control_name``
     - ``/cxlmemsim_bulk``
     - POSIX shared-memory control name used by the server.
   * - ``offset_bytes``
     - ``0``
     - Start of the adapter-owned region in the simulator data arena.
   * - ``capacity_bytes``
     - remaining capacity
     - Size of the adapter-owned region before rounding to complete slots.
   * - ``timeout_ms``
     - ``5000``
     - Native open and transfer timeout in milliseconds.
   * - ``num_store_workers``
     - ``1``
     - Host-to-simulator worker count.
   * - ``num_lookup_workers``
     - ``1``
     - In-memory index lookup worker count.
   * - ``num_load_workers``
     - up to ``4``
     - Simulator-to-host worker count.

Behavior and limits
-------------------

- Each KV object is sent in one native bulk operation. Python does not issue
  one call per cache line; the simulator expands the byte range into touched
  64-byte cache lines and applies its latency and bandwidth model.
- Store and load buffers must be contiguous CPU-backed ``MemoryObj`` objects.
  CUDA pointers and the GDS L1 tier are not supported by this adapter.
- A key becomes visible only after a complete successful bulk write. Failed
  writes roll their slot reservation back.
- Lookup locks prevent deletion. A delete racing an active load removes the
  key immediately but delays slot reuse until the native read finishes.
- The key index is process-local and volatile. Restarting LMCache starts with
  an empty index even if bytes remain in the simulator arena.
- Capacity and eviction accounting are slot-based, not payload-byte-based.

Status counters
---------------

The adapter's status report includes a ``transport`` object with successful
read/write request counts, payload bytes, host-copy time, modeled latency,
serialization time, and touched cache-line counts. These are cumulative for
the current native client. Adapter status also exposes live, locked,
borrowed, pending-free, and occupied slot counts.

GPU-direct modeled shared tier
------------------------------

The MP server also has an opt-in GPU-visible shared-tier path. Unlike the L2
adapter above, its CUDA executor copies directly between vLLM HBM and the
page-aligned data area owned by CXLMemSim. CXLMemSim receives metadata only and
models the corresponding CXL access concurrently with CUDA.

Start the simulator with modeled access enabled, then configure LMCache with
the exact advertised data capacity (``--capacity`` MiB minus the 4096-byte
shared-memory header):

.. code-block:: bash

    ./build-vllm/cxlmemsim_server \
        --comm-mode=bulk-shm \
        --bulk-shm-name=/cxlmemsim_bulk \
        --enable-gpu-direct-modeled-access=true \
        --modeled-access-shm-name=/cxlmemsim_modeled \
        --capacity=256

    lmcache server \
        --supported-transfer-mode=lmcache_driven \
        --cxl-shared-tier-enabled \
        --cxl-shared-tier-provider=cxlmemsim_shm \
        --cxl-shared-tier-shm-name=/cxlmemsim_shared \
        --cxl-shared-tier-capacity-bytes=268431360 \
        --cxl-shared-tier-model-mode=cxlmemsim \
        --cxl-shared-tier-model-control-name=/cxlmemsim_modeled \
        --cxl-shared-tier-model-client-library=/path/to/CXLMemSim/build-vllm/libcxlmemsim_modeled_client.so

The default ``model_mode`` remains ``noop`` for the earlier POSIX-SHM path.
The ``cxlmemsim`` mode does not fall back: startup fails if the native library,
capability, protocol, backing header, capacity, or alignment is incompatible.
For each operation, LMCache reports CUDA elapsed time, modeled queue and
service time, and effective elapsed time. Effective completion is
``max(cuda_complete, modeled_complete)``; the branch durations are never
summed.
