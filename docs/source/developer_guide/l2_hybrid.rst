Peer DRAM and simulated CXL retention
=====================================

The experimental L2-hybrid backend connects the existing CXLMemSim payload
mapping directly to the LMCache-driven GPU transfer path. Pinned DRAM and
simulator-backed host shared memory are peer retention pools behind one object
directory. This implementation uses vLLM v0.29.0 and LMCache v0.5.5 with the
single-GPU dense/GQA configuration recorded in ``examples/l2_hybrid/``.

Data ownership and completion
-----------------------------

Each complete immutable KV chunk has one retained owner. Placement first tries
DRAM, then CXL. When necessary, it evicts only unreferenced CXL objects in LRU
order, using stable object identity to break ties. It does not migrate objects.
Hybrid references are explicitly released; elapsed time cannot expire an active
reader, writer, or transfer reservation.

The server copies actual KV bytes between HBM and the selected host buffer.
For CXL objects it also submits metadata to CXLMemSim. It publishes a STORE or
returns a usable LOAD completion only after both CUDA and the model finish.
Per-operation events and deferred replies allow other submissions to proceed.
Cancellation does not release storage that submitted copies still own.
Each lookup owns its logical read references. LOAD exchanges those references
for physical transfer pins before submission; cancellation releases only the
unused lookup share. Hybrid session cleanup is explicit, without a TTL.

CXLMemSim backing is host shared memory, not physical CXL hardware. Reports
keep measured GPU transfers, modeled delay, and CPU observation times separate.

Running the bounded reference workload
--------------------------------------

The source checkout includes an environment/build runbook, frozen workload,
correctness probes, and a bounded service runner. From the common workspace
containing the three existing repositories, with the pinned environment set up
as described in ``LMCache/examples/l2_hybrid/RUN.md``::

   LMCACHE_VERIFY_KV_CONTENT=0 LMCache/.venv/bin/python \
     LMCache/examples/l2_hybrid/run_service.py \
     --workload LMCache/examples/l2_hybrid/workload-001.json \
     --build CXLMemSim/build-l2-hybrid --mode hybrid --trace light \
     --out LMCache/examples/l2_hybrid/evidence/service-NEW

The output directory must be new. The runner creates private simulator and
LMCache servers, streams twelve actual requests, drains references, and stops
its owned processes. Other modes are ``hbm``, ``dram``, and ``dram_equal``.
The ``off``, ``light``, and ``heavy`` observation levels separate performance
measurements from scheduler tracing and native torch profiling.

Manual server configuration
---------------------------

The MP server accepts ``--l1-hybrid-config /path/to/hybrid.json``. For a simulator
publishing at least 84 MiB of payload, an example configuration is::

   {
     "dram_bytes": 29360128,
     "cxl_bytes": 88080384,
     "bulk_control": "/example_bulk",
     "modeled_control": "/example_modeled",
     "client_library": "/path/to/libcxlmemsim_modeled_client.so",
     "policy": "dram_first"
   }

Control names must match the live simulator. Set ``--l1-size-gb 0.02734375``,
``--no-l1-use-lazy``, an empty ``--shm-name``,
``--supported-transfer-mode lmcache_driven``, and ``--eviction-policy noop``.
Hybrid mode owns capacity-driven eviction and rejects incompatible lazy,
GDS/DAX, and additional-L2 configurations.

The worker connector must set
``lmcache.mp.require_terminal_completion=true`` in its extra configuration.
Heartbeat loss then preserves pending GPU ownership rather than synthesizing a
completion. A permanently lost terminal reply requires restarting the affected
engine/backend. The connector follows vLLM's ``kv_load_failure_policy``:
``recompute`` retries through native computation, while ``fail`` returns an
error for that request. Neither path may issue a duplicate final block release.

Evidence boundaries
-------------------

``probe_hybrid.py`` supports forced placement, partial-copy failure, pressure,
and cancellation diagnostics. Independent KV byte checks deliberately
synchronize and are excluded from service performance runs. The normal
workload uses no HBM reset or forced placement.

``analyze_service.py`` checks cross-run output equality, logprob tolerance,
native cache counters, reference balance, and receive-before-use ordering.
``service_timeline.py`` exports a Perfetto-compatible host/model timeline.
Exact GPU boundaries are absent from light traces; native receive adoption is
an interval bounded by the scheduler call. Consult ``STATUS.md`` and the raw
reports for verified coverage. ``PROFILING.md`` indexes independent copy,
contention, heldout, and full-model compute samples. ``ACCEPTANCE_REPORT.md``
records the requirement-by-requirement audit and measurement limits.
