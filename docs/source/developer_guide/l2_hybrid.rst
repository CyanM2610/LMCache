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

Confirmed cleanup and bounded arrivals
--------------------------------------

Hybrid lookup cleanup is confirmed against the original server generation.
Result consumption preserves demand until worker LOAD or explicit END; END
releases logical reservations without releasing physical transfer pins. A
shutdown with unconfirmed cleanup fails closed. Impossible allocation previews
aligned free ranges and legal victims before evicting valid objects.

The service runner accepts ``--max-model-len``, ``--num-gpu-blocks``,
``--max-num-seqs`` and ``--max-num-batched-tokens`` (defaults 1024, 96, 2, 1024).
Root arrival adds ``root_arrival_s`` and ``think_s`` to the run origin. Dependent
requests use actual predecessor completions plus think time. Inputs above the
configured context are rejected. Async scheduling remains enabled. The pinned
model has 32768 positions; longer inputs require a separately validated model
or position configuration. ``NATIVE_FIXES.md`` distinguishes repaired-source
validation from the original measured results.

Minimal profiling canaries
--------------------------

``run_service.py --trace light --worker-timing`` adds an observation-only worker
adapter that records separate execute/sample host and current-stream CUDA
envelopes. It preserves native async scheduling and output objects. Uncorrelated
output-copy streams remain unknown. ``profile_native_transfer.py`` probes the
compiled object-group gather/scatter executor using registered host backing;
byte verification is outside measurement. JSONL-only MP observation now exports
completed native phase samples without enabling heavy tracing or metrics.

``analyze_canaries.py`` checks frozen sample completeness, unique worker-step
correlation, independent byte checks and OFF/ON overhead. See
``PROFILING_CANARIES.md`` for the measured small-shape coverage and overhead;
these are instrument-validation/training data, not a new blind heldout.

Pending fences and longer training inputs
-----------------------------------------

``probe_pending_fence.py`` deliberately holds a real CUDA copy behind a device
semaphore. It verifies END/cancellation with an incomplete fence, protected
registered source/destination buffers, and both CUDA-first and model-cancel-first
completion ordering. This is an explicit fault-injection state-machine test,
not a claim of active copy-engine occupancy or a performance sample.

``generate_context_training.py`` creates independent complete synthetic 1K, 4K,
8K and 16K prompts, with one/two-request cohorts and first/repeated shape labels.
Use the frozen plan with ``run_service.py --max-model-len 32768`` and explicit
``--num-gpu-blocks 2560 --max-num-batched-tokens 1024 --max-num-seqs 2``. Both
OFF and ON retain the light scheduler observer; ON adds ``--worker-timing``.
Resolved scheduler limits, actual batches and nonzero root arrivals are audited.

The timing adapter correlates the existing native output-copy completion event
and delegates native future consumption. Readiness observations and the inclusive
host consumption envelope do not measure output DMA duration or CPU-active time.
No new stream wait is added. Inputs beyond the model's position limit require a
separate explicit model/position decision and independent numerical validation;
the D2 YaRN candidate is preparation only and is not activated by these runners.

Native execution phases and stable replay
------------------------------------------

Replayer version 2 dispatches eligible requests by planned time and frozen plan
index, using one timer. Each dispatch awaits native client submission, while
inference and output delivery continue asynchronously. Children still follow
actual predecessor completion plus think time. This defines external tie order;
native batching continues to depend on actual engine arrival and progress.

``run_service.py --trace light --phase-timing`` enables optional native v2 probes
around the sample output fork, copy stream after its original dependency,
postprocess tail, original output-fence wait, and CPU output conversion. Host
spans retain wall, thread CPU and process CPU clocks. Empty ``no_forward`` steps
record control work separately. The probes add CUDA event records and queries;
the original copy-event wait and native scheduler retain their semantics.

``analyze_execution_costs.py --off RUN_OFF --on RUN_ON --out NEW_AUDIT.json``
requires complete phase/error accounting, per-batch output joins, poll-to-adoption
receipts, and actual MP transfer/allocator/model-begin trees. Raw transfer plans
record gather/scatter shapes, host-pool addresses, staging segments and operation
identities. A producer event may be shared by a save call's requests; MP STORE
submission still occurs separately for each request.

Outer worker, RPC, transfer, model and nested phase intervals overlap. Copy-stream
elapsed time includes launch gaps and is not DMA-active time. Async client CPU
clock deltas can include other tasks. Do not add these durations as independent
service costs. ``NATIVE_EXECUTION_003.md`` links the bounded training evidence and
remaining transport/queue/contention gaps; it does not declare a calibrated model.

D2 numerical acceptance gate
----------------------------

``prepare_d2_numeric.py`` freezes complete supplied documents and synthetic short
inputs, hashes local model shards, and checks an independent NumPy YaRN formula
against pinned vLLM and Transformers CPU implementations. ``run_d2_numeric.py``
uses separate native and HF/PyTorch processes, full reference prefill and complete
past, retaining all vocabulary logprobs for four independently generated tokens.
``analyze_d2_numeric.py`` requires distinct backends, frozen input/config hashes,
normalized complete vectors, exact greedy IDs and the predeclared probability
thresholds. A failed short gate prevents longer inference and the capacity canary.

The native004 short gate failed its top10-union tolerance despite matching greedy
IDs. An operator diagnosis confirmed different BF16 RMSNorm casting order before
attention; it did not establish exclusive attribution of the full-model error.
The reference and thresholds were retained. See ``NATIVE_D2_004.md`` for the
failure evidence and deferred long-context/capacity work.

Bounded host-cost attribution
-----------------------------

``run_service.py --capture-environment`` records actual worker threads, affinity,
normalization dispatch and resolved settings before and after measurement.
``CustomOp.get_dispatch_info`` reads the stored callable without redispatch.
Configured IR priority must be distinguished from executed kernel evidence.

Native005 completed six short HBM-only OFF/phase processes and two explanatory
CPU+CUDA profiles. The optional ``L2_HYBRID_MODEL_DISPATCH=1`` hook in the heavy
processes records model-call host boundaries, preserving native output, waits
and exceptions. Profiler durations are explanatory and cannot be used as OFF
service costs. API/kernel correlations show asynchronous overlap; stream event
envelopes include launch gaps, and output waiting wall depends on prior GPU work.

``NATIVE_HOST_COSTS_005.md`` documents exact shapes, actual RMSNorm dispatch,
observer costs, unavailable hardware occupancy counters, a disclosed C-phase
metadata source-freeze exception, and the remaining cost-identification limits.
The native004 failed numerical gate remains unchanged.

Native retirement and physical page reuse
-----------------------------------------

The pinned async ``kv_both`` configuration both drops preempted stale output
and defers physical block freeing. Retirement captures the global scheduled-step
fence, which can be later than the request's own last step. Nonempty output
consumption advances the processed fence and drains old allocations before
handling sampled output. Rescheduling eligibility does not mean pages are free.

``L2_HYBRID_RETIREMENT_WITNESS=1`` enables optional scheduler/allocator snapshots
with immutable submitted generations. It preserves native return values and
waits; callback errors invalidate evidence without replacing native execution.
The existing preemption flush/device-sync calls have separate optional host
spans. Future observation does not consume replies or import completion events.

``NATIVE_PREEMPTION_006.md`` describes two complete service witnesses at95 and83
usable pages, six observed retirements/reuses, output parity and explicit gaps
for a later global fence and active MP work during preemption. Snapshot overhead
is material; these traces do not supply OFF cost estimates or BLIS acceptance.
