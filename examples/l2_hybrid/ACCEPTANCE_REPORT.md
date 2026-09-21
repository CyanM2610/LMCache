# Acceptance report

2026-09-21. The bounded single-GPU implementation passes the amended functional,
service, event and profiling gates below. The user's instruction explicitly
replaces physical CXL with CXLMemSim and requires development in the existing
fixed vLLM/LMCache directories. It does not convert simulated delay into
physical-CXL evidence. Original requirements are preserved in
[spec/ACCEPTANCE.md](spec/ACCEPTANCE.md).

The final machine audit checks 23 service/mechanism runs, 192 normal-service
requests, 39 independent KV byte checks, explicit reference drain, pool bounds,
terminal child exits and continuity of 27 runtime source files:
[delivery-audit.json](evidence/delivery-audit.json). This report also records
limits that cannot be reduced to a Boolean audit.

## Deployment and actual path

| Requirement | Status | Evidence and interpretation |
| --- | --- | --- |
| Fixed public release base, reproducible extension | Pass | [RUN.md](RUN.md), [source/ABI](evidence/bootstrap/source-abi.json), [dependency lock](evidence/bootstrap/requirements.lock), [inventory/model hashes](evidence/bootstrap/inventory-before-install.json), final source overlay against the three full SHAs in STATUS.md |
| Actual CXL backing ownership | Superseded; simulated path passes | [transport report](evidence/transport-002/report.json), [primitive provenance](evidence/copy-training-001/report.json): exact simulator SHM descriptor, process mapping, registered pointer and resident NUMA pages; physical CXL was not available or claimed |
| Payload directly between selected backing and HBM | Pass for amended path | Native LMCache transfer calls use exact registered SHM pointers; [transport bytes](evidence/transport-002/report.json), [forced CXL content](evidence/cxl-forced-001/mp/report.json), latest force-CXL failure/cancel probes. Model RPC transfers metadata, with zero model-handler payload bytes |
| New process can start/run/drain/stop | Pass | [12 terminal control processes](evidence/controls-002/runs.json), [9 terminal replay processes](evidence/revalidation-001/runs.json), fresh private server names and commands in each report; RUN.md does not require historical logs or personal skills |

vLLM is v0.29.0, LMCache v0.5.5; no upgrade. Native vLLM is the hash-verified
official CUDA-13.0 release wheel with editable source imports, and LMCache
extensions were built from source. The local metadata suffix `+cu133` describes
the toolkit, not a nightly binary. CXLMemSim has isolated SHM naming and a
contract-test NDEBUG fix; production simulator modeling is otherwise unchanged.

## Functional scenarios and invariants

| Scenario | Status | Raw/derived evidence |
| --- | --- | --- |
| Full miss with real generation and KV identity | Pass | [final control comparison](evidence/controls-002/analysis-observation.json), original tokenized workload and per-run reports |
| HBM prefix reuse, matching compute/copy counts | Pass | HBM-only native counters: 1,472 local-hit tokens and 6,307 computed of 7,779; [DRAM diagnostic](evidence/dram-probe-invariant-002/report.json) explicitly separates miss/HBM/remote stages |
| DRAM restore after HBM release | Pass | DRAM diagnostic independent bytes; [latest service trace](evidence/service-lease-light-001/analysis.json) has real DRAM LOAD followed by native adoption and schedule |
| CXL restore without a DRAM retained copy | Pass | One-owner directory, force-CXL diagnostics, actual host pointer/slot/object-key events; [latest normal service](evidence/service-lease-light-001/report.json) naturally restores CXL |
| Revisit after physical HBM overwrite and CXL eviction | Pass | [final pressure report](evidence/revalidation-001/pressure/mp/report.json) requires overwritten original physical blocks and CXL-only eviction before revisit |
| Partial prefixes, versions and multiple readers | Pass | All final traces observe two objects with two logical readers; no eviction while referenced. Public two-reader/pin-pressure test and immutable-version/pool tests pass in [runtime suite](evidence/bootstrap/runtime-stage4.log) |
| STORE/LOAD in-flight source and destination protection | Pass | Completion tests, actual transfer metadata, [LOAD cancellation audit](evidence/revalidation-001/cancel-load/mp/cancellation-audit.json), [STORE cancellation audit](evidence/revalidation-001/cancel-store/mp/cancellation-audit.json) prove held-block non-reuse and one terminal release |
| Cancellation before LOAD submission | Pass after fix | [lookup audit](evidence/hybrid-cancel-lookup-001/mp/cancellation-audit.json): two owned references released, zero submitted copies or compute batches, exact successful retry; red regression retained |
| Partial copying failure and recovery | Pass | [STORE](evidence/revalidation-001/store-failure/mp/report.json), [LOAD recompute](evidence/revalidation-001/load-failure-recompute/mp/report.json), [LOAD fail](evidence/revalidation-001/load-failure-fail/mp/report.json). Expected failed request is retained; later recovery succeeds |
| Request end and cleanup | Pass | Every final hybrid status has zero sessions, lookup jobs, reader/writer/temporary/physical reservations and pending completion; cached valid objects/backing are separately counted |

Independent content checks compare defined KV bytes selected with a separate
PyTorch block-index path, excluding undefined padding. Debug synchronization
is confined to diagnostic runs. All 192 final service outputs are exactly equal;
chosen-token logprob differences are 0.0 under native batch-invariant kernels,
within the original 0.02 tolerance. Earlier non-invariant and script failures
are retained rather than relabeled passing.

| Invariant | Status and evidence |
| --- | --- |
| Quotas and backing are distinct; no over-capacity allocation | Pass: allocator rollback and all-protected refusal tests; per-pool used/capacity/backing checks in the audit |
| Readers/copies prevent overwrite and eviction | Pass: explicit logical lease and physical pins, two-reader public-path tests and real cancellation block audit |
| No READY before real visibility and modeled completion | Pass: per-operation CUDA event plus model fence; controlled fence tests and actual partial-copy failures |
| One retained DRAM/CXL owner per immutable version | Pass: directory duplicate-version tests and exclusive allocator placement; HBM copies remain legal |
| No duplicate publication or residual refs/deadlock | Pass for exercised paths: failure/cancel/close tests, terminal process records and final status audit; valid retained cache is not called a leak |
| No policy access to future requests/output lengths | Pass by implementation review: replacement sees current directory, references, medium, recency and stable identity only |

## Workload, controls and costs

| Requirement | Status and coverage |
| --- | --- |
| Generated/frozen inputs, seeds, arrival rules and output budgets | Pass: [service workload](workload-001.json), [compute workload](compute-workload-001.json), both generators and per-run input SHA/text/token records |
| No reuse / shared prefixes / competition / burst / idle revisit | Pass: 36 independent no-reuse inputs plus 12-request six-family service workload; [frozen protocol](EXPERIMENT_PROTOCOL.md) |
| Active capacity fits but all idle content cannot remain | Pass: 84 observed protected HBM blocks of 95 usable, zero native preemptions, pressure/overwrite proof. Generic wait states are not mislabeled placement wait |
| Fixed DRAM plus CXL, and equal total capacity control | Pass: HBM-only, 28 MiB DRAM, 28+84 MiB hybrid, 112 MiB DRAM; same model/HBM/inference limits, three independent processes each |
| Normal path after diagnostic hooks are disabled | Pass: 144 trace-off and 48 observed requests with empty diagnostic-hooks list |
| Trace-off vs light/heavy disturbance | Pass: [RESULTS.md](RESULTS.md), three light repeats and one structural heavy run; no superiority/statistical significance claim |
| Independent copy sizes, directions and competition | Pass: 24 training cases/168 measured samples, 16 heldout/112, 24 heavy/168; same-direction, both bidirectional orders and compute-copy competition |
| Raw primitive samples and independent heldout checks | Pass: [frozen fit](evidence/copy-training-001/frozen-fit.json) precedes heldout process; [analysis](evidence/copy-analysis-001.json) verifies hashes/time order and keeps all errors; no service TTFT/ranking used |
| Independent native compute and actual batch shapes | Pass: [coverage](evidence/compute-independent-001/coverage.json), 36 requests, zero reuse, 14,976 prompt tokens, 576 outputs, 396 native batch/GPU range matches |

## Event and timing contract

[SCHEMA.md](SCHEMA.md) defines fields, units, clocks, nulls and denominators.
Environment/configuration, public/engine/session IDs, actual inputs/arrivals,
core enqueue, output/failure/cancellation, immutable object/chunk/block/slot
identity, effective/actual bytes, ownership and completion phases are retained.
Batch events include native prefill/decode work and contexts. Final reports
retain mean/p95 TTFT, defined TPOT, coalesced output chunks, completion/throughput,
cache denominators, bytes, actual recomputation and observed protection peaks.

The light trace distinguishes demand/submission, CUDA observation, logical
model completion, reply readiness, native adoption interval and first resulting
schedule. Exact DMA or GPU-start times remain null. Heavy service profiling
joins 164 actual native batches/GPU ranges; independent copy CUPTI profiling
captures actual DMA/kernel intervals and overlap. Model and CUDA intervals are
concurrent and are not blindly added.

[PROFILING.md](PROFILING.md) records cold context/map/touch/register/allocation,
raw branch durations, competition and actual overlap. [Stage costs](evidence/service-lease-light-001/stage-costs.json)
retain inclusive lookup/hash/directory/status-consumption and submission/fence/
reply intervals. Hash-only, IPC-only, policy-only, gather/scatter-only, poll-only,
publication-only and output-serialization-only costs are unseparated and null;
the containing intervals are stated. Native model ranges include preparation
and profiler disturbance, not pure kernel service time.

## Checks, review and delivery

- [Affected regression suite](evidence/bootstrap/runtime-stage4.log): 333 tests
  and 15 subtests pass. It retains 154 OpenTelemetry warnings and one CUDA IPC
  producer teardown warning; full-repository CI is not claimed.
- [Changed-file hooks](evidence/bootstrap/pre-commit-delivery.log) pass headers,
  format, lint and mypy, with explicit Rust skips.
  [Strict full Sphinx](evidence/bootstrap/sphinx-stage4.log) passes without
  warnings; rendered-page checks are retained. Release/Debug simulator contract
  suites pass. [REVIEW.md](REVIEW.md) separates Standards and Spec findings/fixes.
- [RUN.md](RUN.md) gives rebuild, generation, execution, analysis and replay
  commands. [Final source manifest](evidence/delivery-final/source-manifest.json)
  lists fixed bases and hashes; `source-overlay.zip` contains all added/changed
  task sources, while repository patches contain tracked changes.
- [Evidence manifest](evidence/delivery-final/evidence-manifest.json) indexes raw
  inputs, events, profiles, failed attempts and reports. `snapshot_delivery.py`
  verifies archive CRCs and source SHA-256. Base repositories, dependencies,
  model weights and unrelated work are excluded and identified by the runbook.

Physical CXL and real fatal GPU-loss recovery are unverified. TP/multi-GPU,
other layouts, arbitrary larger workloads and production lifetime scaling are
outside this single-GPU gate. A permanently lost terminal reply deliberately
retains protected resources and requires restart. Request-ID tombstones grow
until server close. These limits do not imply measured physical CXL, universal
prediction accuracy, simulator calibration or algorithm superiority.
