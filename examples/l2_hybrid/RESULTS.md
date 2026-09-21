# Measured results

2026-09-21. Fixed vLLM v0.29.0, LMCache v0.5.5, Qwen2.5-7B BF16 on GPU 1.
CXL is CXLMemSim-backed SHM with real GPU KV copies. These are simulated-backing
results, not physical-CXL performance or a completed simulator calibration.

## Final service controls

After the lookup-ownership fix, `controls-002` repeats the frozen twelve-request
workload in three fresh processes per mode with rotated order. All 144
trace-off requests pass. Including three light and one heavy observation runs,
all 192 outputs have exact token equality and maximum selected-token logprob
delta 0.0 (predeclared tolerance 0.02). Service runs use no forced placement,
HBM reset, content-check synchronization or failure/lookup-delay hook.

| Mode | Retention quota | Mean TTFT across processes | Range of process means |
| --- | --- | ---: | ---: |
| HBM only | none | 114.30 ms | 111.97–115.67 ms |
| HBM + DRAM | 28 MiB DRAM | 120.66 ms | 116.09–125.00 ms |
| Hybrid | 28 MiB DRAM + 84 MiB CXL | 121.57 ms | 119.78–124.60 ms |
| Equal-total DRAM | 112 MiB DRAM | 120.11 ms | 119.42–120.68 ms |

Hybrid reduces actual prompt recomputation in this workload but shows no clear
TTFT advantage. In the first repeat, locally computed prompt tokens are
6,307 / 6,179 / 4,579 / 4,835 respectively out of 7,779. Retention outcomes can
vary with upstream periodic DRAM eviction; every process's counters are kept.
Connector block holds also change HBM reuse: 1,472 tokens for HBM-only versus
832 for connector modes. Hybrid uses DRAM-first/CXL-only LRU, whereas DRAM
controls retain upstream periodic LRU. These are whole-implementation
comparisons, not isolated memory-medium latency measurements.

Three light runs average 128.34 ms TTFT, 5.57% above hybrid trace-off. The
single heavy run averages 130.62 ms, 7.45% above trace-off. These percentages
include run-to-run variation and are observed perturbation, not an isolated
causal estimate. Heavy profiling is for structure, with one repeat. Full
per-process mean/p95 TTFT, TPOT, throughput, counters, and output comparison:
`evidence/controls-002/analysis-observation.json`. Twelve mechanism requests
per process do not establish statistical significance or production behavior.

Earlier `controls-001` and `service-hybrid-*` results predate the ownership
fix and remain separate. They are not pooled with these final measurements.
No claim of an isolated host is made; GPU 0 was occupied when work began and
no unrelated process was stopped.

## State and correctness

Every final light/heavy service trace naturally restores from DRAM and CXL,
observes two CXL objects with two logical readers, and rejects eviction during
reader ownership. Logical references transfer into physical pins before LOAD;
DMA intervals can serialize despite overlapping logical demand.

Observed native protection peaks at 84 of 95 usable HBM blocks, with no
preemption. All protected blocks are released after requests drain; 94 valid
unreferenced blocks remain cached. Event footers report zero drops/errors.
`service-lease-light-001/timeline.json` contains 1,119 Perfetto events. Its
stage-cost report preserves inclusive host intervals with unmeasured pieces
explicitly null.

`revalidation-001` reruns actual HBM overwrite/eviction/revisit, partial STORE,
partial LOAD with native fail/recompute policies, and cancellation after LOAD
or STORE submission. The cancel audits check held blocks, competitor non-reuse,
one terminal release, independent KV bytes, exact retry output and drained
reservations. Diagnostic slow model completion does not assert CUDA remained
active at the cancellation instant. The separate lookup-cancel audit proves
two unused references released and zero submitted copies/computation batches.

## Independent profiling

Copy training: 24 cases, 168 measured samples; heldout: 16 cases, 112 samples;
separate heavy copy profiler: 24 cases, 168 samples. All cases verify actual
bytes and zero final reservations. Four size-median fits are frozen before
heldout, with no service-TTFT input. Eight heldout checks have maximum absolute
relative error 8.46%, without refitting or a general accuracy claim.

Heavy CUPTI activities show same-direction copies serializing and actual
bidirectional/copy-compute overlap. Independent full-model compute has 36
zero-reuse requests, 14,976 computed prompt tokens, 576 output tokens and
396 shape-checked GPU ranges. The final service heavy trace separately maps
164 native batches to 164 GPU ranges covering eight shapes. See PROFILING.md
for raw paths, full coverage, cold setup costs and timing limitations.

## Checks and limits

The affected suite passes 333 tests and 15 subtests. The log retains 154
OpenTelemetry warnings and a CUDA IPC producer teardown warning from the
broader test suite. Real service/probe evidence independently verifies explicit
reference drain. Changed-file lint/type checks and the strict full Sphinx build
pass; whole-repository CI is not claimed.

Real fatal GPU loss, physical CXL, multi-GPU/TP, other layouts and production
lifetime scaling remain unverified. Missing terminal replies deliberately fail
closed. Hash-only, IPC-only, policy-only, gather/scatter-only and other
unseparated costs are null with containing intervals documented. The results
support the amended bounded acceptance, not universal prediction or superiority.
