# L2-hybrid implementation status

Updated: 2026-09-21. Implementation and measured evidence are in the existing
repositories. The bounded amended acceptance and final affected-path replay
pass. ACCEPTANCE_REPORT.md is the authoritative requirement ledger; the final
source/evidence manifests and archive verification are in `evidence/delivery-final/`.

## Scope and frozen versions

The user's instruction supersedes the handoff's physical-CXL/no-simulator and
empty-workspace clauses: use CXLMemSim and develop in the existing fixed vLLM
and LMCache folders. Simulated backing is host SHM with real GPU payload
copies. Modeled CXL delay and measured CUDA/host time remain separate.

| Repository | Release/base | Full SHA |
| --- | --- | --- |
| vLLM | v0.29.0 | 98dff2a81d747d1dba01a47f939f48c3526d4206 |
| LMCache | v0.5.5 | 05a013b29da78cf2321b9b46ec5039dde2fb0bb0 |
| CXLMemSim | existing checkout | 992d5f522e7dd4c117340841d9193cd2152107f8 |

No source versions were upgraded and no commits were made. Pre-existing
untracked vLLM benchmarks/docs/results/tests are untouched and excluded from
this task's source overlay. The native vLLM binary is from the exact official
v0.29.0 CUDA-13.0 release wheel, hash verified. Editable source metadata's
`+cu133` suffix reflects the local build toolkit, not a nightly wheel.
LMCache native extensions were built from the fixed source. Actual import/ABI
paths, dependency lock, model/tokenizer hashes and build commands are retained
in `evidence/bootstrap/` and RUN.md.

## Implemented behavior

- One immutable-object directory manages peer pinned-DRAM and registered CXL
  SHM pools. DRAM-first placement and legal CXL-only LRU eviction use explicit
  ownership, quotas, stable ties and rollback. No payload stages through a
  separate DRAM L2 cache; HBM copies may coexist with one retained host owner.
- Actual native LMCache gather/scatter copies use the selected host pointer.
  Per-operation CUDA completion plus CXLMemSim completion gates publication
  and LOAD replies. Deferred replies leave submission workers available.
- A lookup owns exactly its read reservations. LOAD exchanges its own logical
  references for physical pins; cancellation releases only unused references.
  Pins remain until submitted work drains. Hybrid sessions have explicit
  cleanup; no TTL expires active ownership. Request-ID tombstones reject late
  lookup RPCs until server close and retain no payload references.
- Native vLLM AsyncScheduler decisions are preserved. The light observer only
  brackets calls and reads state. Worker glue handles fail/recompute and
  cancellation with one terminal release. Heartbeat loss cannot fake DMA
  completion; permanently lost terminal replies fail closed and need restart.
- Bytewise KV diagnostics, JSONL loss/error footers, native schedule/state
  events, request/output records, Perfetto timelines and independent profiling
  retain real identities and measured/model/unknown timing boundaries.

## Evidence index

All paths here are relative to `evidence/`. Earlier failed attempts remain
available; passing names below refer to reports and raw data, not names alone.

| Gate | Evidence |
| --- | --- |
| Exact mapped SHM -> real GPU round trip | `transport-002/` (4 KiB, 1 MiB, 8 MiB, independent bytes/SHA) |
| Native miss/HBM/DRAM restore | `dram-probe-invariant-002/` (715 input, 16 output, exact tokens/logprobs and KV bytes) |
| Forced and natural CXL restore | `cxl-forced-001/`, `hybrid-natural-001/`, latest normal service traces |
| Overwritten HBM, CXL eviction and revisit | `revalidation-001/pressure/` |
| Partial STORE/LOAD failures, both LOAD policies | `revalidation-001/{store-failure,load-failure-recompute,load-failure-fail}/` |
| Cancel before LOAD | `hybrid-cancel-lookup-001/mp/cancellation-audit.json` |
| Cancel after LOAD/STORE submission | `revalidation-001/{cancel-load,cancel-store}/mp/cancellation-audit.json` |
| Three fresh processes per control mode | `controls-002/` (144 trace-off requests) |
| Three light and one heavy observation run | `service-lease-light-001/`, `revalidation-001/{light-002,light-003,heavy-001}/` |
| All 192 service outputs and trace invariants | `controls-002/analysis-observation.json` |
| Independent copy/competition/heldout | `copy-{training,holdout,heavy}-001/`, `copy-analysis-001.json` |
| Independent full-model compute | `compute-independent-001/coverage.json` (36 requests, 396 native batches) |
| Inclusive host stage costs and timeline | `service-lease-light-001/{stage-costs,timeline}.json` |
| Final machine-readable acceptance checks | `delivery-audit.json` |

Current performance conclusions are in RESULTS.md; primitive coverage and
unseparated costs are in PROFILING.md. No physical-CXL, broad prediction
accuracy, production representativeness or algorithm-superiority claim is made.

## Validation and review

`bootstrap/runtime-stage4.log`: 333 tests and 15 subtests pass, including real
private CXLMemSim/CUDA allocation, two-reader ownership, cancellation before
status consumption, late lookup replay, pin-safe eviction, adapter terminal
release and legacy lookup/session paths. The exact command is in
`runtime-stage4-command.json`. The log retains 154 OpenTelemetry deprecation
warnings and a CUDA IPC producer teardown warning from the broader suite;
this is not a warning-free test-suite claim. Real engine probes separately
verify all session/reader/writer/physical reservations drain.

`bootstrap/pre-commit-delivery.log` passes changed-file formatting, lint,
mypy and headers;
Rust hooks are explicitly skipped. Strict full Sphinx (`-E -a -W`) passes in
`bootstrap/sphinx-stage4.log`. Existing duplicate labels, missing includes,
malformed directives and missing toctree entries were repaired so the whole
site builds cleanly. `sphinx-render-audit.json` checks the new rendered page.

REVIEW.md records separate Standards/Spec reviews. All hard findings were
fixed. Two optional naming/teardown-coupling heuristics remain documented.

## Failed attempts and fixes retained

- Release CXLMemSim tests removed side-effecting calls inside `assert` under
  NDEBUG, causing SIGBUS. `cxl-sigbus.strace` proves missing ftruncate. Assertions
  are now enabled for six contract-test targets; Release and Debug pass.
- Non-invariant BF16 initially produced exact tokens but excessive logprob
  deltas. Native `VLLM_BATCH_INVARIANT=1` removes that difference without
  relaxing the predeclared 0.02 tolerance. A separate script ID-correlation
  failure was repaired; both failed attempts remain.
- Pressure attempts without correct startup capacity or proven physical
  overwrite remain failed. The successful probe proves old HBM blocks were
  overwritten before revisit.
- A partial LOAD failure exposed duplicate connector release under native
  fail policy. Worker retirement now waits for submitted work and reports one
  final release. Red tests and the original engine assertion remain retained.
- The first cancellation probe reset too early after frontend output. It now
  waits for native block retirement instead of forcing it.
- Review found a lookup-cancellation leak before LOAD. The red public-path
  test is `lookup-cancel-red.log`; the lease fix passes both status-polled and
  unpolled cancellation, two-reader/pin pressure and real Qwen cancellation.

`controls-001`, prior light/heavy runs and `bootstrap/source-stage3/` are earlier
revision evidence. They are retained for provenance, not pooled into final
performance results. Independent copy and HBM-only compute profiling are
unaffected by the later lookup-ownership fix. The 27 runtime source files
in `bootstrap/runtime-stage4-source.json` are checked unchanged before delivery.

## Environment and reproduction boundaries

Use RUN.md from the common workspace and fresh output directories. GPU 1 is
the only task GPU. GPU 0 was occupied initially; no unrelated process was
stopped. This is not an isolated-host benchmark. All project temporary files
are under `/home/zjhuang/.tmp/l2-hybrid`; private POSIX SHM names are runtime
resources cleaned by successful owning launchers.

The initial unused `l2_hybrid/` checkout directory remains because automatic
approval review rejected its recursive deletion. It is not used or packaged.
Six zero-byte SHM names from the original crashing upstream tests remain as
failure artifacts; successful task probes clean their private names.

Physical CXL, fatal real device-loss recovery, multi-GPU/TP, other KV layouts,
and long-lived production memory growth are not validated. Lost terminal
replies deliberately retain protected resources instead of claiming safe
completion. The finite request-ID tombstone set grows for a server lifecycle.
