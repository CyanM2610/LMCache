# Frozen service protocol

Protocol written before the three-process control series. Input artifact:
`workload-001.json` (each run records its SHA-256). There are 12 measured
requests, six independent 640-token prefix families, three possible suffixes,
16 output tokens, seed 74123, maximum concurrency two, and one small unmeasured
warmup. Each non-root arrival follows actual predecessor completion plus the
frozen think time. No generated output is used in placement decisions.

All modes use one Qwen2.5-7B BF16 engine on GPU 1, TP=1, the pinned native
AsyncScheduler, eager execution, batch-invariant kernels, maximum context 1024,
batch token limit 1024, 96 HBM blocks of 16 tokens (one null block), and greedy
generation. Content-check synchronization, forced placement, HBM reset, and
failure injection are disabled. Original topology and version amendments in
STATUS.md apply: CXL is CXLMemSim-backed host shared memory, not physical CXL.

| Mode | DRAM quota | CXL simulated quota | Comparison |
| --- | ---: | ---: | --- |
| hbm | 0 | 0 | Engine-only reference |
| dram | 28 MiB | 0 | Fixed DRAM reference, upstream retention behavior |
| hybrid | 28 MiB | 84 MiB | Added capacity, DRAM-first/CXL-only LRU |
| dram_equal | 112 MiB | 0 | Same total retention quota as hybrid |

The CXL server publishes 128 MiB backing, of which only the configured 84 MiB
quota is allocatable. Report these separately. Hybrid policies are not tuned
to the measured ranking. DRAM uses the upstream periodic LRU implementation;
this is a baseline implementation comparison, not an isolated memory-medium
latency experiment. The independent primitive suite addresses copy costs.

The preliminary normal-policy light trace completed all 12 requests. It
observed 84 protected HBM blocks at native schedule/output boundaries, fitting
within 95 usable blocks, while total distinct prefixes cannot all remain in
HBM. These are observed boundary peaks, not a claim to capture every internal
allocation transient. It showed natural DRAM/CXL restores and no preemption.

Three fresh processes per mode, sequentially on GPU 1, with rotated order:

1. hbm, dram, hybrid, dram_equal
2. dram, hybrid, dram_equal, hbm
3. hybrid, dram_equal, hbm, dram

Each process has a 600 s outer timeout, including engine startup; measured
requests have a 180 s aggregate deadline. Retain every failure and stop the
series if a runner fails or cannot cleanly terminate. Never stop an unrelated
process. GPU 0 has an unrelated workload; GPU 1 is reserved by this driver
only for each sequential run. This is not an isolated-host benchmark.

The primary control series uses trace-off. Native metric callbacks and client
stream timestamps are retained equally in all modes. Separately run light
observer repeats and one bounded heavy profiler run; compare their latency
range with trace-off and report disturbance. The observer subclasses native
AsyncScheduler solely to bracket inherited calls; it never changes decisions.

TTFT is actual client arrival to first nonempty-text stream output. TPOT is
(last token-ID observation minus first)/(output token count minus one). Raw
stream chunks preserve coalescing: they are not invented exact per-token
arrival times. P95 uses nearest-rank over 12 requests. Throughput includes the
frozen think/idle gaps. Startup and the small warmup are excluded.

Compare output token IDs exactly and chosen-token logprobs with predeclared
absolute tolerance 0.02 across modes and repeats. Preserve failures, request
inputs, actual arrivals, raw native counters, configuration, and source hashes.
Report each process and its spread; twelve mechanism requests do not establish
statistical significance, production representativeness, or general superiority.
