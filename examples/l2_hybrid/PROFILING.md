# Independent profiling and coverage

All measurements use GPU 1, an A100 PCIe 40GB. CXL payload backing is the
CXLMemSim-published host SHM mapping. `/proc/self/maps` and `numa_maps` in each
copy report prove the actual allocation; both training buffers were on NUMA
node 1. CPU affinity was unrestricted within the recorded 112 CPUs. These
measurements describe this host and simulated backing, not physical CXL.

## Copy and competition samples

`profile_copies.py` uses CUDA Driver asynchronous H2D/D2H APIs on the exact
registered buffer. Every case validates actual destination bytes outside its
timed interval. The model RPC exchanges metadata only. CUDA events bracket
each branch; those durations include within-stream/copy-engine wait. Host
submission, observed completion, model RPC, modeled queue and modeled service
remain separate. All model reservations drain at process end.

| Evidence | Sizes | Cases | Warmups / measured per case |
| --- | --- | ---: | ---: |
| `copy-training-001` | 4 KiB, 1 MiB, 14 MiB, 28 MiB | 24 | 2 / 7 |
| `copy-holdout-001` | 7 MiB, 21 MiB | 16 | 2 / 7 |
| `copy-heavy-001` | training sizes, separate profiler process | 24 | 2 / 7 |

Training has 16 solo cases (two media, two directions, four sizes), four
shared-link cases (read/read, write/write, both read/write orders), three
copy/compute cases and one compute-only baseline. Compute competition runs
eight native BF16 GEMMs of shape `[512,3584,3584]`; heldout uses M=384. This is
one projection primitive, not whole-model inference. Competition cases use
14 MiB in training and 7 MiB heldout.

14 MiB CUDA event medians in training, microseconds:

| Backing | LOAD | STORE |
| --- | ---: | ---: |
| pinned DRAM | 725.376 | 856.448 |
| registered CXLMemSim SHM | 725.408 | 867.008 |

The near-equal LOAD values are expected for two registered host buffers on
the same physical host path. Modeled CXL delay is not presented as a measured
physical latency.

`fit_copy_costs.py` fits four unweighted size medians per medium/direction.
`copy-training-001/frozen-fit.json` was frozen before the heldout process;
analysis verifies its SHA, training SHA and wall-time ordering. It never reads
service TTFT or policy ranking. Eight heldout solo checks retain prediction
errors; maximum absolute relative error is 8.46%, with no refit or declared
general accuracy threshold. Raw coefficients and all errors are machine-readable
in `copy-analysis-001.json`.

The heavy run records actual CUPTI memcpy/kernel activities. Analysis maps
them to per-stream case annotations, verifies copied bytes and operation types,
and intersects actual device intervals. Median overlap in microseconds:

| Concurrent operations | Actual overlap |
| --- | ---: |
| DRAM LOAD + CXL LOAD | 0 |
| DRAM STORE + CXL STORE | 0 |
| DRAM LOAD + CXL STORE | 1,053.19 |
| DRAM STORE + CXL LOAD | 1,022.43 |
| DRAM LOAD + GEMM | 611.30 |
| CXL LOAD + GEMM | 609.89 |
| CXL STORE + GEMM | 792.99 |

Same-direction copies serialize on this measured path. The zero overlap is
an observation, not an omitted case. CUDA-event overlap alone is not used as
proof of simultaneous DMA.

Cold setup is recorded separately from steady samples. Training observed
context initialization 162.50 ms, SHM mmap 0.043 ms, SHM first touch 71.49 ms,
SHM registration 9.26 ms, DRAM allocation/pinning 11.49 ms, DRAM first touch
25.76 ms and GPU allocation 0.507 ms. These are single cold observations for
the configured allocations, not per-copy constants.

## Independent full-model compute

`compute-workload-001.json` contains 36 unique first-block inputs, contexts
128/512/640 for training and 384 heldout, requested concurrency 1/2, three
occurrences per shape and 16 generated tokens per request. First occurrence
is separated from two repeats; the engine has already loaded and warmed up.
No compute regression is fitted, and no TTFT ranking is used for a fit.

`compute-independent-001/coverage.json` checks every request against the frozen
token input, zero cache reuse, 14,976 actually computed prompt tokens and
576 generated tokens. It joins 396 native batches to the heavy GPU ranges;
10 native shape signatures and 32 context/temperature groups are retained.
Requested concurrency does not imply simultaneous prefill: observed pairs
include mixed prefill/decode batches. Actual context and scheduled token counts
are recorded per member instead of assuming the requested shape occurred.

Native model ranges include worker preparation and inference and incur heavy
profiler overhead. They are not pure kernel service times. Original profiler
clock/correlation metadata and compressed raw CPU/GPU traces are retained.
The frontend trace logged an external-callback thread warning; worker GPU
ranges and their schedule mapping are complete. The service worker trace
does not capture the separate MP process's payload DMA.

## Service boundaries and limits

`analyze_stages.py` retains inclusive lookup/hash/directory/status-consumption,
native submission, submission-to-CUDA-observation, and observation-to-reply
intervals. `analyze_service.py` separately brackets scheduler adoption and
first scheduling; light traces leave exact GPU use/DMA endpoints null.
The model and CUDA paths overlap and must not be serially added.

Hash-only, policy-only, IPC-only, gather/scatter-kernel-only, poll-only,
publication-only and output-serialization-only costs remain unseparated.
Their containing intervals and nulls are explicit in the stage report. Direct
copy primitives do not include the native gather/scatter kernels. Cold setup
and representative independent compute/copy cases are measured; arbitrary
layouts, multi-GPU, larger batches, device loss and cross-host generalization
are outside this coverage.
