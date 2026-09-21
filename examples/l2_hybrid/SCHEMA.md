# Event and result schema, version 1

Files are UTF-8 JSON or JSONL. Byte counts are integers; MiB in prose means
2^20 bytes. Token counts exclude undefined KV padding. Public request IDs and
vLLM internal IDs are distinct: analysis requires one exact
`<public-id>-<8 hexadecimal digits>` mapping in the recorded core-enqueue events.

## Workload and client result

`workload-001.json` freezes text, exact tokenizer IDs, canonical compact-JSON
SHA-256 of those IDs, request dependencies, think times, output budgets and
inference settings. Every runner copies and hashes it. Later arrivals use the
actual completion of the declared predecessor(s), not future output lengths.

`report.json.requests` records planned and actual arrival, raw observed stream
chunks, generated token IDs, selected-token logprobs, first token-ID output,
first nonempty-text output, completion and error state. Times ending in `_ns`
are integer Linux CLOCK_MONOTONIC instants from the observing process. All
processes run on the same host/time namespace. Wall-clock metadata is explicitly
named `wall_ns`; it is not substituted for an occurrence timestamp.

TTFT ends at first nonempty text. TPOT is (last token-ID observation - first)
divided by (output token count - 1). Stream chunks may coalesce tokens, so exact
per-token ITL is not inferred. The nearest-rank p95 is across twelve requests;
the comparison table reports each process mean and its three-process range.
`native_metric_rows` are upstream callback increments; analysis excludes warmup
using the measured origin and sums prompt work/cache counters. Local cache hit
denominator is native queried prompt tokens; connector hit denominator is
native tokens queried after local HBM reuse. Zero denominator stays a count,
not a fabricated hit rate.

## LMCache event bus

`events.jsonl` starts with process/clock anchors and ends with event count,
drops, subscriber errors and remaining queue depth. An event has `event`,
`session_id`, original occurrence `timestamp_s`, `sequence`, and `metadata`.
Byte keys are typed hexadecimal objects. `timestamp_s` is Unix float64 seconds
from the event publication or native host callback, not file-write time.

`hybrid.access` identifies the exact immutable object key, host pointer, pool,
slot offset, physical allocation bytes, full host transfer bytes, effective
destination bytes, engine block IDs, chunk index and transfer key. A LOAD may
copy an entire host chunk while scattering only its HBM-missing suffix. Stable
DRAM/CXL residency is exclusive; HBM copies may coexist. Addresses are local
to the process and must be read with mapping/offset/configuration evidence.

`demand_ns`, model-begin observation, CUDA-completion observation and logical
completion observation are separate host boundaries. `modeled_service_ns`,
`modeled_queue_ns`, and `modeled_complete_ns` come from CXLMemSim. They are not
physical DMA costs. CUDA and modeled work overlap and must not be summed.
Exact DMA endpoints are null with a reason in lightweight traces.

`hybrid.completion` distinguishes submission, CUDA observation and server
`reply_ready`. Server reply readiness is not scheduler adoption. Early
development traces used `adopted` for this same server boundary; STATUS.md
identifies them and prevents that older name from becoming an adoption claim.

`l1.read.reserved/finished`, writes, and evictions carry object keys. The current
single-rank service has one logical lock per read reservation. Analysis counts
overlapping read lifetimes and rejects eviction before release. Physical DMA
pins are separate and reported as `transfer_reservation_count`. Retained valid
objects, used bytes, quotas and allocator backing are separately reported.

`hybrid.lookup_lease` records `reserved` and `released` phases with exact keys,
host `observed_ns`, and logical reference counts. LOAD atomically exchanges
its own logical references for physical transfer pins. Thus logical readers
can finish before DMA; transfer pins continue protecting the source. END_SESSION
releases only the unconsumed share. Late duplicate lookup/cleanup cannot borrow
another request's references. Request-ID tombstones last until server close;
they retain no payload or reader reservations.

## Native scheduler observation

`scheduler-<pid>.jsonl` has its own schema/header and loss/error footer. The
observer inherits all native AsyncScheduler decisions. Core enqueue,
`schedule`, and `scheduler_output_processed` bracket actual native calls with
`begin_ns` and `end_ns`. Batch IDs relate returned scheduler outputs to later
processing. Schedule records include request IDs, computed context, scheduled
prompt/decode token counts and physical blocks. They do not assert GPU start.

Receive adoption occurs inside the recorded native output-processing interval;
the exact internal instant is null. Analysis requires successful logical LOAD
completion before adoption, then adoption before the first resulting schedule.
GPU kernels and CPU launch correlation are available only in the separate raw
native heavy profiler trace; no absolute GPU spans are invented in light traces.

State records separate running blocks, remote reservations and finished request
holds, along with actual protected blocks and unreferenced retained HBM blocks.
Category sets can overlap for shared prefixes and must not be blindly summed.
Peaks are observations at native-call boundaries, not an exhaustive accounting
of internal transients. Generic waiting status is not relabeled as placement
or capacity wait without supporting native evidence.

`service_timeline.py` produces Chrome/Perfetto JSON. Pure monotonic boundaries
share one origin. Older wall-clock bus events use the header's paired anchor
and are explicitly marked approximate. Modeled service has its own track.
Raw native profiler traces retain their original clock/correlation metadata.

## Independent profiling

Copy `plan.json` freezes cases and repetition counts before execution.
`report.json.samples` retains warmups (`repeat < 0`) and measured repetitions,
operation sets, bytes, host monotonic bounds, per-stream CUDA event offsets in
microseconds, and separately modeled timing. Content checks occur outside the
timed sample. Cold stage fields end in `_ns`; actual mapping/NUMA ownership is
under `provenance`. `active_after_drain` must be zero.

`frozen-fit.json` records training SHA, freeze wall time, OLS coefficients and
declared heldout sizes. The heldout process records the fit SHA and its later
start time. `copy-analysis-001.json` retains all measured distributions and
heldout errors. Heavy sample activities use original CUPTI timestamps/durations
in microseconds, stream IDs and byte counts; interval intersections determine
actual device overlap. A zero observed overlap is distinct from a null boundary.

`compute-independent-001/coverage.json` maps public and engine IDs to actual
native batch rows, per-request context/work, first/repeated shape role, and
heavy GPU model ranges. Those ranges include preparation/inference and profiler
overhead. `stage-costs.json` gives inclusive host stage samples with separately
unmeasured costs explicitly null. See PROFILING.md for containing intervals,
coverage, fit limits and cold/warm interpretation.
