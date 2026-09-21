# Reproduction runbook

Run from `/home/zjhuang/cxl_offloading`. Fixed source versions and the user's
CXLMemSim scope amendment are recorded in `STATUS.md`. No service-performance
claim is supported by the transport diagnostic below.

## Temporary files and device isolation

```bash
export TMPDIR=/home/zjhuang/.tmp/l2-hybrid
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/LMCache:$PWD/vllm"
export XDG_CACHE_HOME="$TMPDIR/cache"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export HF_HUB_OFFLINE=1
export VLLM_BATCH_INVARIANT=1
export LMCACHE_TRACK_USAGE=false
mkdir -p "$TMPDIR"
nvidia-smi --id=1
```

Use only an idle GPU. GPU 0 was occupied when work began. Do not terminate
other users' processes. GPU/SHM calls must run outside the tool filesystem
sandbox. Probe servers use unique POSIX names and are stopped by their owning
launcher. The files in `evidence/` are permanent results, not temporary files.

## Dependency environment

The completed installation commands are captured in `bootstrap` evidence:

```bash
/home/zjhuang/miniforge3/envs/cxl-lmcache/bin/uv venv \
  --python /home/zjhuang/miniforge3/bin/python3.12 LMCache/.venv
/home/zjhuang/miniforge3/envs/cxl-lmcache/bin/uv pip install \
  --python LMCache/.venv/bin/python vllm==0.29.0 lmcache==0.5.5 \
  pytest ruff setuptools_scm ninja 'setuptools>=77,<81' wheel
```

The following source installs were also completed successfully:

```bash
/home/zjhuang/miniforge3/envs/cxl-lmcache/bin/uv pip install \
  --python LMCache/.venv/bin/python \
  -r vllm/requirements/build/cuda.txt -r LMCache/requirements/build.txt
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 NVCC_THREADS=1 \
  BUILD_WITH_CUDA=1 /home/zjhuang/miniforge3/envs/cxl-lmcache/bin/uv pip install \
  --python LMCache/.venv/bin/python --no-build-isolation --no-deps -e LMCache
VLLM_USE_PRECOMPILED=1 VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1 \
  VLLM_PRECOMPILED_WHEEL_LOCATION="$TMPDIR/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl" \
  /home/zjhuang/miniforge3/envs/cxl-lmcache/bin/uv pip install \
  --python LMCache/.venv/bin/python --no-build-isolation --no-deps -e vllm
```

Obtain that wheel from the exact URL in `evidence/bootstrap/vllm-pypi.json` and
verify SHA-256 against `vllm-wheel.sha256` before extracting it. Do not omit the
wheel-location override: the upstream default may use nightly. The captured
environment lock is `evidence/bootstrap/requirements.lock`; an independent
rebuild must replace its absolute editable paths with the target checkout paths.
Keep `PYTHONPATH` as above so a workspace-level `vllm/` namespace directory does
not shadow the editable package. Actual Python and native imports are recorded
in `source-abi.json`. The `+cu133` editable distribution metadata reflects the
local build toolkit; the extracted vLLM wheel and torch use CUDA 13.0.

For the
first transport probe only, `vllm/.venv/bin/python` supplied Python 3.12 and
LMCache package initialization; the probe used `libcuda.so.1` directly, no
vLLM engine and no LMCache CUDA extension. Full-engine gates must use the new
environment and verified fixed-release binaries.

## Rebuild CXLMemSim

The server and native client were rebuilt from the existing repository. The
local compiler is GCC from the `cxl` toolchain; libbpf headers are the existing
Debian development headers. Rebuilding on another host requires equivalent
GCC/C++20, CMake, spdlog and libbpf headers, not this exact filesystem layout.

```bash
/home/zjhuang/miniforge3/envs/cxl/bin/cmake \
  -S CXLMemSim -B CXLMemSim/build-l2-hybrid \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/home/zjhuang/miniforge3/envs/cxl/bin/x86_64-conda-linux-gnu-cc \
  -DCMAKE_CXX_COMPILER=/home/zjhuang/miniforge3/envs/cxl/bin/x86_64-conda-linux-gnu-c++ \
  -DCMAKE_PREFIX_PATH=/home/zjhuang/miniforge3/envs/cxl \
  '-DCMAKE_CXX_FLAGS=-isystem /home/zjhuang/cxl_offloading/CXLMemSim/build/deps/libbpf/usr/include' \
  -DCXLMEMSIM_BUILD_MICROBENCHMARKS=OFF \
  -DCXLMEMSIM_ENABLE_RDMA=OFF -DCXLMEMSIM_ENABLE_SLUGALLOCATOR=OFF
/home/zjhuang/miniforge3/envs/cxl/bin/cmake \
  --build CXLMemSim/build-l2-hybrid -j 4 \
  --target cxlmemsim_server cxlmemsim_modeled_client
```

## Diagnostic and protocol tests

```bash
LMCache/.venv/bin/python \
  LMCache/examples/l2_hybrid/probe_transport.py \
  --build CXLMemSim/build-l2-hybrid \
  --out LMCache/examples/l2_hybrid/evidence/transport-NEW
CXLSIM_TEST_BUILD=CXLMemSim/build-l2-hybrid \
  LMCache/.venv/bin/python -m unittest discover \
  -s LMCache/tests/v1/distributed -p test_cxl_sim_client.py -v
```

The probe runs exactly three sizes (4 KiB, 1 MiB, 8 MiB), one real H2D and D2H
per size. It creates a 16 MiB server backing (one 4 KiB header page is excluded
from payload capacity). Native calls have a 2 s IPC timeout, startup is bounded
at 15 s and modeled completion at 5 s. Wrap the entire command in `timeout 60s`
if running unattended. Failed output directories are retained; each repeat
requires a fresh output name.

The diagnostic uses `cuMemHostRegister_v2`, `cuMemcpyHtoDAsync_v2`, and
`cuMemcpyDtoHAsync_v2` on the same mapped data address. SHA-256 and a bytewise
comparison cover only the explicitly initialized range. Its CPU initialization,
poisoning and verification copies are **debug operations**, not payload staging
in the transfer path. Synchronization is intentional for content verification.

## Result schema version 1

`report.json` is the raw diagnostic artifact; `server.log` records simulator
operations and `payload_copy_bytes=0` for the metadata handler. `physical_cxl`
is always false. Addresses are process-local; mapping name, offset, capacity,
API, GPU and native generation identify the actual route. Byte fields exclude
the backing header. `active_after_drain` counts client transfer reservations.

All `*_ns` host instants use Linux CLOCK_MONOTONIC. CUDA event elapsed time is
a device duration measured around the actual copy; it is not assigned a fake
absolute host timestamp. The absolute DMA fields are null with an explanation.
`modeled_service_ns`, `modeled_queue_ns` and `modeled_complete_ns` are simulator
values, kept distinct from observed submission/completion and CUDA event time.
Logical readiness waits for both successful data completion and modeled
completion. Durations of these overlapping branches must not be summed.

These samples do not establish steady-state bandwidth, independent process
variation, simultaneous transfer contention, inference correctness, or TTFT.
Those are subsequent required gates.

## Pure vLLM smoke

The first official-wheel run and the subsequent source/ABI checks passed.
To repeat from the editable source environment:

```bash
timeout 600s LMCache/.venv/bin/python LMCache/examples/l2_hybrid/smoke_vllm.py \
  --model /home/zjhuang/models/Qwen2.5-7B-Instruct \
  --out LMCache/examples/l2_hybrid/evidence/vllm-smoke-NEW
```

This launches one engine, one real request, greedy generation capped at 16
tokens, BF16, 1024 physical KV blocks of 16 tokens, max context 4096 and max
batch 1024 tokens. No LMCache path is enabled at this first engine gate.

## MP DRAM correctness gate

```bash
HF_HUB_OFFLINE=1 LMCACHE_TRACK_USAGE=false VLLM_BATCH_INVARIANT=1 \
  LMCACHE_VERIFY_KV_CONTENT=1 timeout 600s LMCache/.venv/bin/python \
  LMCache/examples/l2_hybrid/probe_dram.py \
  --model /home/zjhuang/models/Qwen2.5-7B-Instruct \
  --out LMCache/examples/l2_hybrid/evidence/dram-probe-NEW
```

The launcher owns a private localhost LMCache MP server and one vLLM engine.
It freezes 512 MiB pinned DRAM, chunk size 256, the same 1024 HBM blocks and
three serial requests: cold miss, HBM hit, HBM-reset DRAM restore. It saves the
literal prompt, actual token IDs, generated IDs and selected-token logprobs.
Expected token IDs must match exactly; logprob absolute tolerance is 0.02.
Server startup is limited to 60 s and each reference-drain check to 30 s.
The outer 600 s timeout bounds initial JIT compilation too.

`LMCACHE_VERIFY_KV_CONTENT=1` enables independent PyTorch block selection,
source/destination byte comparison and SHA-256. It supports the observed dense
GQA layout NL x [NB,BS,NH,CS] (here [1024,16,4,256] per layer, 28 layers), and
fails explicitly for unsupported geometries. This is a synchronizing diagnostic;
never compare its duration with trace-off service TTFT. The HBM reset is also
a diagnostic hook; normal service workloads must not use it.

The successful reference is `evidence/dram-probe-invariant-002`. Both earlier
failed runs remain with their logs: the first isolates BF16 shape-dependent
numerical variation; the second exposes and fixes the diagnostic's request-ID
mapping. No acceptance tolerance was increased to make either failure pass.

The server's opt-in `--event-jsonl PATH` captures source occurrence timestamps,
typed byte hashes, chunk/token bindings, physical block allocation events and
transfer identities. Existing output files are rejected. Header/footer records
declare schema, wall/monotonic clock anchors, queue drops and subscriber errors.
Timestamps are Unix float64 seconds from publication or the native stream host
callback, **not** file-write times and **not** exact DMA endpoints. Debug content
events correlate by transfer key and internal request ID. The probe maps public
IDs to the pinned vLLM convention `<public-id>-<8-hex nonce>`, requires one match,
and saves both IDs in each request record. Independent primitive profiling and
native model-range analysis are described below and in PROFILING.md.

Relevant regression suites (all project temporary files under TMPDIR):

```bash
CXLSIM_TEST_BUILD="$PWD/CXLMemSim/build-l2-hybrid" \
  LMCache/.venv/bin/python -m pytest \
  LMCache/tests/v1/distributed/test_cxl_sim_client.py \
  LMCache/tests/v1/distributed/test_hybrid_policy.py \
  LMCache/tests/v1/distributed/test_l1_manager.py \
  LMCache/tests/v1/multiprocess/test_kv_content_check.py \
  LMCache/tests/v1/multiprocess/test_lmcache_driven_transfer_skip.py \
  LMCache/tests/v1/mp_observability/test_event_bus.py \
  LMCache/tests/v1/mp_observability/test_config.py
```

## Hybrid correctness gates

Use the same environment above, GPU 1 and `LMCACHE_VERIFY_KV_CONTENT=1`:

```bash
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/probe_hybrid.py \
  --build CXLMemSim/build-l2-hybrid \
  --model /home/zjhuang/models/Qwen2.5-7B-Instruct \
  --policy dram_first --pressure \
  --out LMCache/examples/l2_hybrid/evidence/hybrid-pressure-NEW
```

Omit `--pressure` for the mixed-pool three-request gate. Use `--policy force_cxl`
for the forced CXL gate. Add `--failure store_partial` or `--failure load_partial`
for one-shot failure after the first submitted chunk. LOAD failure defaults to
native `--load-failure-policy recompute`; `--load-failure-policy fail` expects
one error output and then validates a successful later request. The server
continues to own submitted buffers until actual CUDA and modeled completion.
These options are diagnostics, not performance policies.

## Normal service and repeated controls

`workload-001.json` is frozen. Regeneration must use a new filename:

```bash
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/generate_workload.py \
  --model /home/zjhuang/models/Qwen2.5-7B-Instruct --out workload-NEW.json
LMCACHE_VERIFY_KV_CONTENT=0 LMCache/.venv/bin/python \
  LMCache/examples/l2_hybrid/run_service.py \
  --workload LMCache/examples/l2_hybrid/workload-001.json \
  --build CXLMemSim/build-l2-hybrid --mode hybrid --trace light \
  --out LMCache/examples/l2_hybrid/evidence/service-NEW
```

Modes: `hbm`, `dram`, `hybrid`, `dram_equal`. Observation: `off`, `light`,
`heavy` (native torch profiler). `run_controls.py` accepts `--workload`,
`--build`, `--out` and executes the predeclared twelve-process trace-off series.
See `EXPERIMENT_PROTOCOL.md` for capacities, arrival rules, repetitions and
timing definitions. Never run two GPU probes concurrently.

The light observer inherits the pinned native AsyncScheduler and only reads
its returned decisions and public state. It records actual core enqueue,
scheduled prefill/decode shapes, physical blocks, protection snapshots, and
brackets completion processing. It does not change allocation, priority, or
request order. `scheduler-<pid>.jsonl` has its own loss/error footer. The
normal service never resets HBM, forces placement, or enables content sync.

Hybrid integration requires the worker option
`lmcache.mp.require_terminal_completion=true`. It prevents heartbeat loss from
turning a pending transfer into a false release. Lost terminal replies retain
resources and require restarting that engine/backend; they are not successful
drain evidence. Lazy offload, GDS/DAX adapters and additional L2 tiers are
rejected by hybrid server configuration in this initial dense/GQA path.

Cancellation probes use `--policy force_cxl --cancel load` (or `store`) and
`--reference .../controls-002/r0-hbm/report.json`. The launcher slows only the
relevant modeled completion for this diagnostic. `--cancel lookup` instead
delays the lookup ACK after ownership is acquired and cancels before LOAD.
Normal service configurations leave both hooks disabled. `verify_cancellation.py RUN`
checks held blocks, competing schedules, byte checks and the unique final
release; pass the outer probe directory containing `mp/`.

To analyze the completed comparison and a light trace:

```bash
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/analyze_service.py \
  --controls LMCache/examples/l2_hybrid/evidence/controls-001 \
  --run LMCache/examples/l2_hybrid/evidence/service-hybrid-light-001 \
  --out /home/zjhuang/.tmp/l2-hybrid/analysis-NEW.json
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/service_timeline.py \
  LMCache/examples/l2_hybrid/evidence/service-hybrid-light-001 \
  /home/zjhuang/.tmp/l2-hybrid/timeline-NEW.json
```

Repeat `--run` to include other light/heavy runs. Analysis checks exact outputs,
0.02 logprob tolerance, event loss, reader balance, protected eviction and
receive-before-use ordering. Definitions and null reasons are in SCHEMA.md;
the current bounded conclusions are in RESULTS.md.

## Independent primitive and compute profiling

Each output path must be new. Run the following commands sequentially on the
same idle GPU, with the environment above. Keep fit freeze before heldout:

```bash
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/profile_copies.py \
  --build CXLMemSim/build-l2-hybrid --out evidence-NEW/copy-training
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/fit_copy_costs.py \
  evidence-NEW/copy-training evidence-NEW/copy-training/frozen-fit.json
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/profile_copies.py \
  --build CXLMemSim/build-l2-hybrid --out evidence-NEW/copy-holdout \
  --holdout --frozen-fit evidence-NEW/copy-training/frozen-fit.json
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/profile_copies.py \
  --build CXLMemSim/build-l2-hybrid --out evidence-NEW/copy-heavy --heavy
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/analyze_copies.py \
  --training evidence-NEW/copy-training --holdout evidence-NEW/copy-holdout \
  --heavy evidence-NEW/copy-heavy --fit evidence-NEW/copy-training/frozen-fit.json \
  --out evidence-NEW/copy-analysis.json
LMCACHE_VERIFY_KV_CONTENT=0 LMCache/.venv/bin/python \
  LMCache/examples/l2_hybrid/run_service.py \
  --workload LMCache/examples/l2_hybrid/compute-workload-001.json \
  --build CXLMemSim/build-l2-hybrid --mode hbm --trace heavy \
  --out evidence-NEW/compute
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/analyze_profile.py \
  evidence-NEW/compute evidence-NEW/compute/profile-analysis.json
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/analyze_compute.py \
  --workload LMCache/examples/l2_hybrid/compute-workload-001.json \
  --run evidence-NEW/compute --out evidence-NEW/compute/coverage.json
```

To regenerate independent compute inputs, `generate_compute_workload.py` takes
`--model` and `--out`. Keep the original frozen inputs for exact replay.
`analyze_stages.py --run SERVICE --out NEW.json` writes inclusive host cost
samples from a light trace. No service TTFT is used to train copy costs.

## Final affected-path replay

After the ownership fix, `controls-002` repeats the same frozen protocol as
`controls-001`. `revalidation-001` contains two more light runs, one heavy run,
pressure/overwrite, partial STORE failure, both LOAD failure policies, and
submitted LOAD/STORE cancellation. The independent lookup cancellation is
`hybrid-cancel-lookup-001`; its explicit audit shows zero submitted copies.

```bash
LMCACHE_VERIFY_KV_CONTENT=0 LMCache/.venv/bin/python \
  LMCache/examples/l2_hybrid/run_controls.py \
  --workload LMCache/examples/l2_hybrid/workload-001.json \
  --build CXLMemSim/build-l2-hybrid --out evidence-NEW/controls
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/revalidate.py \
  --workload LMCache/examples/l2_hybrid/workload-001.json \
  --build CXLMemSim/build-l2-hybrid \
  --reference evidence-NEW/controls/r0-hbm/report.json \
  --out evidence-NEW/revalidation
```

The replay driver freezes its plan before running and stops after a failure.
Each child has a 600 s deadline; only its own process group may be terminated.
It sets byte-check synchronization for diagnostics and disables it for service
measurements. Preserve failed directories and choose new names for retries.

`evidence/bootstrap/runtime-stage4-command.json` records the exact affected
pytest suite. `runtime-stage4.log` retains all test warnings. Strict Sphinx is
built with a separate documentation environment under the project TMPDIR;
that environment does not modify inference dependencies.

## Delivery and verification

The final archive is `/home/zjhuang/cxl_offloading/l2-hybrid-delivery-20260921.zip`.
Its checksum and verified counts are in
`evidence/delivery-final/delivery-result.json`, written after ZIP closure.
The archive contains the task's source overlay, tracked diffs, immutable input
artifacts, raw event/profile data, failed attempts, reports and manifests.
It excludes the full base repositories, model weights, environments, build
trees and pre-existing unrelated files.

For a fresh checkout, obtain the exact base SHAs in STATUS.md, extract
`evidence/delivery-final/source-overlay.zip` into their common parent, then
follow the build/environment steps above. Patches contain tracked changes;
the overlay also contains new files. Do not apply both over an already modified
checkout. Original handoff files remain byte-identical to their manifest.

The evidence manifest covers files present before its creation and excludes
its own recursive hash. The outer archive's CRCs and every overlay source SHA
are verified by `snapshot_delivery.py`; its final result is outside that archive.
To reproduce the acceptance audit or create another package, use new outputs:

```bash
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/audit_delivery.py \
  --workspace "$PWD" --evidence LMCache/examples/l2_hybrid/evidence \
  --out /home/zjhuang/.tmp/l2-hybrid/delivery-audit-NEW.json
LMCache/.venv/bin/python LMCache/examples/l2_hybrid/snapshot_delivery.py \
  --workspace "$PWD" \
  --out LMCache/examples/l2_hybrid/evidence/delivery-NEW \
  --archive "$PWD/l2-hybrid-delivery-NEW.zip"
```
