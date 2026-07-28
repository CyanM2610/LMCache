# CXL shared tier for multiprocess vLLM

The CXL shared tier is an opt-in LMCache multiprocess data path for immutable,
GPU-visible KV objects. vLLM still owns HBM allocation and scheduling; LMCache
owns external residency, placement, transfer tickets, and CXL completion.

## Boundaries

- `VLLMDataPlaneAdapter` translates registered cache contexts and block IDs
  into versioned, pointer-free transfer plans.
- `CXLSharedTierModule` owns engine registration, STORE/RETRIEVE lifecycle,
  aliases, and request cancellation.
- `MultiResidencyDirectory` owns independent DRAM and CXL residency state.
- `TicketManager` binds a lookup decision to an exact residency generation and
  read lease before the connector reports an external hit.
- `ModeledCompletionCoordinator` composes the real CUDA copy with CXLMemSim's
  metadata-only modeled branch. Success requires both branches.

The directory, tickets, and policy layer never receive CUDA pointers. Native
pointers exist only inside `RegisteredRegionView` and the immediate CUDA
executor call.

## STORE lifecycle

STORE reserves each configured target independently, marks it WRITING, and
publishes it READY only after its composite completion succeeds. Required and
optional targets have separate outcomes. The source HBM remains retained until
every target has completed or failed, so one target cannot outlive its source.

No DRAM staging object is allocated by this path. A DRAM residency is a
separate GPU-visible backing region, not a staging copy for CXL.

## RETRIEVE lifecycle

Policy lookup constructs a fresh observation, chooses a source, and binds a
generation-specific ticket before returning a positive match. RETRIEVE
validates that ticket and fully overwrites the destination from the selected
residency.

If the first transfer fails, the request may bind and try one different READY
residency. The alternate attempt also performs a full overwrite. Cancellation
never falls back. A missing alternate or a second failure invalidates the
destination block IDs and returns a load error; the worker adapter then hands
those blocks to vLLM recomputation. A request has one fallback budget even when
it contains multiple chunks.

## CXLMemSim integration

`cxlmemsim_shm` validates the server-published, page-aligned data region.
`CXLMemSimModelClient` negotiates `gpu_direct_modeled_access_v1`, registers the
same region, reserves modeled service before CUDA launch, and reports CUDA
terminal state afterward. Logical completion is the later successful branch,
not the sum of their durations. Error, cancellation, and timeout fail closed.

## Observability and bounded state

Operation and policy events contain only stable IDs, descriptors, timings, and
reason codes. Per-operation logging is DEBUG-level. In-memory evidence,
latency samples, cancellation tombstones, and retired-lease tombstones are
bounded; active operations and live leases remain exact until terminal.
