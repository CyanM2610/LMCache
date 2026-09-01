# Multi-instance HotPrefix control plane

This branch adds the shared-Host control plane used by the vLLM HotPrefix
policy. It is intentionally separate from LMCache's normal incremental STORE
policy.

## Configuration

Start each MP server with the same policy parameters:

```bash
lmcache_server \
  --hotprefix-host-capacity-bytes 1073741824 \
  --hotprefix-frequency-threshold 10 \
  --hotprefix-aging-interval 50 \
  --hotprefix-lease-ttl-seconds 30
```

Enable canonical mode in the vLLM connector extra config:

```json
{
  "lmcache.mp.hotprefix_enabled": true,
  "lmcache.mp.hotprefix_instance_id": 0,
  "lmcache.mp.hotprefix_promotion_budget_bytes": 67108864
}
```

Every serving instance needs a stable, non-negative instance ID. When omitted,
the connector generates one for the scheduler process.

The first faithful baseline requires LMCache `chunk_size` to equal the vLLM
scheduler block size. The connector fails at startup on a coarser chunk instead
of silently bypassing admission for an unrepresentable physical victim.

## State and protocol

- `HOT_PREFIX_ACCESS` merges per-instance ordered event streams into a Global
  Host Prefix Tree. The path is truncated to complete LMCache chunks so its
  prefix IDs exactly match Local tree publication and physical residency
  endpoints. Duplicate `(instance_id, local_event_seq)` events are idempotent.
- `HOT_PREFIX_ADMIT` uses server-authoritative Global frequency and clock. A
  scheduler-proposed generation is adopted by every MP server, avoiding
  generation drift after partial failure.
- `HOT_PREFIX_PUBLISH` and `HOT_PREFIX_ABORT` complete or roll back an immutable
  residency reservation. Replacement victims are restored if a STORE aborts.
- `HOT_PREFIX_CANDIDATES` returns only READY sources. The adapter intersects
  `(prefix_id, generation, size)` across all servers.
- `HOT_PREFIX_ACQUIRE`/`HOT_PREFIX_RELEASE` bind a promotion to an exact
  generation. Multiple instances may hold concurrent read leases; leased
  residencies are not replacement candidates. `HOT_PREFIX_RENEW` extends an
  active lease; abandoned leases expire server-side.

With `hotprefix_enabled`, normal per-step incremental payload STORE is
suppressed. An actual allocation attempt identifies the cold physical victim
and is deferred while its chunk-aligned full prefix is pinned, admitted, and
sent through the existing STORE executor. The residency is published only after
every worker succeeds; otherwise admission is rolled back and allocation may
retry without the shared copy. Promotion uses
the existing RETRIEVE executor under a renewable generation ticket, and vLLM
publishes the detached target blocks only after every worker succeeds.

On-demand native fetch remains available and is accounted separately from
background promotion. Promotion has one transaction per instance, slices the
prefix into LMCache-aligned per-step budget ranges, and guarantees at least one
chunk as a starvation quantum. Each background RETRIEVE range first performs
the standard LOOKUP/WAIT handshake so its physical objects hold ordinary L1
read locks in addition to the generation lease; the lease alone only prevents
policy replacement. Multi-tier placement, adaptive budgets, and global flow
arbitration are intentionally left for experiments.

## Physical retention ownership

The HotPrefix directory and the distributed StorageManager share one physical
generation contract. A canonical STORE completion binds the generation to every
local `ObjectKey` produced for its chunks, object groups, and server rank. The
binding is published only after the device-stream completion callback finishes
all L1 writes and atomically acquires non-expiring retention pins.

Retention pins are distinct from TTL read locks. Generic L1 eviction, clear,
and in-place update skip pinned objects; HotPrefix replacement explicitly
retires a generation. Objects shared by several prefix generations are pinned
once and deleted only after the final generation reference is retired. If a
reader is still active, deletion becomes pending and completes after its final
read lock is released.

Replacement victims remain physically pinned while the candidate STORE is in
flight. Publication deletes the committed logical victims; abort deletes only
the candidate payload and restores the victims. A failed or late stream
completion cannot republish an aborted generation. Finally, every L1 deletion
callback is reverse-mapped from `ObjectKey` to affected generations so a forced
physical loss tombstones the logical residency before it can be acquired again.

## Observability

Every Global control handler emits a paired `HOTPREFIX_CONTROL_START/END`
EventBus operation with total, lock-wait, and handler-body duration. Admission
and residency transitions emit bounded decision/state events. Existing
stream-timed MP STORE/RETRIEVE events carry a fixed `purpose` classification
(`eviction_store`, `promotion`, or `foreground_fetch`) so GPU copy time is not
measured twice with a CPU clock.

Prometheus aggregation excludes request, prefix, generation, and ticket IDs.
Those fields are available only in sampled traces, correlated by
`HOTPREFIX_RUN_ID` and a per-operation ID. Logical residency bytes/generations,
active leases, physical retained keys, and physical tombstone/publication counts
are pull gauges; they cannot drift permanently if the EventBus drops an event.
