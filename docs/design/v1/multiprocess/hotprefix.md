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
  Host Prefix Tree. Duplicate `(instance_id, local_event_seq)` events are
  idempotent.
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
chunk as a starvation quantum. Multi-tier placement, adaptive budgets, and
global flow arbitration are intentionally left for experiments.
