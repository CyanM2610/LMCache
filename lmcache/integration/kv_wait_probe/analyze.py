# SPDX-License-Identifier: Apache-2.0
"""Evidence-based disjoint TTFT partition and raw Perfetto/CUPTI export.

Host intervals are observations, not counterfactual savings. GPU service is
assigned only through same-process CUDA API correlation inside our NVTX range.
Missing boundaries remain unknown; they are never replaced with callback time.
"""

# Standard
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import sqlite3
import statistics


def read_jsonl(path: Path) -> list[dict]:
    """Read complete records; fail rather than silently ignore trace corruption."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def partition(start: int, end: int, intervals: list[tuple]) -> list[dict]:
    """Cover TTFT once; larger priority owns overlap, all holes stay unknown."""
    if end < start:
        raise ValueError("negative TTFT")
    active = [(max(start, a), min(end, b), name, priority)
              for a, b, name, priority in intervals if a < b and a < end and b > start]
    boundaries = sorted({start, end} | {v for a, b, _, _ in active for v in (a, b)})
    output = []
    for a, b in zip(boundaries, boundaries[1:]):
        covered = [(priority, name) for lo, hi, name, priority in active
                   if lo <= a and hi >= b]
        name = max(covered)[1] if covered else "unexplained"
        if output and output[-1]["stage"] == name:
            output[-1]["end_ns"] = b
        else:
            output.append(dict(start_ns=a, end_ns=b, stage=name))
    return output


class Timeline:
    """Stream a gzip Chrome trace without retaining millions of GPU kernels."""

    def __init__(self, path: Path, origin: int):
        self.file = gzip.open(path, "wt", encoding="utf-8")
        self.file.write('{"traceEvents":[\n')
        self.first = True
        self.origin = origin

    def add(self, name, start, end=None, *, pid=0, tid=0, args=None, cat="host"):
        row = dict(name=name, ts=(start-self.origin)/1000, pid=pid, tid=tid,
                   cat=cat, args=args or {})
        if end is None:
            row.update(ph="i", s="t")
        else:
            row.update(ph="X", dur=max(0, end-start)/1000)
        if not self.first:
            self.file.write(",\n")
        self.first = False
        self.file.write(json.dumps(row, separators=(",", ":"), default=str))

    def close(self):
        self.file.write('],"displayTimeUnit":"ms"}\n')
        self.file.close()


def gpu_evidence(path: Path, timeline: Timeline) -> dict:
    """Join our NVTX ranges to runtime APIs and CUPTI activity timestamps."""
    result = dict(available=False, restores={}, computes={}, activities=0)
    if not path.exists():
        result["reason"] = "Nsight SQLite absent; actual GPU boundaries unknown"
        return result
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    if "NVTX_EVENTS" not in tables:
        result["reason"] = "NVTX_EVENTS absent"
        db.close()
        return result
    strings = dict(db.execute("select id,value from StringIds"))

    def label(row):
        return row.get("text") or strings.get(row.get("textId"), "")

    anchors, ranges = [], []
    for item in db.execute("select * from NVTX_EVENTS"):
        row = dict(item)
        name = label(row)
        if name.startswith("kvprobe_anchor:"):
            anchors.append(int(name.split(":", 1)[1]) - row["start"])
        elif name.startswith("kvprobe:") and row.get("end"):
            ranges.append((row, name))
    if not anchors:
        result["reason"] = "no monotonic/NVTX clock anchors"
        db.close()
        return result
    # An anchor records immediately BEFORE nvtx.mark. This median offset is
    # approximate; spread + host brackets are retained, never claimed exact.
    offset = round(statistics.median(anchors))
    result.update(available=True, offset_ns=offset, anchor_count=len(anchors),
                  anchor_offset_spread_ns=max(anchors)-min(anchors))
    selected = defaultdict(list)
    seen_compute = set()
    for row, name in sorted(ranges, key=lambda r: r[0]["start"]):
        _, stage, rid = name.split(":", 2)
        timeline.add(stage, row["start"]+offset, row["end"]+offset,
                     pid=(row["globalTid"] >> 24) & 0xffffff,
                     tid=row["globalTid"] & 0xffffff,
                     args={"request_id": rid}, cat="nvtx")
        ids = rid.split(",")
        if stage == "restore_cuda_submit" or (stage == "model_forward" and
                                               any(i not in seen_compute for i in ids)):
            selected[row["globalTid"]].append((row["start"], row["end"], stage, rid))
            if stage == "model_forward":
                seen_compute.update(ids)
    starts = {tid: [r[0] for r in values] for tid, values in selected.items()}
    correlations = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for raw in db.execute(f"select start,end,globalTid,correlationId from {table}"):
            a, b, tid, correlation = raw
            if tid not in starts:
                continue
            idx = bisect_right(starts[tid], a) - 1
            if idx >= 0:
                lo, hi, stage, rid = selected[tid][idx]
                if lo <= a and b <= hi:
                    correlations[((tid >> 24) & 0xffffff, correlation)] = (stage, rid)
    for table, category in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table not in tables:
            continue
        for item in db.execute(f"select * from {table}"):
            row = dict(item)
            pid = (row["globalPid"] >> 24) & 0xffffff
            a, b = row["start"]+offset, row["end"]+offset
            group = correlations.get((pid, row["correlationId"]))
            name = strings.get(row.get("demangledName"), category)
            args = {k: row[k] for k in ("bytes", "copyKind", "correlationId", "deviceId") if k in row}
            if group:
                stage, rid = group
                args["request_id"] = rid
                store = result["restores"] if stage == "restore_cuda_submit" else result["computes"]
                for req in ([rid] if stage == "restore_cuda_submit" else rid.split(",")):
                    value = store.setdefault(req, dict(start_ns=a, end_ns=b, activities=0,
                                                      memcpy_bytes=0, activity_sum_ns=0))
                    value["start_ns"] = min(value["start_ns"], a)
                    value["end_ns"] = max(value["end_ns"], b)
                    value["activities"] += 1
                    value["activity_sum_ns"] += b-a
                    value["memcpy_bytes"] += row.get("bytes", 0)
            timeline.add(name, a, b, pid=f"GPU:{row.get('deviceId',0)}:process:{pid}",
                         tid=f"stream:{row.get('streamId',0)}", args=args, cat=category)
            result["activities"] += 1
    db.close()
    return result


def analyze(out: str | Path) -> dict:
    """Write coverage, attribution, selected chains and raw visual timeline."""
    out = Path(out)
    if not (out / "replay.json").exists():
        report = {"complete": False, "reason": "no complete replay artifact"}
        (out / "analysis.json").write_text(json.dumps(report, indent=2))
        return report
    replay = json.loads((out / "replay.json").read_text())
    hosts = sorted([row for p in (out / "host").glob("probe-*.jsonl")
                    for row in read_jsonl(p)], key=lambda r: r["ts_ns"])
    bus = read_jsonl(out / "events.jsonl")
    aliases = {r["request_id"]: r["logical_id"] for r in hosts
               if r["event"] == "input_known" and r.get("logical_id")}
    grouped = defaultdict(list)
    timeline = Timeline(out / "timeline.json.gz", replay["origin_ns"])
    for event in hosts:
        rid = aliases.get(event.get("request_id"), event.get("request_id"))
        grouped[rid].append(event)
        if event["event"].endswith(".end") and "begin_ns" in event:
            timeline.add(event["event"][:-4], event["begin_ns"], event["ts_ns"],
                         pid=event["pid"], tid=event["tid"], args=event)
        else:
            timeline.add(event["event"], event["ts_ns"],
                         pid=event["pid"], tid=event["tid"], args=event)
    header = next(r for r in bus if r["record"] == "header")
    for event in bus:
        if event["record"] != "event":
            continue
        stamp = event.get("monotonic_ns") or (round(event["timestamp_s"]*1e9)
                    - header["clock_anchor_wall_ns"] + header["clock_anchor_monotonic_ns"])
        timeline.add(event["event"], stamp, pid=header["pid"], tid="LMCache event bus",
                     args=event, cat="cache")
    gpu = gpu_evidence(out / "gpu.sqlite", timeline)
    rows = []
    totals, incidence = Counter(), Counter()
    native_for = {logical: native for native, logical in aliases.items()}
    for index, req in enumerate(replay["requests"]):
        rid = req["id"]
        if req["status"] != "ok":
            rows.append(dict(id=rid, status=req["status"], error=req.get("error")))
            continue
        events = grouped[rid]
        first = {}
        for event in events:
            first.setdefault(event["event"], event)
        intervals = []
        milestones = {name: e["ts_ns"] for name, e in first.items()}
        milestones.update(eligible=req["eligible_ns"], dispatch=req["dispatch_ns"],
                          first_token_client=req["first_token_ns"])
        native = native_for.get(rid)
        gpu_restore = gpu["restores"].get(native)
        if gpu_restore:
            milestones.update(gpu_restore_start=gpu_restore["start_ns"],
                              gpu_restore_end=gpu_restore["end_ns"])

        def interval(left, right, stage, priority=1):
            if left in milestones and right in milestones:
                a, b = milestones[left], milestones[right]
                if a <= b:
                    intervals.append((a, b, stage, priority))

        interval("eligible", "dispatch", "client_arrival_lag")
        interval("dispatch", "input_known", "ingress_and_preprocessing")
        interval("input_known", "lookup_submitted", "lookup_trigger_delay")
        interval("lookup_submitted", "lookup_result", "lookup_and_result_polling")
        restore = "restore_submit" in first
        interval("lookup_result", "target_reserved", "admission_unexplained")
        if restore:
            interval("target_reserved", "restore_trigger", "restore_trigger_dispatch")
            interval("restore_trigger", "restore_submit", "target_fence_and_submit_cpu")
            interval("restore_submit", "restore_handler", "restore_ipc_and_handler_queue")
            interval("restore_handler", "restore_cuda_submit.begin", "source_acquisition_and_staging")
            interval("restore_cuda_submit.begin", "gpu_restore_start", "cuda_launch_and_stream_queue")
            interval("gpu_restore_start", "gpu_restore_end", "gpu_restore_envelope")
            interval("gpu_restore_end", "restore_reply_ready", "completion_observation_and_model_gate")
            interval("restore_reply_ready", "worker_recv_ready", "reply_and_worker_poll")
            interval("worker_recv_ready", "scheduler_recv_notification", "worker_to_scheduler_notification")
            interval("scheduler_recv_notification", "kv_published", "scheduler_adoption")
            interval("kv_published", "first_schedule", "ready_to_batch")
        else:
            interval("target_reserved", "first_schedule", "ready_to_batch")
        interval("first_schedule", "first_compute_launch", "batch_to_compute_launch")
        interval("first_compute_launch", "first_token_core", "prefill_compute_sampling_and_core_return")
        interval("first_token_core", "first_token_client", "output_delivery")
        denials = [e for e in events if e["event"] == "capacity_denied"
                   and e["ts_ns"] < req["first_token_ns"]]
        if denials and "target_reserved" in milestones:
            intervals.append((denials[0]["ts_ns"], milestones["target_reserved"],
                              "capacity_refusal_to_retry_upper_bound", 2))
        parts = partition(req["eligible_ns"], req["first_token_ns"], intervals)
        durations = Counter()
        for part in parts:
            ns = part["end_ns"]-part["start_ns"]
            durations[part["stage"]] += ns
            timeline.add(part["stage"], part["start_ns"], part["end_ns"],
                         pid="request TTFT", tid=rid, cat="attribution")
        for name, ns in durations.items():
            totals[name] += ns
            incidence[name] += ns > 0
        timeline.add(rid, req["eligible_ns"], req["completed_ns"],
                     pid="request lifetime", tid=rid, args={"output_tokens": len(req["output_tokens"])})
        rows.append(dict(id=rid, status="ok", native_id=native,
                         ttft_ms=(req["first_token_ns"]-req["eligible_ns"])/1e6,
                         actual_restore=restore,
                         lower_hit_tokens=first.get("lookup_result", {}).get("hit_tokens", 0),
                         capacity_denials=len(denials),
                         stage_ms={k:v/1e6 for k,v in durations.items()},
                         milestones_ns=milestones, gpu_restore=gpu_restore,
                         partition=parts,
                         event_chain=[e for e in events if e["ts_ns"] <= req["first_token_ns"]]))
    timeline.close()
    good = [r for r in rows if r["status"] == "ok"]
    total_ns = sum(totals.values())
    restored = [r for r in good if r["actual_restore"]]
    known_wait = [r for r in restored if r["milestones_ns"].get("input_known", 0)
                  < r["milestones_ns"].get("kv_published", 0)]
    footer = next((r for r in reversed(bus) if r["record"] == "footer"), None)
    witnesses = {}
    for kind, candidates, score in (
            ("capacity", [r for r in good if r["capacity_denials"]],
             lambda r:r["stage_ms"].get("capacity_refusal_to_retry_upper_bound", 0)),
            ("transport", restored, lambda r:r["stage_ms"].get("gpu_restore_envelope", 0)),
            ("trigger", restored, lambda r:r["stage_ms"].get("restore_trigger_dispatch", 0))):
        witnesses[kind] = (dict(observed=True, request_id=max(candidates, key=score)["id"],
                               interpretation="observed interval, not proven avoidable savings")
                           if candidates else dict(observed=False,
                                                   reason="no matching event chain in this trajectory"))
    unknown_names = {"unexplained", "admission_unexplained"}
    report = dict(complete=replay["complete"], coverage=dict(
        requests=len(rows), completed=len(good), actual_lower_restores=len(restored),
        lower_lookup_hit_requests=sum(r["lower_hit_tokens"] > 0 for r in good),
        restore_requests_waiting_after_input_known=len(known_wait),
        capacity_denied_requests=sum(r["capacity_denials"] > 0 for r in good),
        actual_gpu_restore_boundaries=sum(bool(r["gpu_restore"]) for r in restored)),
        stages=[dict(stage=k, total_ms=v/1e6, mean_all_requests_ms=v/1e6/len(good),
                     ttft_fraction=v/total_ns, requests=incidence[k])
                for k,v in sorted(totals.items(), key=lambda item:-item[1])],
        unexplained_ms=sum(totals[k] for k in unknown_names)/1e6,
        mean_ttft_ms=total_ns/1e6/len(good) if good else None,
        witnesses=witnesses, trace_footer=footer, gpu=gpu, requests=rows,
        limits=["one diagnostic run with instrumentation and Nsight; not a policy speedup",
                "GPU envelope includes device gaps; bandwidth contention is not identified by duration alone",
                "capacity refusal-to-retry is an observed upper bound, not all proven capacity wait",
                "CXL uses modeled registered shared memory, not physical CXL",
                "no trace-off comparison; observer and profiler overhead remains in measured latency"])
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 原生调度与 KV 恢复等待归因", "", f"完成 {len(good)}/{len(rows)} 请求；实际下层恢复 {len(restored)} 个。",
             "", f"平均 TTFT：{report['mean_ttft_ms'] or 0:.3f} ms。未解释部分合计：{report['unexplained_ms']:.3f} ms。",
             "", "| 观测阶段 | 每请求均摊 ms | TTFT 占比 | 涉及请求 |",
             "|---|---:|---:|---:|"]
    lines += [f"| {s['stage']} | {s['mean_all_requests_ms']:.3f} | {s['ttft_fraction']:.2%} | {s['requests']} |" for s in report['stages']]
    lines += ["", "事件链见 analysis.json；timeline.json.gz 可直接导入 Perfetto。", "", "## 解释边界", ""]
    lines += ["- " + item for item in report["limits"]]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
