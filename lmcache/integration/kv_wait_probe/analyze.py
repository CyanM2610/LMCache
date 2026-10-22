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
import math
import re
import sqlite3
import statistics


INGRESS_MILESTONES = (
    "api_request_received",
    "api_body_received",
    "api_handler_enter",
    "api_preprocess_begin",
    "api_preprocess_end",
    "engine_input_process_begin",
    "engine_input_process_end",
    "engine_ipc_submit_begin",
    "engine_ipc_submit_end",
    "engine_request_received",
    "engine_request_preprocessed",
    "engine_request_dequeued",
)


def group_request_events(hosts: list[dict]) -> tuple[dict, dict]:
    """Join logical API IDs and native IDs without losing early input events.

    Args:
        hosts: Host records containing event, request_id and optional logical_id.
    Returns:
        Native-to-logical scheduler aliases and chronologically grouped records.
        Explicit logical_id wins for records emitted before native ID assignment.
    """
    aliases = {
        row["request_id"]: row["logical_id"]
        for row in hosts
        if row["event"] == "input_known" and row.get("logical_id")
    }
    grouped = defaultdict(list)
    for row in sorted(hosts, key=lambda record: record["ts_ns"]):
        rid = row.get("logical_id") or aliases.get(
            row.get("request_id"), row.get("request_id")
        )
        grouped[rid].append(row)
    return aliases, grouped


def ingress_intervals(milestones: dict) -> tuple[list[tuple], dict]:
    """Split observed ingress boundaries, retaining holes and IPC concurrency.

    Args:
        milestones: Event names mapped to integer monotonic nanoseconds; dispatch
            and input_known bound the old coarse interval when present.
    Returns:
        Priority intervals for partition() and coverage/ordering diagnostics.
        Client IPC completion may overlap core receipt, so its raw span is
        reported independently and is never added to a concurrent core stage.
        Old traces without ingress probes retain their explicitly coarse stage.
    """
    measured = any(name in milestones for name in INGRESS_MILESTONES)
    intervals, observed, violations = [], [], []

    def add(left: str, right: str, stage: str, priority: int = 1) -> None:
        if left not in milestones or right not in milestones:
            return
        start, end = milestones[left], milestones[right]
        if start > end:
            violations.append(dict(left=left, right=right, delta_ns=end - start))
            return
        intervals.append((start, end, stage, priority))
        if priority:
            observed.append(
                dict(stage=stage, left=left, right=right, start_ns=start, end_ns=end)
            )

    add(
        "dispatch",
        "input_known",
        "ingress_unexplained" if measured else "ingress_and_preprocessing",
        0,
    )
    if measured:
        for left, right, stage in (
            ("dispatch", "api_request_received", "client_submit_to_api_receive"),
            ("api_request_received", "api_body_received", "api_body_receive"),
            ("api_body_received", "api_handler_enter", "api_parse_validate_dispatch"),
            ("api_handler_enter", "api_preprocess_begin", "api_handler_setup"),
            ("api_preprocess_begin", "api_preprocess_end", "api_preprocessing"),
            (
                "api_preprocess_end",
                "engine_input_process_begin",
                "api_to_engine_handoff",
            ),
            (
                "engine_input_process_begin",
                "engine_input_process_end",
                "engine_input_processing",
            ),
            (
                "engine_input_process_end",
                "engine_ipc_submit_begin",
                "engine_submit_preparation",
            ),
        ):
            add(left, right, stage)
        receipt = (
            "engine_socket_received"
            if "engine_socket_received" in milestones
            else "engine_request_received"
        )
        # Sending and the receiver's decode/preprocessing run in parallel. Use
        # completion to split only when it lies on the pre-receipt path.
        begin = milestones.get("engine_ipc_submit_begin")
        end = milestones.get("engine_ipc_submit_end")
        received = milestones.get(receipt)
        if (
            begin is not None
            and end is not None
            and received is not None
            and begin <= end <= received
        ):
            add(
                "engine_ipc_submit_begin",
                "engine_ipc_submit_end",
                "ipc_serialize_and_submit",
            )
            add("engine_ipc_submit_end", receipt, "ipc_submit_to_core_receive")
        else:
            add("engine_ipc_submit_begin", receipt, "ipc_submission_and_delivery")
        if receipt == "engine_socket_received":
            add(receipt, "engine_request_received", "engine_message_decode")
        add(
            "engine_request_received",
            "engine_request_preprocessed",
            "engine_core_preprocessing",
        )
        add(
            "engine_request_preprocessed",
            "engine_request_dequeued",
            "engine_input_queue",
        )
        add("engine_request_dequeued", "input_known", "engine_dispatch_to_scheduler")
    begin, end = (
        milestones.get("engine_ipc_submit_begin"),
        milestones.get("engine_ipc_submit_end"),
    )
    client_span = None
    if begin is not None and end is not None:
        if begin <= end:
            client_span = (end - begin) / 1e6
        else:
            violations.append(
                dict(
                    left="engine_ipc_submit_begin",
                    right="engine_ipc_submit_end",
                    delta_ns=end - begin,
                )
            )
    decoded = milestones.get("engine_request_received")
    return intervals, dict(
        mode="milestone_split" if measured else "legacy_coarse",
        missing=[name for name in INGRESS_MILESTONES if name not in milestones],
        order_violations=violations,
        client_ipc_span_ms=client_span,
        client_ipc_overlaps_decoded_receipt=(
            end > decoded if end is not None and decoded is not None else None
        ),
        intervals=observed,
        chain=[
            dict(event=name, ts_ns=stamp)
            for name, stamp in sorted(milestones.items(), key=lambda item: item[1])
            if name in INGRESS_MILESTONES
            or name in ("dispatch", "engine_socket_received", "input_known")
        ],
    )


def read_jsonl(path: Path) -> list[dict]:
    """Read complete records; fail rather than silently ignore trace corruption."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def partition(start: int, end: int, intervals: list[tuple]) -> list[dict]:
    """Cover TTFT once; larger priority owns overlap, all holes stay unknown."""
    if end < start:
        raise ValueError("negative TTFT")
    active = [
        (max(start, a), min(end, b), name, priority)
        for a, b, name, priority in intervals
        if a < b and a < end and b > start
    ]
    boundaries = sorted({start, end} | {v for a, b, _, _ in active for v in (a, b)})
    output = []
    for a, b in zip(boundaries, boundaries[1:]):
        covered = [
            (priority, name) for lo, hi, name, priority in active if lo <= a and hi >= b
        ]
        name = max(covered)[1] if covered else "unexplained"
        if output and output[-1]["stage"] == name:
            output[-1]["end_ns"] = b
        else:
            output.append(dict(start_ns=a, end_ns=b, stage=name))
    return output


def evidence_validity(out: Path, replay: dict, hosts: list, bus: list) -> dict:
    """Separate replay completion from drained, loss-audited evidence."""
    failures, unknowns = [], []
    if not replay.get("complete") or any(
        r.get("status") != "ok" for r in replay.get("requests", [])
    ):
        failures.append("replay incomplete")
    service_path = out / "service.json"
    service = json.loads(service_path.read_text()) if service_path.exists() else None
    if service is None:
        unknowns.append("service shutdown/drain status absent")
    elif not service.get("passed"):
        failures.append("service shutdown/drain validation did not pass")
    footer = next((r for r in reversed(bus) if r.get("record") == "footer"), None)
    records = [r for r in bus if r.get("record") == "event"]
    if footer is None:
        unknowns.append("event bus terminal loss accounting absent")
    else:
        if footer.get("events_written") != len(records):
            failures.append("event bus record count does not match footer")
        for name in ("bus_dropped_events", "bus_queue_depth"):
            if name not in footer:
                unknowns.append(name + " not audited")
            elif footer[name] != 0:
                failures.append(name + " is nonzero")
        for name in ("subscriber_errors", "host_phase_errors"):
            if name not in footer:
                unknowns.append(name + " not audited")
            else:
                value = footer[name]
                nonzero = (
                    any(value.values()) if isinstance(value, dict) else bool(value)
                )
                if nonzero:
                    failures.append(name + " is nonzero")
    if [r.get("sequence") for r in records] != list(range(len(records))):
        failures.append("event bus sequence is not contiguous from zero")
    by_pid = defaultdict(list)
    for row in hosts:
        by_pid[row["pid"]].append(row)
    if not by_pid:
        unknowns.append("host trace absent")
    for pid, records in by_pid.items():
        summaries = [r for r in records if r["event"] == "observer_summary"]
        if not summaries:
            unknowns.append(f"host process {pid} has no terminal observer summary")
        elif len(summaries) != 1 or summaries[0].get("events") != len(records) - 1:
            failures.append(f"host process {pid} terminal record count mismatch")
    return dict(
        status="invalid" if failures else "unconfirmed" if unknowns else "valid",
        passed=not failures and not unknowns,
        failures=failures,
        unconfirmed=unknowns,
        replay_complete=bool(replay.get("complete")),
        service_passed=service.get("passed") if service else None,
    )


def select_witnesses(rows: list[dict]) -> dict:
    """Report observed intervals without equating service with blocking."""
    result = {}
    for kind, candidates, stage in (
        (
            "capacity",
            [r for r in rows if r["capacity_denials"]],
            "capacity_refusal_to_retry_upper_bound",
        ),
        (
            "transport",
            [r for r in rows if r["gpu_restore"]],
            "gpu_restore_envelope",
        ),
        (
            "trigger",
            [r for r in rows if r["stage_ms"].get("restore_trigger_dispatch", 0) > 0],
            "restore_trigger_dispatch",
        ),
    ):
        if not candidates:
            result[kind] = dict(
                observed=False,
                blocking_observed=False if kind == "capacity" else None,
                reason="no matching measured interval in this trajectory",
            )
            continue
        witness = max(candidates, key=lambda r: r["stage_ms"].get(stage, 0))
        result[kind] = dict(
            observed=True,
            request_id=witness["id"],
            interval=stage,
            interval_ms=witness["stage_ms"].get(stage, 0),
            blocking_observed=True if kind == "capacity" else None,
            interpretation=(
                "explicit allocation refusal; interval to retry is only an upper bound"
                if kind == "capacity"
                else "GPU service observed; queue or bandwidth blocking not identified"
                if kind == "transport"
                else "reservation-to-trigger interval includes dispatch and CPU work; avoidability unknown"
            ),
        )
    return result


class Timeline:
    """Stream Chrome events with separate logical tracks for CPU observers."""

    def __init__(self, path: Path, origin: int):
        self.file = gzip.open(path, "wt", encoding="utf-8")
        self.file.write('{"traceEvents":[\n')
        self.first = True
        self.origin = origin
        self.processes = {}
        self.threads = {}

    def _write(self, row):
        if not self.first:
            self.file.write(",\n")
        self.first = False
        self.file.write(json.dumps(row, separators=(",", ":"), default=str))

    def add(self, name, start, end=None, *, pid=0, tid=0, args=None, cat="host"):
        if cat in ("host", "nvtx") and isinstance(pid, int) and isinstance(tid, int):
            args = {**(args or {}), "real_pid": pid, "real_tid": tid}
            source = "NVTX" if cat == "nvtx" else "host"
            pid = f"{source} / process:{pid}"
            tid = f"thread:{tid}"
        pid, tid = self._track(pid, tid)
        row = dict(
            name=name,
            ts=(start - self.origin) / 1000,
            pid=pid,
            tid=tid,
            cat=cat,
            args=args or {},
        )
        if end is None:
            row.update(ph="i", s="t")
        else:
            row.update(ph="X", dur=max(0, end - start) / 1000)
        self._write(row)

    def counter(
        self, name: str, stamp: int, value: float, *, source: str, device: str
    ) -> None:
        """Write one unscaled, finite numeric counter on a source/device track.

        Args:
            name: Native metric name including its known unit.
            stamp: Monotonic nanoseconds, already aligned to host observations.
            value: Native sample value; absent values must not call this method.
            source: Measurement provider; keeps NVML and Nsight independent.
            device: Device/type identity, never a CUDA-visible ordinal guess.
        Raises:
            ValueError: A nonfinite value would produce invalid trace JSON.
        """
        if not math.isfinite(value):
            raise ValueError("nonfinite GPU counter")
        pid, tid = self._track(f"{source} counters / {device}", name)
        self._write(
            dict(
                name=name,
                ph="C",
                ts=(stamp - self.origin) / 1000,
                pid=pid,
                tid=tid,
                cat="gpu_counter",
                args={"value": value},
            )
        )

    def _track(self, pid: int | str, tid: int | str) -> tuple[int, int]:
        if isinstance(pid, str):
            label = pid
            if label not in self.processes:
                self.processes[label] = 1_000_000_000 + len(self.processes)
                self._write(
                    dict(
                        ph="M",
                        name="process_name",
                        pid=self.processes[label],
                        tid=0,
                        args={"name": label},
                    )
                )
            pid = self.processes[label]
        if isinstance(tid, str):
            key = (pid, tid)
            if key not in self.threads:
                self.threads[key] = 1_000_000_000 + len(self.threads)
                self._write(
                    dict(
                        ph="M",
                        name="thread_name",
                        pid=pid,
                        tid=self.threads[key],
                        args={"name": tid},
                    )
                )
            tid = self.threads[key]
        return pid, tid

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
    tables = {
        r[0] for r in db.execute("select name from sqlite_master where type='table'")
    }
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
    result.update(
        available=True,
        offset_ns=offset,
        anchor_count=len(anchors),
        anchor_offset_spread_ns=max(anchors) - min(anchors),
    )
    selected = defaultdict(list)
    seen_compute = set()
    for row, name in sorted(ranges, key=lambda r: r[0]["start"]):
        _, stage, rid = name.split(":", 2)
        timeline.add(
            stage,
            row["start"] + offset,
            row["end"] + offset,
            pid=(row["globalTid"] >> 24) & 0xFFFFFF,
            tid=row["globalTid"] & 0xFFFFFF,
            args={"request_id": rid},
            cat="nvtx",
        )
        ids = rid.split(",")
        if stage == "restore_cuda_submit" or (
            stage == "model_forward" and any(i not in seen_compute for i in ids)
        ):
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
                    correlations[((tid >> 24) & 0xFFFFFF, correlation)] = (stage, rid)
    for table, category in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        if table not in tables:
            continue
        for item in db.execute(f"select * from {table}"):
            row = dict(item)
            pid = (row["globalPid"] >> 24) & 0xFFFFFF
            a, b = row["start"] + offset, row["end"] + offset
            group = correlations.get((pid, row["correlationId"]))
            name = strings.get(row.get("demangledName"), category)
            args = {
                k: row[k]
                for k in ("bytes", "copyKind", "correlationId", "deviceId")
                if k in row
            }
            if group:
                stage, rid = group
                args["request_id"] = rid
                store = (
                    result["restores"]
                    if stage == "restore_cuda_submit"
                    else result["computes"]
                )
                for req in [rid] if stage == "restore_cuda_submit" else rid.split(","):
                    value = store.setdefault(
                        req,
                        dict(
                            start_ns=a,
                            end_ns=b,
                            activities=0,
                            memcpy_bytes=0,
                            activity_sum_ns=0,
                        ),
                    )
                    value["start_ns"] = min(value["start_ns"], a)
                    value["end_ns"] = max(value["end_ns"], b)
                    value["activities"] += 1
                    value["activity_sum_ns"] += b - a
                    value["memcpy_bytes"] += row.get("bytes", 0)
            timeline.add(
                name,
                a,
                b,
                pid=f"GPU:{row.get('deviceId', 0)}:process:{pid}",
                tid=f"stream:{row.get('streamId', 0)}",
                args=args,
                cat=category,
            )
            result["activities"] += 1
    db.close()
    return result


def _metric_unit(definition: dict) -> tuple[str | None, str]:
    for column in ("unit", "metricUnit", "units"):
        if isinstance(definition.get(column), str) and definition[column]:
            return definition[column], f"TARGET_INFO_GPU_METRICS.{column}"
    name = definition["metricName"]
    suffix = re.search(r"\[([^\]]+)\]$", name)
    if suffix:
        return suffix.group(1), "native metric name suffix"
    # Only documented names have a fallback unit. Unknown future metrics remain
    # unscaled and explicitly unit-unknown; in particular not all are percent.
    if name in {
        "SMs Active",
        "SM Issue",
        "DRAM Read Bandwidth",
        "DRAM Write Bandwidth",
        "PCIe Read Throughput",
        "PCIe Write Throughput",
    }:
        return (
            "%",
            "https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics",
        )
    return None, "unknown; raw value preserved"


def _metric_device(definition: dict, devices: list[dict]) -> dict:
    # NVIDIA documents low 8 bits of GPU metric typeId as the GPU id, and the
    # upper 16 bits as hardware/VM. Verify against TARGET_INFO_GPU where present.
    # https://forums.developer.nvidia.com/t/338545/6
    explicit = next(
        (definition[key] for key in ("gpuId", "deviceId") if key in definition),
        None,
    )
    gpu_id = explicit if explicit is not None else definition["typeId"] & 0xFF
    scope = definition["typeId"] & 0xFFFF000000000000
    matches = [
        row
        for row in devices
        if row.get("id", row.get("gpuId")) == gpu_id
        and ("vmId" not in row or row["vmId"] & 0xFFFF000000000000 == scope)
    ]
    return dict(
        gpu_id=gpu_id,
        mapping="explicit definition column"
        if explicit is not None
        else "documented typeId low 8 bits",
        metadata=matches[0] if len(matches) == 1 else None,
        metadata_matched=len(matches) == 1,
        label=f"GPU:{gpu_id} / type:{definition['typeId']}",
    )


def nsight_counters(path: Path, timeline: Timeline, clock: dict) -> dict:
    """Export native GPU_METRICS samples without estimating missing counters.

    Args:
        path: Optional Nsight SQLite export; opened read-only.
        timeline: Counter destination, preserving raw values and names.
        clock: gpu_evidence result with measured host/NVTX offset and anchors.
    Returns:
        Availability, schema, per-metric units/device/coverage and clock evidence.
        No samples are emitted without an observed clock anchor. Unknown schemas
        remain explicitly unavailable, without affecting host-event attribution.
    """
    result = dict(available=False, emitted_samples=0, raw_samples=0, metrics=[])
    result["clock"] = {
        key: clock.get(key)
        for key in ("offset_ns", "anchor_count", "anchor_offset_spread_ns")
    }
    if not path.exists():
        result["reason"] = "Nsight SQLite absent"
        return result
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        required = {
            "GPU_METRICS": {"timestamp", "typeId", "metricId", "value"},
            "TARGET_INFO_GPU_METRICS": {"typeId", "metricId", "metricName"},
        }
        result["schema"] = {
            table: [r[1] for r in db.execute(f"pragma table_info({table})")]
            for table in required
        }
        if any(
            not columns.issubset(result["schema"][table])
            for table, columns in required.items()
        ):
            result["reason"] = "GPU metric tables absent or schema unsupported"
            return result
        result["raw_samples"] = db.execute(
            "select count(*) from GPU_METRICS"
        ).fetchone()[0]
        devices = (
            [dict(r) for r in db.execute("select * from TARGET_INFO_GPU")]
            if "TARGET_INFO_GPU" in tables
            else []
        )
        definitions = {}
        for raw in db.execute("select * from TARGET_INFO_GPU_METRICS"):
            definition = dict(raw)
            key = (definition["typeId"], definition["metricId"])
            if key in definitions:
                result["reason"] = "ambiguous duplicate GPU metric definition"
                return result
            unit, evidence = _metric_unit(definition)
            definitions[key] = dict(
                definition=definition,
                unit=unit,
                unit_evidence=evidence,
                device=_metric_device(definition, devices),
                samples=0,
                first_ns=None,
                last_ns=None,
                invalid_samples=0,
            )
        result["metrics"] = list(definitions.values())
        if (
            not clock.get("available")
            or not clock.get("anchor_count")
            or not isinstance(clock.get("offset_ns"), int)
        ):
            result["reason"] = "GPU metrics captured but no host/NVTX clock alignment"
            return result
        offset = clock["offset_ns"]
        result["unmapped_samples"] = 0
        result["invalid_timestamps"] = 0
        for row in db.execute(
            "select timestamp,typeId,metricId,value from GPU_METRICS order by timestamp"
        ):
            stamp, type_id, metric_id, value = row
            metric = definitions.get((type_id, metric_id))
            if metric is None:
                result["unmapped_samples"] += 1
                continue
            if not isinstance(stamp, int) or stamp < 0 or stamp + offset < 0:
                result["invalid_timestamps"] += 1
                metric["invalid_samples"] += 1
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                metric["invalid_samples"] += 1
                continue
            aligned = stamp + offset
            name = metric["definition"]["metricName"]
            if not re.search(r"\[[^\]]+\]$", name):
                name += f" [{metric['unit'] or 'unit unknown'}]"
            timeline.counter(
                name,
                aligned,
                value,
                source="Nsight GPU Metrics",
                device=metric["device"]["label"],
            )
            metric["samples"] += 1
            metric["first_ns"] = (
                aligned if metric["first_ns"] is None else metric["first_ns"]
            )
            metric["last_ns"] = aligned
            result["emitted_samples"] += 1
        result["available"] = result["emitted_samples"] > 0
        if not result["available"]:
            result["reason"] = "no usable GPU metric samples"
    finally:
        db.close()
    return result


def gpu_counters(out: Path, timeline: Timeline, clock: dict) -> dict:
    """Add optional Nsight and NVML counters and return coverage/provenance.

    Args:
        out: Run directory containing SQLite and optional telemetry/status files.
        timeline: Raw trace destination; missing values do not become zeros.
        clock: Clock alignment returned by gpu_evidence for this same run.
    Returns:
        Independent provider availability and sampling limitations. NVML sample
        timestamps are query midpoints, not exact hardware interval boundaries.
    """
    nsight = nsight_counters(out / "gpu.sqlite", timeline, clock)
    telemetry = dict(available=False, samples=0, emitted_samples=0, metrics={})
    result = dict(
        nsight=nsight,
        nvml=telemetry,
        interpretation=[
            "Device-wide sampled counters are not per-request attribution.",
            "NVML GPU busy is time with any kernel, not the fraction of active SMs.",
            "NVML memory busy is temporal utilization, not bandwidth in GB/s.",
            "PCIe samples cover the device and cannot identify modeled CXL traffic.",
            "No interpolation or zero fill; gaps and absent counters stay unmeasured.",
        ],
    )
    for filename, key in (
        ("gpu-metrics-status.json", "capture_status"),
        ("gpu-telemetry-status.json", "telemetry_status"),
    ):
        path = out / filename
        if path.exists():
            result[key] = json.loads(path.read_text())
    path = out / "gpu-telemetry.jsonl"
    if not path.exists():
        telemetry["reason"] = "NVML telemetry absent"
        return result
    status = result.get("telemetry_status", {})
    fields = {
        "gpu_kernel_active_percent": ("GPU kernel busy", "%"),
        "memory_busy_percent": ("Device memory busy", "%"),
        "pcie_rx_kb_per_s": ("PCIe Rx", "KB/s"),
        "pcie_tx_kb_per_s": ("PCIe Tx", "KB/s"),
    }
    rows = read_jsonl(path)
    telemetry["samples"] = len(rows)
    telemetry["invalid_timestamps"] = 0
    telemetry["max_query_duration_ns"] = None
    telemetry["devices"] = {}
    for row in rows:
        stamp = row.get("monotonic_ns")
        if not isinstance(stamp, int):
            telemetry["invalid_timestamps"] += 1
            continue
        gpu, uuid = row.get("gpu"), row.get("gpu_uuid")
        device = f"GPU:{gpu} / {uuid or 'UUID unavailable'}"
        telemetry["devices"][device] = dict(gpu=gpu, uuid=uuid)
        duration = row.get("query_duration_ns")
        if isinstance(duration, int):
            telemetry["max_query_duration_ns"] = max(
                telemetry["max_query_duration_ns"] or 0, duration
            )
        for field, (label, default_unit) in fields.items():
            unit = status.get("units", {}).get(field, default_unit)
            key = f"{device} / {field}"
            metric = telemetry["metrics"].setdefault(
                key,
                dict(
                    name=field,
                    device=device,
                    unit=unit,
                    samples=0,
                    missing_samples=0,
                    first_ns=None,
                    last_ns=None,
                ),
            )
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                metric["missing_samples"] += 1
                continue
            timeline.counter(
                f"{label} [{unit}]", stamp, value, source="NVML", device=device
            )
            metric["samples"] += 1
            metric["first_ns"] = (
                stamp if metric["first_ns"] is None else min(metric["first_ns"], stamp)
            )
            metric["last_ns"] = (
                stamp if metric["last_ns"] is None else max(metric["last_ns"], stamp)
            )
            telemetry["emitted_samples"] += 1
    telemetry["available"] = telemetry["emitted_samples"] > 0
    if not telemetry["available"]:
        telemetry["reason"] = "no usable NVML samples"
    return result


def analyze(out: str | Path) -> dict:
    """Write coverage, attribution, selected chains and raw visual timeline."""
    out = Path(out)
    if not (out / "replay.json").exists():
        report = {"complete": False, "reason": "no complete replay artifact"}
        (out / "analysis.json").write_text(json.dumps(report, indent=2))
        return report
    replay = json.loads((out / "replay.json").read_text())
    hosts = sorted(
        [row for p in (out / "host").glob("probe-*.jsonl") for row in read_jsonl(p)],
        key=lambda r: r["ts_ns"],
    )
    bus_path = out / "events.jsonl"
    bus = read_jsonl(bus_path) if bus_path.exists() else []
    validity = evidence_validity(out, replay, hosts, bus)
    aliases, grouped = group_request_events(hosts)
    # Absolute MONOTONIC timestamps keep startup GPU events nonnegative too.
    # Trace viewers normalize their display origin while preserving intervals.
    timeline = Timeline(out / "timeline.json.gz", 0)
    for event in hosts:
        if event["event"].endswith(".end") and "begin_ns" in event:
            timeline.add(
                event["event"][:-4],
                event["begin_ns"],
                event["ts_ns"],
                pid=event["pid"],
                tid=event["tid"],
                args=event,
            )
        else:
            timeline.add(
                event["event"],
                event["ts_ns"],
                pid=event["pid"],
                tid=event["tid"],
                args=event,
            )
    header = next((r for r in bus if r["record"] == "header"), None)
    for event in bus:
        if event["record"] != "event":
            continue
        if header is None:
            raise ValueError("event bus has records without a clock header")
        stamp = event.get("monotonic_ns") or (
            round(event["timestamp_s"] * 1e9)
            - header["clock_anchor_wall_ns"]
            + header["clock_anchor_monotonic_ns"]
        )
        timeline.add(
            event["event"],
            stamp,
            pid=header["pid"],
            tid="LMCache event bus",
            args=event,
            cat="cache",
        )
    gpu = gpu_evidence(out / "gpu.sqlite", timeline)
    counters = gpu_counters(out, timeline, gpu)
    accesses = defaultdict(list)
    for event in bus:
        metadata = event.get("metadata", {})
        if (
            event.get("event") == "hybrid.access"
            and metadata.get("direction") == "load"
            and metadata.get("phase") == "complete"
        ):
            accesses[event.get("session_id")].append(metadata)
    rows = []
    totals, incidence = Counter(), Counter()
    native_for = {logical: native for native, logical in aliases.items()}
    for index, req in enumerate(replay["requests"]):
        rid = req["id"]
        if req["status"] != "ok":
            rows.append(dict(id=rid, status=req["status"], error=req.get("error")))
            continue
        # This is a TTFT analysis, not an all-lifetime first-event join. Later
        # preemption/recovery must not be charged to initial token production.
        events = [e for e in grouped[rid] if e["ts_ns"] <= req["first_token_ns"]]
        first = {}
        for event in events:
            first.setdefault(event["event"], event)
        intervals = []
        milestones = {name: e["ts_ns"] for name, e in first.items()}
        milestones.update(
            eligible=req["eligible_ns"],
            dispatch=req["dispatch_ns"],
            first_token_client=req["first_token_ns"],
        )
        socket_received = first.get("engine_request_received", {}).get(
            "socket_received_ns"
        )
        if socket_received is not None:
            milestones["engine_socket_received"] = socket_received
        native = native_for.get(rid)
        gpu_restore = gpu["restores"].get(native)
        if gpu_restore and (
            gpu_restore["start_ns"] < req["dispatch_ns"]
            or gpu_restore["end_ns"] > req["first_token_ns"]
        ):
            # The GPU join currently aggregates a request's restore ranges.
            # Repeated recovery beyond TTFT is not one initial-load envelope.
            gpu_restore = None
        if gpu_restore:
            milestones.update(
                gpu_restore_start=gpu_restore["start_ns"],
                gpu_restore_end=gpu_restore["end_ns"],
            )

        def interval(left, right, stage, priority=1):
            if left in milestones and right in milestones:
                a, b = milestones[left], milestones[right]
                if a <= b:
                    intervals.append((a, b, stage, priority))

        interval("eligible", "dispatch", "client_arrival_lag")
        ingress_parts, ingress = ingress_intervals(milestones)
        intervals.extend(ingress_parts)
        for boundary in ingress["chain"]:
            boundary["since_origin_ms"] = (
                boundary["ts_ns"] - replay["origin_ns"]
            ) / 1e6
        interval("input_known", "lookup_submitted", "lookup_trigger_delay")
        interval("lookup_submitted", "lookup_result", "lookup_and_result_polling")
        restore = "restore_submit" in first
        reply = first.get("restore_reply_ready", {})
        worker = first.get("worker_recv_ready", {})
        successful_restore = (
            restore
            and reply.get("success") is True
            and worker.get("failed") is False
            and "kv_published" in first
            and first.get("lookup_result", {}).get("hit_tokens", 0) > 0
        )
        interval("lookup_result", "target_reserved", "admission_unexplained")
        if restore:
            interval("target_reserved", "restore_trigger", "restore_trigger_dispatch")
            interval("restore_trigger", "restore_submit", "target_fence_and_submit_cpu")
            interval(
                "restore_submit", "restore_handler", "restore_ipc_and_handler_queue"
            )
            interval(
                "restore_handler",
                "restore_cuda_submit.begin",
                "source_acquisition_and_staging",
            )
            interval(
                "restore_cuda_submit.begin",
                "gpu_restore_start",
                "cuda_launch_and_stream_queue",
            )
            interval("gpu_restore_start", "gpu_restore_end", "gpu_restore_envelope")
            interval(
                "gpu_restore_end",
                "restore_reply_ready",
                "completion_observation_and_model_gate",
            )
            interval(
                "restore_reply_ready", "worker_recv_ready", "reply_and_worker_poll"
            )
            interval(
                "worker_recv_ready",
                "scheduler_recv_notification",
                "worker_to_scheduler_notification",
            )
            interval(
                "scheduler_recv_notification", "kv_published", "scheduler_adoption"
            )
            interval("kv_published", "first_schedule", "ready_to_batch")
        else:
            interval("target_reserved", "first_schedule", "ready_to_batch")
        interval("first_schedule", "first_compute_launch", "batch_to_compute_launch")
        interval(
            "first_compute_launch",
            "first_token_core",
            "prefill_compute_sampling_and_core_return",
        )
        interval("first_token_core", "first_token_client", "output_delivery")
        denials = [
            e
            for e in events
            if e["event"] == "capacity_denied" and e["ts_ns"] < req["first_token_ns"]
        ]
        if denials and "target_reserved" in milestones:
            intervals.append(
                (
                    denials[0]["ts_ns"],
                    milestones["target_reserved"],
                    "capacity_refusal_to_retry_upper_bound",
                    2,
                )
            )
        parts = partition(req["eligible_ns"], req["first_token_ns"], intervals)
        durations = Counter()
        for part in parts:
            ns = part["end_ns"] - part["start_ns"]
            durations[part["stage"]] += ns
            timeline.add(
                part["stage"],
                part["start_ns"],
                part["end_ns"],
                pid="request TTFT",
                tid=rid,
                cat="attribution",
            )
        for name, ns in durations.items():
            totals[name] += ns
            incidence[name] += ns > 0
        timeline.add(
            rid,
            req["eligible_ns"],
            req["completed_ns"],
            pid="request lifetime",
            tid=rid,
            args={"output_tokens": len(req["output_tokens"])},
        )
        rows.append(
            dict(
                id=rid,
                status="ok",
                native_id=native,
                ttft_ms=(req["first_token_ns"] - req["eligible_ns"]) / 1e6,
                restore_attempt=restore,
                actual_restore=successful_restore,
                restore_success_confirmed=successful_restore,
                restore_media=sorted(
                    {
                        a["medium"]
                        for a in accesses.get(native, [])
                        if a.get("state") == 3
                        and a.get("logical_observed_ns", req["first_token_ns"] + 1)
                        <= req["first_token_ns"]
                    }
                ),
                lower_hit_tokens=first.get("lookup_result", {}).get("hit_tokens", 0),
                capacity_denials=len(denials),
                stage_ms={k: v / 1e6 for k, v in durations.items()},
                milestones_ns=milestones,
                ingress=ingress,
                gpu_restore=gpu_restore,
                partition=parts,
                event_chain=[e for e in events if e["ts_ns"] <= req["first_token_ns"]],
            )
        )
    timeline.close()
    good = [r for r in rows if r["status"] == "ok"]
    missing = {
        r["id"]: [
            name
            for name in (
                "input_known",
                "first_schedule",
                "first_compute_launch",
                "first_token_core",
            )
            if name not in r["milestones_ns"]
        ]
        for r in good
    }
    missing = {rid: names for rid, names in missing.items() if names}
    if missing:
        validity["unconfirmed"].append("required request host boundaries are missing")
        validity["missing_request_boundaries"] = missing
        validity["passed"] = False
        if validity["status"] == "valid":
            validity["status"] = "unconfirmed"
    total_ns = sum(totals.values())
    restored = [r for r in good if r["actual_restore"]]
    known_wait = [
        r
        for r in restored
        if "input_known" in r["milestones_ns"]
        and r["milestones_ns"]["input_known"]
        < r["milestones_ns"].get("kv_published", 0)
    ]
    footer = next((r for r in reversed(bus) if r["record"] == "footer"), None)
    witnesses = select_witnesses(good)
    unknown_names = {"unexplained", "admission_unexplained", "ingress_unexplained"}
    ingress_measured = [r for r in good if r["ingress"]["mode"] == "milestone_split"]
    ingress_incomplete = {
        r["id"]: r["ingress"]["missing"] for r in good if r["ingress"]["missing"]
    }
    ingress_invalid = {
        r["id"]: r["ingress"]["order_violations"]
        for r in good
        if r["ingress"]["order_violations"]
    }
    if ingress_measured and ingress_incomplete:
        validity["unconfirmed"].append(
            "ingress instrumentation has missing request boundaries"
        )
        validity["passed"] = False
        if validity["status"] == "valid":
            validity["status"] = "unconfirmed"
    if ingress_invalid:
        validity["failures"].append("ingress milestone order violations")
        validity["passed"] = False
        validity["status"] = "invalid"
    workload_path = out / "workload.json"
    workload_requests = (
        json.loads(workload_path.read_text()).get("requests", [])
        if workload_path.exists()
        else []
    )
    root_ids = {r["id"] for r in workload_requests if r.get("parent") is None}
    ingress_stages = defaultdict(list)
    for request in good:
        for value in request["ingress"]["intervals"]:
            ingress_stages[value["stage"]].append(value)
    report = dict(
        complete=bool(replay["complete"] and validity["passed"]),
        replay_complete=bool(replay["complete"]),
        validity=validity,
        coverage=dict(
            requests=len(rows),
            completed=len(good),
            actual_lower_restores=len(restored),
            restore_attempts=sum(r["restore_attempt"] for r in good),
            restore_attempts_without_confirmed_success=sum(
                r["restore_attempt"] and not r["actual_restore"] for r in good
            ),
            actual_recovery_sessions=len({r["id"].rsplit(":", 1)[0] for r in restored}),
            actual_restore_media=dict(
                Counter(medium for r in restored for medium in r["restore_media"])
            ),
            lower_lookup_hit_requests=sum(r["lower_hit_tokens"] > 0 for r in good),
            restore_requests_waiting_after_input_known=len(known_wait),
            capacity_denied_requests=sum(r["capacity_denials"] > 0 for r in good),
            actual_gpu_restore_boundaries=sum(bool(r["gpu_restore"]) for r in restored),
        ),
        ingress=dict(
            interpretation="observed wall-clock intervals include CPU execution and queuing; not pure wait or causal savings",
            measured_requests=len(ingress_measured),
            complete_boundary_requests=sum(not r["ingress"]["missing"] for r in good),
            milestone_requests={
                name: sum(name in r["milestones_ns"] for r in good)
                for name in (
                    *INGRESS_MILESTONES,
                    "engine_socket_received",
                    "input_known",
                )
            },
            incomplete_requests=ingress_incomplete,
            order_violations=ingress_invalid,
            observed_interval_requests={
                name: len(values) for name, values in ingress_stages.items()
            },
            first_roots=[
                dict(id=r["id"], chain=r["ingress"]["chain"], stage_ms=r["stage_ms"])
                for r in sorted(good, key=lambda r: r["milestones_ns"]["dispatch"])
                if r["id"] in root_ids
            ][:3],
        ),
        stages=[
            dict(
                stage=k,
                total_ms=v / 1e6,
                mean_all_requests_ms=v / 1e6 / len(good),
                ttft_fraction=v / total_ns,
                requests=incidence[k],
            )
            for k, v in sorted(totals.items(), key=lambda item: -item[1])
        ],
        unexplained_ms=sum(totals[k] for k in unknown_names) / 1e6,
        mean_ttft_ms=total_ns / 1e6 / len(good) if good else None,
        witnesses=witnesses,
        trace_footer=footer,
        gpu=gpu,
        gpu_counters=counters,
        requests=rows,
        limits=[
            "one diagnostic run with instrumentation and Nsight; not a policy speedup",
            "GPU envelope includes device gaps; bandwidth contention is not identified by duration alone",
            "TTFT partition describes composite observed intervals, not pure waiting time or causal savings",
            "transport and trigger intervals alone do not establish avoidable blocking",
            "successful restores require positive lookup, successful reply, nonfailed worker notification and publication",
            "capacity refusal-to-retry is an observed upper bound, not all proven capacity wait",
            "CXL uses modeled registered shared memory, not physical CXL",
            "no trace-off comparison; observer and profiler overhead remains in measured latency",
            "client IPC send completion may overlap EngineCore receipt; its independent span is not added twice",
            "legacy traces without ingress probes keep their coarse ingress interval; missing new boundaries remain unknown",
        ],
    )
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# 原生调度与 KV 恢复等待归因",
        "",
        f"完成 {len(good)}/{len(rows)} 请求；实际下层恢复 {len(restored)} 个。",
        f"证据有效性：{validity['status']}；回放完成不等于事件及退出状态已通过核验。",
        "",
        f"平均 TTFT：{report['mean_ttft_ms'] or 0:.3f} ms。未解释部分合计：{report['unexplained_ms']:.3f} ms。",
        "",
        "以下为互斥的观测区间分解；复合区间包括执行、CPU 工作和等待，不能全部解释为可优化等待。",
        "",
        "| 观测区间 | 每请求均摊 ms | TTFT 占比 | 涉及请求 |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {s['stage']} | {s['mean_all_requests_ms']:.3f} | {s['ttft_fraction']:.2%} | {s['requests']} |"
        for s in report["stages"]
    ]
    lines += [
        "",
        "## 入口细分",
        "",
        f"带入口探针 {len(ingress_measured)}/{len(good)} 请求；全部新边界齐全 {report['ingress']['complete_boundary_requests']}/{len(good)}。",
        "缺失边界保留为未知；客户端 IPC 提交完成可能晚于 EngineCore 收到请求，其独立跨度不重复计入 TTFT。",
        "",
        "| 时间点 | 已观测请求数 |",
        "|---|---:|",
    ]
    lines += [
        f"| {name} | {count} |"
        for name, count in report["ingress"]["milestone_requests"].items()
    ]
    roots = report["ingress"]["first_roots"]
    if roots:
        lines += [
            "",
            "首批三个根请求，时间为相对回放起点的 ms；表格为逻辑路径，原始 chain 按实际时间排序。",
            "",
            "| 时间点 | " + " | ".join(r["id"] for r in roots) + " |",
            "|---|" + "---:|" * len(roots),
        ]
        for name in (
            "dispatch",
            *INGRESS_MILESTONES,
            "engine_socket_received",
            "input_known",
        ):
            stamps = []
            for request in roots:
                point = next((b for b in request["chain"] if b["event"] == name), None)
                stamps.append(f"{point['since_origin_ms']:.3f}" if point else "未观测")
            lines.append("| " + name + " | " + " | ".join(stamps) + " |")
    lines += [
        "",
        "## 典型事件链",
        "",
    ]
    for kind, witness in witnesses.items():
        detail = (
            f"{witness['request_id']}，{witness['interval_ms']:.3f} ms；{witness['interpretation']}"
            if witness["observed"]
            else witness["reason"]
        )
        lines.append(f"- {kind}：{detail}")
    lines += [
        "",
        "事件链见 analysis.json；timeline.json.gz 可直接导入 Perfetto。",
        "",
        "## GPU 资源采样",
        "",
        f"Nsight GPU Metrics：{counters['nsight']['emitted_samples']} 个已对齐样本；"
        + (
            "可用。"
            if counters["nsight"]["available"]
            else counters["nsight"].get("reason", "未观测")
        ),
        f"NVML：{counters['nvml']['emitted_samples']} 个数值样本；"
        + (
            "可用。"
            if counters["nvml"]["available"]
            else counters["nvml"].get("reason", "未观测")
        ),
        "原生指标名、单位依据、设备映射及样本覆盖见 analysis.json 的 gpu_counters；"
        "Perfetto 中按 Nsight GPU Metrics / NVML 独立显示 counter。",
        "NVML GPU busy 不等于 SM active，memory busy 不等于 GB/s；"
        "设备级采样不归因到单个请求，也不代表模拟 CXL 的物理带宽。缺测不补零。",
        "",
        "## 解释边界",
        "",
    ]
    lines += ["- " + item for item in report["limits"]]
    lines += ["- 核验未通过：" + item for item in validity["failures"]]
    lines += ["- 尚未核验：" + item for item in validity["unconfirmed"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
