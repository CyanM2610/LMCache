# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from benchmarks.beluga_proxy.capture_path_evidence import summarize_evidence


pytestmark = pytest.mark.no_shared_allocator


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value))
    return path


def _trace_report(tmp_path: Path) -> Path:
    raw = tmp_path / "trace.nsys-rep"
    raw.write_bytes(b"trace")
    return _write_json(
        tmp_path / "trace.json",
        {
            "usable": True,
            "host_memcpy_count": 0,
            "unexpected_cuda_memory_operation_count": 0,
            "raw_artifacts": [raw.name],
        },
    )


def _counter_report(tmp_path: Path, *, usable: bool) -> Path:
    raw = tmp_path / "perf-stat.txt"
    raw.write_text("raw counters")
    return _write_json(
        tmp_path / "counters.json",
        {
            "usable": usable,
            "payload_bytes": 1048576,
            "direct_traffic_bytes": 1048576,
            "staged_baseline_traffic_bytes": 2097152,
            "raw_artifacts": [raw.name],
        },
    )


def test_no_bounce_evidence_stays_blocked_for_unusable_counter_report(
    tmp_path: Path,
) -> None:
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {
            "all_checksums_ok": True,
            "path": "cuda_registered_posix_shm_direct",
            "payload_staging": False,
        },
    )
    trace = _trace_report(tmp_path)
    counters = _counter_report(tmp_path, usable=False)

    result = summarize_evidence(benchmark, trace, counters)

    assert result["no_payload_bounce_evidence"] == "blocked"
    assert result["gate_a_status"] == "blocked"


def test_no_bounce_evidence_passes_only_with_explicitly_usable_reports(
    tmp_path: Path,
) -> None:
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {
            "all_checksums_ok": True,
            "path": "cuda_registered_posix_shm_direct",
            "payload_staging": False,
        },
    )
    trace = _trace_report(tmp_path)
    counters = _counter_report(tmp_path, usable=True)

    result = summarize_evidence(benchmark, trace, counters)

    assert result["no_payload_bounce_evidence"] == "pass"
    assert result["gate_a_status"] == "pass"


def test_usable_flag_without_raw_metrics_cannot_pass_evidence(tmp_path: Path) -> None:
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {
            "all_checksums_ok": True,
            "path": "cuda_registered_posix_shm_direct",
            "payload_staging": False,
        },
    )
    trace = _write_json(tmp_path / "trace.json", {"usable": True})
    counters = _write_json(tmp_path / "counters.json", {"usable": True})

    result = summarize_evidence(benchmark, trace, counters)

    assert result["no_payload_bounce_evidence"] == "blocked"
