# SPDX-License-Identifier: Apache-2.0
"""Summarize Gate A checksum, trace, and memory-counter path evidence."""

# Standard
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json


def _read_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None


def _raw_artifacts_exist(report_path: Path, report: dict[str, Any]) -> bool:
    artifacts = report.get("raw_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    return all(
        isinstance(item, str) and (report_path.parent / item).is_file()
        for item in artifacts
    )


def _trace_is_usable(path: Path | None) -> bool:
    report = _read_report(path)
    if path is None or report is None or report.get("usable") is not True:
        return False
    return (
        report.get("host_memcpy_count") == 0
        and report.get("unexpected_cuda_memory_operation_count") == 0
        and _raw_artifacts_exist(path, report)
    )


def _counters_exclude_bounce(path: Path | None) -> bool:
    report = _read_report(path)
    if path is None or report is None or report.get("usable") is not True:
        return False
    try:
        payload_bytes = int(report["payload_bytes"])
        direct_bytes = int(report["direct_traffic_bytes"])
        staged_bytes = int(report["staged_baseline_traffic_bytes"])
    except (KeyError, TypeError, ValueError):
        return False
    if payload_bytes <= 0 or direct_bytes < 0 or staged_bytes < 0:
        return False
    direct_ratio = direct_bytes / payload_bytes
    staged_ratio = staged_bytes / payload_bytes
    return (
        direct_ratio <= 1.5
        and staged_ratio >= 1.75
        and staged_ratio - direct_ratio >= 0.5
        and _raw_artifacts_exist(path, report)
    )


def summarize_evidence(
    command_json: Path,
    trace_report: Path | None,
    counter_report: Path | None,
) -> dict[str, Any]:
    """Summarize explicit Gate A path evidence.

    Args:
        command_json: Direct-copy benchmark result.
        trace_report: JSON report declaring whether trace evidence is usable.
        counter_report: JSON report declaring whether counter evidence is usable.

    Returns:
        Machine-readable Gate A status. Missing, malformed, or explicitly
        unusable evidence blocks the gate.

    Raises:
        OSError: If the benchmark result cannot be read.
        json.JSONDecodeError: If the benchmark result is not valid JSON.
    """
    benchmark = json.loads(command_json.read_text())
    trace_available = _trace_is_usable(trace_report)
    counters_available = _counters_exclude_bounce(counter_report)
    checksums_ok = bool(benchmark.get("all_checksums_ok"))
    direct_contract = (
        benchmark.get("path") == "cuda_registered_posix_shm_direct"
        and benchmark.get("payload_staging") is False
    )
    no_bounce_status = "pass" if trace_available and counters_available else "blocked"
    return {
        "schema_version": 1,
        "benchmark": str(command_json),
        "checksums": "pass" if checksums_ok else "fail",
        "direct_path_contract": "pass" if direct_contract else "fail",
        "trace_report": str(trace_report) if trace_available else None,
        "counter_report": str(counter_report) if counters_available else None,
        "no_payload_bounce_evidence": no_bounce_status,
        "gate_a_status": (
            "pass"
            if checksums_ok and direct_contract and no_bounce_status == "pass"
            else "blocked"
            if checksums_ok and direct_contract
            else "fail"
        ),
        "note": (
            "No-bounce remains blocked unless both a CUDA/OS runtime trace and "
            "usable CPU memory-controller counters are supplied."
        ),
    }


def main() -> None:
    """Validate benchmark evidence without inferring unavailable counters.

    Raises:
        OSError: If an input or output artifact cannot be accessed.
        json.JSONDecodeError: If the benchmark JSON is malformed.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path)
    parser.add_argument("--counter-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarize_evidence(
        args.command_json, args.trace_report, args.counter_report
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
