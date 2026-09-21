# SPDX-License-Identifier: Apache-2.0
"""Package the fixed-base source overlay and immutable measured evidence."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import subprocess
import time
import zipfile


def digest(path: Path) -> str:
    """Return streaming SHA-256 without reading large profiles into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    """Write new source/evidence manifests and a verified standalone ZIP.

    Raises:
        FileExistsError: If a delivery output would replace existing evidence.
        RuntimeError: If the audit, base SHA or archive checksum validation fails.
        ValueError: If the outer archive would recursively include itself.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    examples = workspace / "LMCache/examples/l2_hybrid"
    if args.archive.resolve().is_relative_to(examples):
        raise ValueError("outer archive must be outside the included example tree")
    audit = json.loads((examples / "evidence/delivery-audit.json").read_text())
    if not audit["passed"]:
        raise RuntimeError("delivery acceptance audit has not passed")
    args.out.mkdir(parents=True, exist_ok=False)
    bases = {
        "LMCache": "05a013b29da78cf2321b9b46ec5039dde2fb0bb0",
        "vllm": "98dff2a81d747d1dba01a47f939f48c3526d4206",
        "CXLMemSim": "992d5f522e7dd4c117340841d9193cd2152107f8",
    }
    sources: set[Path] = set()
    for name, expected in bases.items():
        repo = workspace / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
        if actual.decode().strip() != expected:
            raise RuntimeError(f"fixed source base changed: {name}")
        patch = subprocess.check_output(["git", "diff", "--binary", expected], cwd=repo)
        (args.out / (name + ".patch")).write_bytes(patch)
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "-z", expected], cwd=repo
        ).decode()
        sources.update(repo / path for path in changed.split("\0") if path)
        if name == "LMCache":
            untracked = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo
            ).decode()
            for relative in untracked.split("\0"):
                if not relative or relative.startswith("examples/l2_hybrid/evidence/"):
                    continue
                file = repo / relative
                if file.suffix in (".py", ".md", ".rst", ".json"):
                    sources.add(file)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "captured_wall_ns": time.time_ns(),
        "fixed_bases": bases,
        "scope": "final source overlay; patches contain tracked changes only",
        "files": {
            str(path.relative_to(workspace)): {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(sources)
        },
        "native_binaries": {
            str(path.relative_to(workspace)): digest(path)
            for path in (
                workspace / "CXLMemSim/build-l2-hybrid/cxlmemsim_server",
                workspace / "CXLMemSim/build-l2-hybrid/libcxlmemsim_modeled_client.so",
            )
        },
    }
    (args.out / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    with zipfile.ZipFile(
        args.out / "source-overlay.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in sorted(sources):
            z.write(path, str(path.relative_to(workspace)))
    evidence = examples / "evidence"
    index = {
        str(path.relative_to(evidence)): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(evidence.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    (args.out / "evidence-manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": index}, indent=2) + "\n"
    )
    include = sources | {
        path
        for path in examples.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    with zipfile.ZipFile(args.archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(include):
            z.write(path, str(path.relative_to(workspace)))
    with zipfile.ZipFile(args.archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"archive CRC check failed: {bad}")
        for name, entry in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive source checksum failed: {name}")
    result = {
        "schema_version": 1,
        "passed": True,
        "archive": str(args.archive.resolve()),
        "sha256": digest(args.archive),
        "bytes": args.archive.stat().st_size,
        "source_files": len(sources),
        "evidence_files": len(index),
        "excluded": "base repos, builds, environments, model weights, unrelated files",
        "self_hash_scope": "delivery-result.json is created after archive closure",
    }
    (args.out / "delivery-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
