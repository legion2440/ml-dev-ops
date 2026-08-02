#!/usr/bin/env python3
"""Reject tracked cache, binary-model, secret, and host-specific evidence junk."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SEGMENTS = {".cache", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".engine",
    ".onnx",
    ".plan",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".tmp",
}
WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")
POSIX_HOST_PATH = re.compile(r"(?<![\w:/])/(?:home|mnt|Users|tmp|var)/")
SECRET_VALUE = re.compile(
    r"(?i)(?:api[_-]?key|authorization|bearer|password|secret|token)"
    r"\s*[:=]\s*(?![<${])(?!change-me)[\"']?[A-Za-z0-9_+/.=-]{12,}"
)


def _tracked_paths() -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail}")
    return [
        Path(raw.decode("utf-8"))
        for raw in process.stdout.split(b"\0")
        if raw
    ]


def validate() -> list[str]:
    errors: list[str] = []
    tracked = _tracked_paths()
    for relative in tracked:
        if any(part in FORBIDDEN_SEGMENTS for part in relative.parts):
            errors.append(f"tracked cache/runtime path: {relative.as_posix()}")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"tracked model/runtime binary: {relative.as_posix()}")
        if relative.name == ".env":
            errors.append("tracked local .env file")

    evidence_paths = [
        relative
        for relative in tracked
        if relative.parts[:2] == ("docs", "evidence")
        or relative.parts[:2] == ("benchmarks", "results")
        or relative.as_posix() == "benchmarks/report.md"
    ]
    for relative in evidence_paths:
        path = REPOSITORY_ROOT / relative
        data = path.read_bytes()
        if b"\r\n" in data:
            errors.append(f"tracked evidence is not LF-only: {relative.as_posix()}")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"tracked evidence is not UTF-8 text: {relative.as_posix()}")
            continue
        if (
            WINDOWS_PATH.search(content)
            or UNC_PATH.search(content)
            or POSIX_HOST_PATH.search(content)
        ):
            errors.append(f"host-specific path in evidence: {relative.as_posix()}")
        if "-----BEGIN PRIVATE KEY-----" in content or SECRET_VALUE.search(content):
            errors.append(f"secret-like value in evidence: {relative.as_posix()}")
    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, RuntimeError, UnicodeDecodeError) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(
            f"[FAIL] Repository hygiene validation found {len(errors)} issue(s).",
            file=sys.stderr,
        )
        return 1
    print("[OK] Tracked evidence contains no cache, model binaries, host paths, or secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
