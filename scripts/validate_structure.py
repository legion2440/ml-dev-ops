"""Validate the step 1 repository scaffold and path policy."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "models",
    "deployment",
    "deployment/docker",
    "deployment/scripts",
    "deployment/triton",
    "monitoring",
    "monitoring/prometheus",
    "monitoring/grafana",
    "client",
    "client/samples",
    "client/logging",
    "benchmarks",
    "benchmarks/configs",
    "benchmarks/raw",
    "benchmarks/results",
    "scripts",
    "scripts/model_preparation",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "docs",
    "docs/generated",
    "schemas",
    "shared",
)

REQUIRED_FILES = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "README.md",
    "module-map.json",
    "dependency-graph.json",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    ".gitattributes",
    ".env.example",
    "docker-compose.yml",
    "schemas/module-map.schema.json",
    "schemas/dependency-graph.schema.json",
    "scripts/validate_structure.py",
    "scripts/validate_module_map.py",
    "scripts/generate_dependency_graph.py",
    "docs/troubleshooting.md",
    "docs/audit-evidence.md",
    "docs/generated/dependency-graph.md",
)

METADATA_FILES = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "README.md",
    "module-map.json",
    "dependency-graph.json",
)

WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")
PARENT_SEGMENT = re.compile(r"(?<![A-Za-z0-9._-])\.\.(?:/|\\)")
LF_TEXT_SUFFIXES = {
    ".csv",
    ".env",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LF_TEXT_NAMES = {"Makefile", ".env.example", ".gitattributes"}


def _metadata_paths() -> list[Path]:
    paths = [REPOSITORY_ROOT / relative_path for relative_path in METADATA_FILES]
    paths.extend(REPOSITORY_ROOT.glob("docs/**/*.md"))
    paths.extend(
        path
        for path in REPOSITORY_ROOT.glob("**/README.md")
        if ".git" not in path.parts
    )
    return sorted(set(paths))


def _check_required_paths(errors: list[str]) -> None:
    for relative_path in REQUIRED_DIRECTORIES:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_dir():
            errors.append(f"Missing required directory: {relative_path}")

    for relative_path in REQUIRED_FILES:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing required file: {relative_path}")


def _check_json_files(errors: list[str]) -> None:
    json_paths = (
        "module-map.json",
        "dependency-graph.json",
        "schemas/module-map.schema.json",
        "schemas/dependency-graph.schema.json",
    )
    for relative_path in json_paths:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as json_file:
                json.load(json_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"Invalid JSON in {relative_path}: {error}")


def _check_metadata_paths(errors: list[str]) -> None:
    for path in _metadata_paths():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            errors.append(f"Cannot read {path.relative_to(REPOSITORY_ROOT).as_posix()}: {error}")
            continue

        relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
        checks = (
            ("Windows absolute path", WINDOWS_ABSOLUTE_PATH),
            ("UNC path", UNC_PATH),
            ("parent path segment", PARENT_SEGMENT),
        )
        for label, pattern in checks:
            match = pattern.search(text)
            if match:
                errors.append(
                    f"{relative_path} contains a forbidden {label}: {match.group(0)!r}"
                )


def _check_lf_line_endings(errors: list[str]) -> None:
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix not in LF_TEXT_SUFFIXES and path.name not in LF_TEXT_NAMES:
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            errors.append(f"Cannot read {path.relative_to(REPOSITORY_ROOT).as_posix()}: {error}")
            continue
        if b"\r\n" in content:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            errors.append(f"{relative_path} uses CRLF; repository text files must use LF")


def main() -> int:
    errors: list[str] = []
    _check_required_paths(errors)
    _check_json_files(errors)
    _check_metadata_paths(errors)
    _check_lf_line_endings(errors)

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Structure validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print("[OK] Repository structure, metadata paths, JSON, and line endings are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
