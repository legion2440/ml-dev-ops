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
    "monitoring/grafana/provisioning",
    "monitoring/grafana/provisioning/datasources",
    "monitoring/grafana/provisioning/dashboards",
    "monitoring/grafana/dashboards",
    "client",
    "client/samples",
    "client/logging",
    "benchmarks",
    "benchmarks/configs",
    "benchmarks/results",
    "benchmarks/results/raw",
    "scripts",
    "scripts/model_preparation",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "docs",
    "docs/evidence",
    "docs/evidence/step-2",
    "docs/evidence/step-3",
    "docs/evidence/step-4",
    "docs/evidence/portability",
    "docs/generated",
    "schemas",
    "shared",
    "models/resnet50_onnx",
    "models/resnet50_tensorrt",
    "models/yolo11n_onnx",
    "tests/unit/model_preparation",
)

REQUIRED_FILES = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "module-map.json",
    "dependency-graph.json",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    ".gitattributes",
    ".env.example",
    "docker-compose.yml",
    "deployment/docker/Dockerfile",
    "deployment/scripts/compose_common.sh",
    "deployment/scripts/run_environment.sh",
    "deployment/scripts/stop_environment.sh",
    "deployment/scripts/check_environment.sh",
    "deployment/scripts/capture_runtime_evidence.py",
    "deployment/scripts/run_triton.sh",
    "deployment/scripts/smoke_environment.py",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "monitoring/grafana/provisioning/dashboards/provider.yml",
    "schemas/module-map.schema.json",
    "schemas/dependency-graph.schema.json",
    "scripts/validate_structure.py",
    "scripts/validate_module_map.py",
    "scripts/validate_deployment.py",
    "scripts/validate_runtime_evidence.py",
    "scripts/validate_model_repository.py",
    "scripts/validate_model_evidence.py",
    "scripts/model_preparation/prepare_models.py",
    "scripts/model_preparation/Dockerfile.exporter",
    "scripts/model_preparation/requirements.lock",
    "scripts/model_preparation/yolo_export_adapter.py",
    "deployment/triton/smoke_models.py",
    "tests/unit/test_deployment_validation.py",
    "scripts/generate_dependency_graph.py",
    "docs/troubleshooting.md",
    "docs/audit-evidence.md",
    "docs/evidence/step-2/smoke.json",
    "docs/evidence/step-2/compose-ps.txt",
    "docs/evidence/step-2/environment.txt",
    "docs/evidence/step-3/preparation.json",
    "docs/evidence/step-3/triton-model-smoke.json",
    "docs/evidence/step-3/model-repository.txt",
    "docs/evidence/step-4/runtime-integrity.json",
    "docs/evidence/step-4/runtime-model-manifest.json",
    "docs/evidence/step-4/runtime-model-spec.yaml",
    "docs/evidence/portability/build-record.json",
    "docs/evidence/portability/serving-runtime.json",
    "docs/evidence/portability/repository-versions.txt",
    "models/model-spec.yaml",
    "models/model-manifest.json",
    "models/resnet50_onnx/config.pbtxt",
    "models/resnet50_onnx/imagenet1k.txt",
    "models/resnet50_tensorrt/config.pbtxt",
    "models/resnet50_tensorrt/imagenet1k.txt",
    "models/yolo11n_onnx/config.pbtxt",
    "models/yolo11n_onnx/coco80.txt",
    "schemas/model-spec.schema.json",
    "schemas/model-manifest.schema.json",
    "schemas/portability-build-record.schema.json",
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
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LF_TEXT_NAMES = {
    "Makefile",
    "Dockerfile",
    "Dockerfile.exporter",
    "LICENSE",
    ".env.example",
    ".gitattributes",
}


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
        "schemas/model-spec.schema.json",
        "schemas/model-manifest.schema.json",
        "schemas/portability-build-record.schema.json",
        "docs/evidence/step-4/runtime-integrity.json",
        "docs/evidence/step-4/runtime-model-manifest.json",
        "docs/evidence/portability/build-record.json",
        "docs/evidence/portability/serving-runtime.json",
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
