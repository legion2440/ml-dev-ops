"""Validate committed step 2 runtime evidence without requiring a Docker daemon."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.runtime_evidence import (  # noqa: E402
    canonical_sha256,
    compatibility_projection,
    evidence_artifact_hashes,
)

EVIDENCE_RELATIVE = Path("docs/evidence/step-2")
INTEGRITY_NAME = "runtime-integrity.json"
INTEGRITY_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/runtime-integrity.schema.json"
EXPECTED_SERVICES = {"triton", "prometheus", "grafana", "dcgm-exporter"}
EXPECTED_SMOKE_CHECKS = {
    "containers",
    "triton_liveness",
    "triton_readiness",
    "triton_repository",
    "triton_metrics",
    "prometheus_health",
    "prometheus_targets",
    "grafana_health",
    "grafana_datasource",
    "dcgm_metrics",
}
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+._][A-Za-z0-9.-]+)?$")
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path.name}:{line_number} is not KEY=VALUE")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"{path.name}:{line_number} duplicates {key}")
        values[key] = value
    return values


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _validate_smoke(smoke: dict[str, Any], errors: list[str]) -> None:
    if smoke.get("ok") is not True:
        errors.append("smoke.json must record ok=true")
    if smoke.get("env_file") != ".env.example":
        errors.append("smoke.json must record the canonical .env.example")

    checks = smoke.get("checks", [])
    if not isinstance(checks, list):
        errors.append("smoke.json checks must be an array")
        return
    names = [check.get("name") for check in checks if isinstance(check, dict)]
    if len(names) != len(set(names)):
        errors.append("smoke.json contains duplicate check names")
    if set(names) != EXPECTED_SMOKE_CHECKS:
        errors.append("smoke.json does not contain the exact step 2 check set")
    for check in checks:
        if not isinstance(check, dict) or check.get("ok") is not True:
            name = check.get("name", "<unknown>") if isinstance(check, dict) else "<invalid>"
            errors.append(f"smoke.json check is not successful: {name}")


def _validate_compose_ps(
    path: Path,
    expected_images: dict[str, Any],
    errors: list[str],
) -> None:
    with path.open(encoding="utf-8", newline="") as evidence_file:
        reader = csv.DictReader(evidence_file, delimiter="\t")
        rows = list(reader)
        expected_fields = ["service", "image", "state", "health", "published_ports"]
        if reader.fieldnames != expected_fields:
            errors.append("compose-ps.txt has an invalid header")
            return

    services = [row["service"] for row in rows]
    if len(services) != len(set(services)) or set(services) != EXPECTED_SERVICES:
        errors.append("compose-ps.txt must contain exactly the four step 2 services")

    for row in rows:
        service = row["service"]
        if row["image"] != expected_images.get(service):
            errors.append(f"compose-ps.txt image is stale for {service}")
        if row["state"] != "running" or row["health"] != "healthy":
            errors.append(f"compose-ps.txt does not record {service} as running and healthy")
        publishers = row["published_ports"].split(";") if row["published_ports"] else []
        if not publishers or any(not item.startswith("127.0.0.1:") for item in publishers):
            errors.append(f"compose-ps.txt has invalid loopback publishers for {service}")


def _validate_environment(
    path: Path, expected_triton_source_image: str, errors: list[str]
) -> None:
    values = _load_env(path)
    required = {
        "captured_at_utc",
        "compose_project",
        "compose_env_file",
        "docker_server_version",
        "docker_compose_version",
        "docker_operating_system",
        "triton_source_image",
        "triton_source_digest",
        "triton_server_version",
        "gpu_count",
    }
    missing = sorted(required - set(values))
    if missing:
        errors.append(f"environment.txt is missing fields: {', '.join(missing)}")
        return

    try:
        captured_at = datetime.fromisoformat(values["captured_at_utc"].replace("Z", "+00:00"))
        if captured_at.tzinfo is None:
            errors.append("environment.txt capture timestamp must include UTC timezone")
    except ValueError:
        errors.append("environment.txt has an invalid capture timestamp")

    if values["compose_project"] != "ml-dev-ops":
        errors.append("environment.txt has an unexpected Compose project")
    if values["compose_env_file"] != ".env.example":
        errors.append("environment.txt must record .env.example")
    if values["triton_source_image"] != expected_triton_source_image:
        errors.append("environment.txt Triton image differs from the runtime snapshot")
    if not DIGEST.fullmatch(values["triton_source_digest"]):
        errors.append("environment.txt has an invalid Triton digest")
    for key in ("docker_server_version", "docker_compose_version", "triton_server_version"):
        if not VERSION.fullmatch(values[key]):
            errors.append(f"environment.txt has an invalid {key}")
    if not values["docker_operating_system"].strip():
        errors.append("environment.txt must record the Docker operating system")

    try:
        gpu_count = int(values["gpu_count"])
    except ValueError:
        gpu_count = 0
    if gpu_count < 1:
        errors.append("environment.txt must record at least one GPU")
    allowed_fields = set(required)
    for index in range(gpu_count):
        gpu_fields = {
            f"gpu_{index}_name",
            f"gpu_{index}_driver_version",
            f"gpu_{index}_compute_capability",
        }
        allowed_fields.update(gpu_fields)
        if not gpu_fields.issubset(values):
            errors.append(f"environment.txt is missing fields for GPU {index}")
            continue
        if not values[f"gpu_{index}_name"].strip():
            errors.append(f"environment.txt has an empty name for GPU {index}")
        if not re.fullmatch(r"\d+(?:\.\d+)+", values[f"gpu_{index}_driver_version"]):
            errors.append(f"environment.txt has an invalid driver version for GPU {index}")
        if not re.fullmatch(r"\d+\.\d+", values[f"gpu_{index}_compute_capability"]):
            errors.append(f"environment.txt has an invalid compute capability for GPU {index}")

    unexpected = sorted(set(values) - allowed_fields)
    if unexpected:
        errors.append(
            "environment.txt contains non-whitelisted fields: " + ", ".join(unexpected)
        )

    sensitive_markers = ("PASSWORD", "TOKEN", "SECRET", "API_KEY")
    if any(marker in key.upper() for key in values for marker in sensitive_markers):
        errors.append("environment.txt contains a sensitive field name")


def _validate_sanitization(evidence_directory: Path, errors: list[str]) -> None:
    for name in ("smoke.json", "compose-ps.txt", "environment.txt", INTEGRITY_NAME):
        path = evidence_directory / name
        content = path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content) or "/mnt/" in content:
            errors.append(f"{path.name} contains a host-specific path")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{path.name} must use LF line endings")


def validate_historical(artifact_root: Path = REPOSITORY_ROOT) -> list[str]:
    """Validate only the captured Step 2 runtime and its historical manifest."""
    errors: list[str] = []
    evidence_directory = artifact_root / EVIDENCE_RELATIVE
    integrity = _load_json(evidence_directory / INTEGRITY_NAME)
    schema = _load_json(INTEGRITY_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors.extend(
        f"runtime-integrity.{'.'.join(str(part) for part in error.path) or '$'}: "
        f"{error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(integrity),
            key=lambda item: list(item.path),
        )
    )
    source_manifest = integrity.get("runtime_source_hashes")
    projection = integrity.get("runtime_compatibility_projection")
    if not isinstance(source_manifest, dict) or not isinstance(projection, dict):
        return errors or ["runtime integrity snapshot is incomplete"]
    if integrity.get("runtime_source_manifest_sha256") != canonical_sha256(
        source_manifest
    ):
        errors.append("runtime source manifest hash is stale")
    if integrity.get("runtime_compatibility_projection_sha256") != canonical_sha256(
        projection
    ):
        errors.append("runtime compatibility projection hash is stale")
    expected_artifacts = integrity.get("runtime_evidence_hashes")
    if expected_artifacts != evidence_artifact_hashes(artifact_root):
        errors.append("Step 2 runtime artifact hashes are stale")
    smoke = _load_json(evidence_directory / "smoke.json")
    _validate_smoke(smoke, errors)
    images = projection.get("images", {})
    _validate_compose_ps(evidence_directory / "compose-ps.txt", images, errors)
    expected_triton_image = projection.get("triton_runtime", {}).get("source_image")
    if not isinstance(expected_triton_image, str):
        errors.append("runtime projection has no Triton source image")
    else:
        _validate_environment(
            evidence_directory / "environment.txt", expected_triton_image, errors
        )
    _validate_sanitization(evidence_directory, errors)
    return errors


def validate_current_compatibility(
    integrity: dict[str, Any], source_root: Path = REPOSITORY_ROOT
) -> list[str]:
    current = compatibility_projection(source_root)
    if current != integrity.get("runtime_compatibility_projection") or canonical_sha256(
        current
    ) != integrity.get("runtime_compatibility_projection_sha256"):
        return ["current deployment contract is incompatible with the Step 2 run"]
    return []


def source_drift(
    integrity: dict[str, Any], source_root: Path = REPOSITORY_ROOT
) -> list[str]:
    manifest = integrity.get("runtime_source_hashes", {})
    if not isinstance(manifest, dict):
        return []
    changed: list[str] = []
    for relative, expected in manifest.items():
        path = source_root / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            changed.append(relative)
    return sorted(changed)


def validate(
    artifact_root: Path = REPOSITORY_ROOT,
    *,
    historical_only: bool = False,
    source_root: Path = REPOSITORY_ROOT,
) -> list[str]:
    errors = validate_historical(artifact_root)
    if not historical_only:
        integrity_path = artifact_root / EVIDENCE_RELATIVE / INTEGRITY_NAME
        if integrity_path.is_file():
            errors.extend(
                validate_current_compatibility(_load_json(integrity_path), source_root)
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="check historical integrity and current semantic compatibility (default)",
    )
    modes.add_argument(
        "--historical-only",
        action="store_true",
        help="check only the immutable historical runtime snapshot",
    )
    args = parser.parse_args()
    try:
        errors = validate(historical_only=args.historical_only)
        integrity = _load_json(
            REPOSITORY_ROOT / EVIDENCE_RELATIVE / INTEGRITY_NAME
        )
        drift = [] if args.historical_only else source_drift(integrity)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        errors = [f"Cannot load runtime evidence: {error}"]
        drift = []

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Runtime evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1

    if drift:
        print(
            "[INFO] Step 2 historical source drift (non-gating): "
            + ", ".join(drift)
        )
    if args.historical_only:
        print("[OK] Step 2 historical runtime integrity passes.")
    else:
        print(
            "[OK] Step 2 historical integrity and current compatibility pass; "
            "10 checks and four healthy services remain proven."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
