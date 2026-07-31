"""Validate committed step 2 runtime evidence without requiring a Docker daemon."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-2"
SMOKE_PATH = EVIDENCE_DIRECTORY / "smoke.json"
COMPOSE_PS_PATH = EVIDENCE_DIRECTORY / "compose-ps.txt"
ENVIRONMENT_PATH = EVIDENCE_DIRECTORY / "environment.txt"
ENV_EXAMPLE_PATH = REPOSITORY_ROOT / ".env.example"
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
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
COMPOSE_VARIABLE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):\?[^}]+\}$")
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


def _load_compose() -> dict[str, Any]:
    value = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("docker-compose.yml must contain a YAML object")
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


def _resolve_image(expression: Any, env: dict[str, str]) -> str:
    if not isinstance(expression, str):
        return ""
    match = COMPOSE_VARIABLE.fullmatch(expression)
    return env.get(match.group(1), "") if match else expression


def _validate_compose_ps(
    env: dict[str, str],
    compose: dict[str, Any],
    errors: list[str],
) -> None:
    with COMPOSE_PS_PATH.open(encoding="utf-8", newline="") as evidence_file:
        reader = csv.DictReader(evidence_file, delimiter="\t")
        rows = list(reader)
        expected_fields = ["service", "image", "state", "health", "published_ports"]
        if reader.fieldnames != expected_fields:
            errors.append("compose-ps.txt has an invalid header")
            return

    services = [row["service"] for row in rows]
    if len(services) != len(set(services)) or set(services) != EXPECTED_SERVICES:
        errors.append("compose-ps.txt must contain exactly the four step 2 services")

    compose_services = compose.get("services", {})
    expected_images = {
        service: _resolve_image(config.get("image"), env)
        for service, config in compose_services.items()
        if isinstance(config, dict)
    }
    for row in rows:
        service = row["service"]
        if row["image"] != expected_images.get(service):
            errors.append(f"compose-ps.txt image is stale for {service}")
        if row["state"] != "running" or row["health"] != "healthy":
            errors.append(f"compose-ps.txt does not record {service} as running and healthy")
        publishers = row["published_ports"].split(";") if row["published_ports"] else []
        if not publishers or any(not item.startswith("127.0.0.1:") for item in publishers):
            errors.append(f"compose-ps.txt has invalid loopback publishers for {service}")


def _validate_environment(env: dict[str, str], errors: list[str]) -> None:
    values = _load_env(ENVIRONMENT_PATH)
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
    if values["triton_source_image"] != env.get("TRITON_IMAGE"):
        errors.append("environment.txt Triton image is stale relative to .env.example")
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


def _validate_sanitization(errors: list[str]) -> None:
    for path in (SMOKE_PATH, COMPOSE_PS_PATH, ENVIRONMENT_PATH):
        content = path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content) or "/mnt/" in content:
            errors.append(f"{path.name} contains a host-specific path")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{path.name} must use LF line endings")


def main() -> int:
    errors: list[str] = []
    try:
        env = _load_env(ENV_EXAMPLE_PATH)
        compose = _load_compose()
        smoke = _load_json(SMOKE_PATH)
        _validate_smoke(smoke, errors)
        _validate_compose_ps(env, compose, errors)
        _validate_environment(env, errors)
        _validate_sanitization(errors)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"[FAIL] Cannot load runtime evidence: {error}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Runtime evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print("[OK] Runtime evidence records 10 successful checks and four healthy services.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
