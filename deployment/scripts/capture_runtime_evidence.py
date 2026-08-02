"""Capture sanitized step 2 runtime evidence from the running Compose project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.runtime_evidence import (  # noqa: E402
    canonical_sha256,
    compatibility_projection,
    evidence_artifact_hashes,
    runtime_source_fingerprint,
    runtime_source_hashes,
)

COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"
ENV_FILE = REPOSITORY_ROOT / ".env.example"
SMOKE_SCRIPT = REPOSITORY_ROOT / "deployment/scripts/smoke_environment.py"
EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-2"
EVIDENCE_VALIDATOR = REPOSITORY_ROOT / "scripts/validate_runtime_evidence.py"
EXPECTED_SERVICES = {"triton", "prometheus", "grafana", "dcgm-exporter"}


def _run(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return process.stdout.strip()


def _compose(*arguments: str) -> str:
    return _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(REPOSITORY_ROOT),
            "--file",
            str(COMPOSE_FILE),
            "--env-file",
            str(ENV_FILE),
            *arguments,
        ]
    )


def _load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _parse_compose_ps(raw_output: str) -> list[dict[str, Any]]:
    if not raw_output:
        return []
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw_output.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def _format_publishers(record: dict[str, Any]) -> str:
    rendered = []
    for publisher in record.get("Publishers", []):
        if not isinstance(publisher, dict):
            continue
        rendered.append(
            f"{publisher.get('URL')}:{publisher.get('PublishedPort')}"
            f"->{publisher.get('TargetPort')}/{publisher.get('Protocol')}"
        )
    return ";".join(sorted(rendered))


def _capture_smoke() -> str:
    raw_output = _run(
        [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--env-file",
            str(ENV_FILE),
            "--format",
            "json",
        ]
    )
    payload = json.loads(raw_output)
    if payload.get("ok") is not True:
        raise RuntimeError("Smoke test did not report a successful runtime")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _capture_compose_ps() -> str:
    records = _parse_compose_ps(_compose("ps", "--format", "json"))
    services = {record.get("Service") for record in records}
    if services != EXPECTED_SERVICES:
        raise RuntimeError(
            "Compose evidence requires exactly four services: "
            f"found {', '.join(sorted(str(item) for item in services))}"
        )

    lines = ["service\timage\tstate\thealth\tpublished_ports"]
    for record in sorted(records, key=lambda item: str(item.get("Service"))):
        lines.append(
            "\t".join(
                (
                    str(record.get("Service", "")),
                    str(record.get("Image", "")),
                    str(record.get("State", "")),
                    str(record.get("Health", "")),
                    _format_publishers(record),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _capture_environment(env: dict[str, str]) -> str:
    triton_image = env["TRITON_IMAGE"]
    repo_digest = _run(
        [
            "docker",
            "image",
            "inspect",
            triton_image,
            "--format",
            "{{index .RepoDigests 0}}",
        ]
    )
    digest = repo_digest.rsplit("@", 1)[-1]
    triton_version = _run(
        [
            "docker",
            "image",
            "inspect",
            triton_image,
            "--format",
            '{{index .Config.Labels "com.nvidia.tritonserver.version"}}',
        ]
    )
    gpu_lines = _compose(
        "exec",
        "-T",
        "triton",
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ).splitlines()
    if not gpu_lines:
        raise RuntimeError("Triton container reported no visible NVIDIA GPUs")

    values = {
        "captured_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "compose_project": "ml-dev-ops",
        "compose_env_file": ".env.example",
        "docker_server_version": _run(
            ["docker", "version", "--format", "{{.Server.Version}}"]
        ),
        "docker_compose_version": _run(["docker", "compose", "version", "--short"]),
        "docker_operating_system": _run(
            ["docker", "info", "--format", "{{.OperatingSystem}}"]
        ),
        "triton_source_image": triton_image,
        "triton_source_digest": digest,
        "triton_server_version": triton_version,
        "gpu_count": str(len(gpu_lines)),
    }
    for index, gpu_line in enumerate(gpu_lines):
        parts = [part.strip() for part in gpu_line.split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi output: {gpu_line}")
        values[f"gpu_{index}_name"] = parts[0]
        values[f"gpu_{index}_driver_version"] = parts[1]
        values[f"gpu_{index}_compute_capability"] = parts[2]

    return "".join(f"{key}={value}\n" for key, value in values.items())


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _capture_integrity() -> str:
    source_hashes = runtime_source_hashes()
    projection = compatibility_projection()
    value = {
        "schema_version": 1,
        "runtime_source_revision": _run(["git", "rev-parse", "HEAD"]),
        "runtime_source_fingerprint_sha256": runtime_source_fingerprint(),
        "runtime_source_hashes": source_hashes,
        "runtime_source_manifest_sha256": canonical_sha256(source_hashes),
        "runtime_evidence_hashes": evidence_artifact_hashes(),
        "runtime_compatibility_projection": projection,
        "runtime_compatibility_projection_sha256": canonical_sha256(projection),
    }
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _validate_evidence(*, historical_only: bool = False) -> int:
    command = [sys.executable, str(EVIDENCE_VALIDATOR)]
    command.append("--historical-only" if historical_only else "--check")
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="Validate the committed snapshot without contacting Docker.",
    )
    modes.add_argument(
        "--historical-only",
        action="store_true",
        help="Validate only the immutable snapshot without contacting Docker.",
    )
    args = parser.parse_args()
    if args.check or args.historical_only:
        return _validate_evidence(historical_only=args.historical_only)

    try:
        env = _load_env()
        smoke = _capture_smoke()
        compose_ps = _capture_compose_ps()
        environment = _capture_environment(env)
        EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        _write_atomic(EVIDENCE_DIRECTORY / "smoke.json", smoke)
        _write_atomic(EVIDENCE_DIRECTORY / "compose-ps.txt", compose_ps)
        _write_atomic(EVIDENCE_DIRECTORY / "environment.txt", environment)
        _write_atomic(EVIDENCE_DIRECTORY / "runtime-integrity.json", _capture_integrity())
        validation_returncode = _validate_evidence()
        if validation_returncode != 0:
            return validation_returncode
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"[FAIL] Cannot capture runtime evidence: {error}", file=sys.stderr)
        return 1

    print(f"[OK] Runtime evidence captured in {EVIDENCE_DIRECTORY.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
