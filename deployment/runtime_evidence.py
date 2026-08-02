"""Step 2 runtime source identity and semantic compatibility projection."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_PATHS = (
    ".env.example",
    "docker-compose.yml",
    "deployment/docker/Dockerfile",
    "deployment/runtime_evidence.py",
    "deployment/scripts/capture_runtime_evidence.py",
    "deployment/scripts/smoke_environment.py",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "schemas/runtime-integrity.schema.json",
    "scripts/validate_deployment.py",
    "scripts/validate_runtime_evidence.py",
)
COMPOSE_VARIABLE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):\?[^}]+\}$")


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a YAML object")
    return value


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _resolve(value: Any, env: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    match = COMPOSE_VARIABLE.fullmatch(value)
    return env.get(match.group(1), "") if match else value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_source_hashes(
    root: Path = REPOSITORY_ROOT,
    paths: Iterable[str] = RUNTIME_SOURCE_PATHS,
) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in paths
    }


def runtime_source_fingerprint(
    root: Path = REPOSITORY_ROOT,
    paths: Iterable[str] = RUNTIME_SOURCE_PATHS,
) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _port_contract(
    service: dict[str, Any], env: dict[str, str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in service.get("ports", []):
        if not isinstance(raw, str):
            continue
        host, remainder = raw.split(":", 1)
        published, target = remainder.rsplit(":", 1)
        result.append(
            {
                "host": host.strip('"'),
                "published": _resolve(published, env),
                "target": int(target.strip('"')),
            }
        )
    return result


def _prometheus_scrape_projection(root: Path) -> dict[str, Any]:
    config = _yaml(root / "monitoring/prometheus/prometheus.yml")
    jobs = []
    for job in config.get("scrape_configs", []):
        if not isinstance(job, dict):
            continue
        jobs.append(
            {
                "job_name": job.get("job_name"),
                "metrics_path": job.get("metrics_path", "/metrics"),
                "static_configs": job.get("static_configs", []),
            }
        )
    return {
        "scrape_interval": config.get("global", {}).get("scrape_interval"),
        "jobs": jobs,
    }


def compatibility_projection(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Return the current four-service contract relevant to the Step 2 run."""
    compose = _yaml(root / "docker-compose.yml")
    env = _env(root / ".env.example")
    services = compose.get("services", {})
    expected = ("triton", "prometheus", "grafana", "dcgm-exporter")
    if not isinstance(services, dict) or any(
        not isinstance(services.get(name), dict) for name in expected
    ):
        raise ValueError("Compose is missing a persistent Step 2 service")

    image_contract = {
        name: _resolve(services[name].get("image"), env) for name in expected
    }
    triton_commands = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in services["triton"].get("command", [])
        if isinstance(item, str)
        and "=" in item
        and item.split("=", 1)[0]
        in {
            "--model-repository",
            "--model-control-mode",
            "--allow-http",
            "--http-port",
            "--allow-grpc",
            "--grpc-port",
            "--allow-metrics",
            "--metrics-port",
        }
    }
    datasource = _yaml(
        root / "monitoring/grafana/provisioning/datasources/prometheus.yml"
    )
    datasource_contract = [
        {
            key: item.get(key)
            for key in ("name", "type", "access", "url", "isDefault")
        }
        for item in datasource.get("datasources", [])
        if isinstance(item, dict)
    ]
    return {
        "schema_version": 1,
        "acceptance": {
            "persistent_services": list(expected),
            "successful_checks": [
                "containers",
                "triton_liveness",
                "triton_readiness",
                "triton_no_ready_models",
                "triton_metrics",
                "prometheus_health",
                "prometheus_targets",
                "grafana_health",
                "grafana_datasource",
                "dcgm_metrics",
            ],
            "all_services_healthy": True,
            "published_ports_loopback_only": True,
            "gpu_visible": True,
        },
        "images": image_contract,
        "ports": {name: _port_contract(services[name], env) for name in expected},
        "healthchecks": {
            name: services[name].get("healthcheck") for name in expected
        },
        "triton_runtime": {
            "source_image": env.get("TRITON_IMAGE"),
            "build": services["triton"].get("build"),
            "command": triton_commands,
            "model_repository_mount": [
                volume
                for volume in services["triton"].get("volumes", [])
                if isinstance(volume, dict) and volume.get("target") == "/models"
            ],
            "gpu_reservation": services["triton"].get("deploy"),
        },
        "dcgm_gpu_reservation": services["dcgm-exporter"].get("deploy"),
        "network": compose.get("networks", {}).get("backend"),
        "prometheus_scrape": _prometheus_scrape_projection(root),
        "grafana_datasource": datasource_contract,
    }


def evidence_artifact_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    evidence = root / "docs/evidence/step-2"
    return {
        name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
        for name in ("smoke.json", "compose-ps.txt", "environment.txt")
    }
