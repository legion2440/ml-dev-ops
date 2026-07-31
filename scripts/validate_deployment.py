"""Validate the step 2 Docker infrastructure without starting containers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = REPOSITORY_ROOT / ".env.example"
PROMETHEUS_PATH = REPOSITORY_ROOT / "monitoring/prometheus/prometheus.yml"
DATASOURCE_PATH = (
    REPOSITORY_ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml"
)
DASHBOARD_PROVIDER_PATH = (
    REPOSITORY_ROOT / "monitoring/grafana/provisioning/dashboards/provider.yml"
)

IMAGE_RULES = {
    "TRITON_IMAGE": (
        "nvcr.io/nvidia/tritonserver",
        re.compile(r"\d{2}\.\d{2}-py3"),
        "a full monthly py3 tag",
    ),
    "PROMETHEUS_IMAGE": (
        "prom/prometheus",
        re.compile(r"v\d+\.\d+\.\d+"),
        "a full v-prefixed semantic version tag",
    ),
    "GRAFANA_IMAGE": (
        "grafana/grafana",
        re.compile(r"\d+\.\d+\.\d+"),
        "a full semantic version tag",
    ),
    "DCGM_EXPORTER_IMAGE": (
        "nvcr.io/nvidia/k8s/dcgm-exporter",
        re.compile(r"\d+\.\d+\.\d+-\d+\.\d+\.\d+-distroless"),
        "a full paired DCGM/exporter distroless tag",
    ),
}
REQUIRED_ENV_KEYS = {
    *IMAGE_RULES,
    "TRITON_HTTP_PORT",
    "TRITON_GRPC_PORT",
    "TRITON_METRICS_PORT",
    "PROMETHEUS_PORT",
    "GRAFANA_PORT",
    "DCGM_METRICS_PORT",
    "MODEL_REPOSITORY_PATH",
    "GRAFANA_ADMIN_USER",
    "GRAFANA_ADMIN_PASSWORD",
}
PORT_KEYS = {
    "TRITON_HTTP_PORT",
    "TRITON_GRPC_PORT",
    "TRITON_METRICS_PORT",
    "PROMETHEUS_PORT",
    "GRAFANA_PORT",
    "DCGM_METRICS_PORT",
}
EXPECTED_SERVICES = {"triton", "prometheus", "grafana", "dcgm-exporter"}
LIFECYCLE_SCRIPTS = (
    "deployment/scripts/run_environment.sh",
    "deployment/scripts/stop_environment.sh",
    "deployment/scripts/check_environment.sh",
    "deployment/scripts/run_triton.sh",
)
REQUIRED_DEPLOYMENT_PATHS = (
    "deployment/docker/Dockerfile",
    "deployment/scripts/compose_common.sh",
    *LIFECYCLE_SCRIPTS,
    "deployment/scripts/smoke_environment.py",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "monitoring/grafana/provisioning/dashboards/provider.yml",
    "monitoring/grafana/dashboards",
)
COMPOSE_VARIABLE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):\?[^}]+\}$")
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as yaml_file:
        value = yaml.safe_load(yaml_file)
    if not isinstance(value, dict):
        raise TypeError(f"{path.relative_to(REPOSITORY_ROOT).as_posix()} must be a YAML object")
    return value


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path.name}:{line_number} is not a KEY=VALUE assignment")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not key:
            raise ValueError(f"{path.name}:{line_number} has an empty key")
        if key in values:
            raise ValueError(f"{path.name}:{line_number} duplicates {key}")
        values[key] = value
    return values


def _compose_variable(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = COMPOSE_VARIABLE.fullmatch(value)
    return match.group(1) if match else None


def _resolve_compose_value(value: Any, env: dict[str, str]) -> str | None:
    variable = _compose_variable(value)
    if variable is not None:
        return env.get(variable)
    return value if isinstance(value, str) else None


def _mount_by_target(service: dict[str, Any], target: str) -> dict[str, Any] | None:
    for mount in service.get("volumes", []):
        if isinstance(mount, dict) and mount.get("target") == target:
            return mount
    return None


def _has_gpu_reservation(service: dict[str, Any]) -> bool:
    devices = (
        service.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices", [])
    )
    return any(
        isinstance(device, dict)
        and device.get("driver") == "nvidia"
        and device.get("count") == "all"
        and "gpu" in device.get("capabilities", [])
        for device in devices
    )


def _validate_env(env: dict[str, str], errors: list[str]) -> None:
    missing = sorted(key for key in REQUIRED_ENV_KEYS if not env.get(key))
    if missing:
        errors.append(f".env.example is missing required values: {', '.join(missing)}")

    port_values: list[int] = []
    for key in sorted(PORT_KEYS):
        value = env.get(key, "")
        try:
            port = int(value)
        except ValueError:
            errors.append(f"{key} must be an integer port, got {value!r}")
            continue
        if not 1 <= port <= 65535:
            errors.append(f"{key} is outside the valid port range: {port}")
        port_values.append(port)
    if len(port_values) != len(set(port_values)):
        errors.append("Published host ports in .env.example must be unique")

    model_path = env.get("MODEL_REPOSITORY_PATH")
    if model_path and not _is_repository_relative(model_path):
        errors.append("MODEL_REPOSITORY_PATH must be a repository-relative POSIX path")


def _is_repository_relative(value: str) -> bool:
    if WINDOWS_ABSOLUTE_PATH.search(value) or UNC_PATH.search(value):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    return ".." not in Path(normalized).parts


def _validate_images(
    services: dict[str, dict[str, Any]],
    env: dict[str, str],
    errors: list[str],
) -> None:
    triton_base = services.get("triton", {}).get("build", {}).get("args", {}).get(
        "TRITON_IMAGE"
    )
    image_references = {
        "TRITON_IMAGE": triton_base,
        "PROMETHEUS_IMAGE": services.get("prometheus", {}).get("image"),
        "GRAFANA_IMAGE": services.get("grafana", {}).get("image"),
        "DCGM_EXPORTER_IMAGE": services.get("dcgm-exporter", {}).get("image"),
    }
    for variable, expression in image_references.items():
        if _compose_variable(expression) != variable:
            errors.append(f"{variable} must be consumed through a required Compose variable")
            continue
        image = env.get(variable, "")
        repository, separator, tag = image.rpartition(":")
        expected_repository, tag_pattern, tag_description = IMAGE_RULES[variable]
        if not separator:
            errors.append(f"{variable} must contain an explicit image tag")
            continue
        if repository != expected_repository:
            errors.append(
                f"{variable} must use the canonical repository {expected_repository}"
            )
        if tag == "latest":
            errors.append(f"{variable} must not use latest")
        elif not tag_pattern.fullmatch(tag):
            errors.append(f"{variable} must use {tag_description}")


def _validate_mounts(
    compose: dict[str, Any],
    services: dict[str, dict[str, Any]],
    env: dict[str, str],
    errors: list[str],
) -> None:
    expected_mounts = {
        ("triton", "/models"): ("bind", "MODEL_REPOSITORY_PATH", True),
        ("prometheus", "/etc/prometheus/prometheus.yml"): (
            "bind",
            "./monitoring/prometheus/prometheus.yml",
            True,
        ),
        ("prometheus", "/prometheus"): ("volume", "prometheus-data", False),
        ("grafana", "/var/lib/grafana"): ("volume", "grafana-data", False),
        ("grafana", "/etc/grafana/provisioning"): (
            "bind",
            "./monitoring/grafana/provisioning",
            True,
        ),
        ("grafana", "/var/lib/grafana/dashboards"): (
            "bind",
            "./monitoring/grafana/dashboards",
            True,
        ),
    }
    for (service_name, target), (mount_type, expected_source, read_only) in (
        expected_mounts.items()
    ):
        mount = _mount_by_target(services.get(service_name, {}), target)
        if mount is None:
            errors.append(f"{service_name} is missing required mount target {target}")
            continue
        if mount.get("type") != mount_type:
            errors.append(f"{service_name}:{target} must use a {mount_type} mount")

        source = mount.get("source")
        if expected_source == "MODEL_REPOSITORY_PATH":
            if _compose_variable(source) != expected_source:
                errors.append("Triton model source must use required MODEL_REPOSITORY_PATH")
            resolved_source = env.get(expected_source, "")
        else:
            resolved_source = _resolve_compose_value(source, env) or ""
            if resolved_source != expected_source:
                errors.append(
                    f"{service_name}:{target} source is {resolved_source!r}, "
                    f"expected {expected_source!r}"
                )
        if mount_type == "bind" and not _is_repository_relative(resolved_source):
            errors.append(f"{service_name}:{target} bind source must be repository-relative")
        if bool(mount.get("read_only", False)) != read_only:
            errors.append(f"{service_name}:{target} read_only must be {read_only}")

    declared_volumes = set(compose.get("volumes", {}))
    if declared_volumes != {"prometheus-data", "grafana-data"}:
        errors.append("Compose must declare only prometheus-data and grafana-data volumes")


def _validate_ports(
    services: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    expected = {
        "triton": {
            "TRITON_HTTP_PORT": 8000,
            "TRITON_GRPC_PORT": 8001,
            "TRITON_METRICS_PORT": 8002,
        },
        "prometheus": {"PROMETHEUS_PORT": 9090},
        "grafana": {"GRAFANA_PORT": 3000},
        "dcgm-exporter": {"DCGM_METRICS_PORT": 9400},
    }
    for service_name, mappings in expected.items():
        ports = services.get(service_name, {}).get("ports", [])
        for variable, container_port in mappings.items():
            prefix = f"127.0.0.1:${{{variable}:?"
            suffix = f"}}:{container_port}"
            if not any(
                isinstance(port, str) and port.startswith(prefix) and port.endswith(suffix)
                for port in ports
            ):
                errors.append(
                    f"{service_name} must publish {container_port} through loopback-bound "
                    f"{variable}"
                )


def _validate_services(
    compose: dict[str, Any],
    env: dict[str, str],
    errors: list[str],
) -> None:
    raw_services = compose.get("services", {})
    services = {
        name: value for name, value in raw_services.items() if isinstance(value, dict)
    }
    missing = sorted(EXPECTED_SERVICES - set(services))
    if missing:
        errors.append(f"Compose is missing services: {', '.join(missing)}")
        return

    unexpected = sorted(set(services) - EXPECTED_SERVICES)
    if unexpected:
        errors.append(f"Step 2 Compose has out-of-scope services: {', '.join(unexpected)}")

    _validate_images(services, env, errors)
    _validate_mounts(compose, services, env, errors)
    _validate_ports(services, errors)

    for service_name in EXPECTED_SERVICES:
        service = services[service_name]
        if "healthcheck" not in service:
            errors.append(f"{service_name} must declare a healthcheck")
        if "depends_on" in service:
            errors.append(f"{service_name} must not declare a hard depends_on")
        if "backend" not in service.get("networks", []):
            errors.append(f"{service_name} must join the backend network")

    network = compose.get("networks", {}).get("backend", {})
    if network.get("driver") != "bridge":
        errors.append("backend must be a bridge network")
    if network.get("internal") is True:
        errors.append("backend must allow loopback-published host endpoints")

    for service_name in ("triton", "dcgm-exporter"):
        if not _has_gpu_reservation(services[service_name]):
            errors.append(f"{service_name} must reserve all NVIDIA GPUs")

    dcgm_health = services["dcgm-exporter"].get("healthcheck", {}).get("test", [])
    if (
        not isinstance(dcgm_health, list)
        or dcgm_health[:2] != ["CMD", "/usr/bin/dcgm-exporter"]
        or any(value in {"sh", "curl", "wget", "CMD-SHELL"} for value in dcgm_health)
    ):
        errors.append("DCGM distroless healthcheck must exec dcgm-exporter directly")

    triton_command = services["triton"].get("command", [])
    if "--model-control-mode=explicit" not in triton_command:
        errors.append("Triton must use explicit model control mode")
    if "--model-repository=/models" not in triton_command:
        errors.append("Triton model repository must be /models")


def _validate_prometheus(config: dict[str, Any], errors: list[str]) -> None:
    jobs = {
        item.get("job_name"): item
        for item in config.get("scrape_configs", [])
        if isinstance(item, dict)
    }
    expected_targets = {
        "prometheus": "prometheus:9090",
        "triton": "triton:8002",
        "dcgm-exporter": "dcgm-exporter:9400",
    }
    for job_name, expected_target in expected_targets.items():
        job = jobs.get(job_name)
        if job is None:
            errors.append(f"Prometheus is missing scrape job {job_name}")
            continue
        targets = {
            target
            for static_config in job.get("static_configs", [])
            for target in static_config.get("targets", [])
        }
        if expected_target not in targets:
            errors.append(f"Prometheus job {job_name} must target {expected_target}")


def _validate_grafana(
    datasource: dict[str, Any],
    provider: dict[str, Any],
    errors: list[str],
) -> None:
    datasources = datasource.get("datasources", [])
    expected_datasource = next(
        (
            item
            for item in datasources
            if isinstance(item, dict) and item.get("uid") == "prometheus"
        ),
        None,
    )
    if expected_datasource is None:
        errors.append("Grafana must provision a datasource with uid prometheus")
    elif (
        expected_datasource.get("url") != "http://prometheus:9090"
        or expected_datasource.get("type") != "prometheus"
        or expected_datasource.get("isDefault") is not True
    ):
        errors.append("Grafana Prometheus datasource has an invalid internal contract")

    providers = provider.get("providers", [])
    file_provider = next(
        (
            item
            for item in providers
            if isinstance(item, dict) and item.get("type") == "file"
        ),
        None,
    )
    dashboard_path = (
        file_provider.get("options", {}).get("path") if file_provider is not None else None
    )
    if dashboard_path != "/var/lib/grafana/dashboards":
        errors.append("Grafana dashboard provider must use /var/lib/grafana/dashboards")


def _validate_scripts(errors: list[str]) -> None:
    for relative_path in REQUIRED_DEPLOYMENT_PATHS:
        path = REPOSITORY_ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing deployment path: {relative_path}")

    for relative_path in LIFECYCLE_SCRIPTS:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if not content.startswith("#!/usr/bin/env bash\n"):
            errors.append(f"{relative_path} must use the Bash env shebang")
        if "set -euo pipefail" not in content:
            errors.append(f"{relative_path} must enable set -euo pipefail")
        if "compose_common.sh" not in content:
            errors.append(f"{relative_path} must use compose_common.sh")
        if re.search(r"\bsource\s+[\"']?[^ \n]*\.env", content):
            errors.append(f"{relative_path} must not source an environment file")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{relative_path} must use LF line endings")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content):
            errors.append(f"{relative_path} contains a host-specific absolute path")
        if "TSchool" in content:
            errors.append(f"{relative_path} contains a local workspace path")

    helper_path = REPOSITORY_ROOT / "deployment/scripts/compose_common.sh"
    if helper_path.is_file():
        helper = helper_path.read_text(encoding="utf-8")
        required_fragments = (
            "--project-directory",
            "--file",
            "--env-file",
            "/.env.example",
            "/.env",
        )
        for fragment in required_fragments:
            if fragment not in helper:
                errors.append(f"compose_common.sh is missing canonical fragment {fragment}")


def _validate_dockerfile(errors: list[str]) -> None:
    path = REPOSITORY_ROOT / "deployment/docker/Dockerfile"
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    required = ("ARG TRITON_IMAGE", "FROM ${TRITON_IMAGE}", 'ENTRYPOINT ["tritonserver"]')
    for fragment in required:
        if fragment not in content:
            errors.append(f"Dockerfile is missing: {fragment}")
    if "--model-" in content or "--http-port" in content or "--grpc-port" in content:
        errors.append("Dockerfile must not own Triton server arguments")


def _run_external_checks() -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    docker = shutil.which("docker")
    if docker is None:
        results.append(("SKIP", "Docker Compose config", "docker CLI is unavailable"))
    else:
        command = [
            docker,
            "compose",
            "--project-directory",
            str(REPOSITORY_ROOT),
            "--file",
            str(COMPOSE_PATH),
            "--env-file",
            str(ENV_EXAMPLE_PATH),
            "config",
            "--quiet",
        ]
        process = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = process.stderr.strip() or process.stdout.strip()
        status = "OK" if process.returncode == 0 else "FAIL"
        results.append((status, "Docker Compose config", detail))

    promtool = shutil.which("promtool")
    if promtool is None:
        results.append(("SKIP", "promtool config", "promtool is unavailable"))
    else:
        process = subprocess.run(
            [promtool, "check", "config", str(PROMETHEUS_PATH)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = process.stderr.strip() or process.stdout.strip()
        status = "OK" if process.returncode == 0 else "FAIL"
        results.append((status, "promtool config", detail))

    bash = shutil.which("bash")
    if bash is None:
        results.append(("SKIP", "Bash syntax", "bash is unavailable"))
    else:
        shell_scripts = (
            "deployment/scripts/compose_common.sh",
            *LIFECYCLE_SCRIPTS,
        )
        process = subprocess.run(
            [bash, "-n", *shell_scripts],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = process.stderr.strip() or process.stdout.strip()
        status = "OK" if process.returncode == 0 else "FAIL"
        results.append((status, "Bash syntax", detail))
    return results


def main() -> int:
    errors: list[str] = []
    try:
        env = _load_env(ENV_EXAMPLE_PATH)
        compose = _load_yaml(COMPOSE_PATH)
        prometheus = _load_yaml(PROMETHEUS_PATH)
        datasource = _load_yaml(DATASOURCE_PATH)
        provider = _load_yaml(DASHBOARD_PROVIDER_PATH)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"[ERROR] Cannot load deployment configuration: {error}", file=sys.stderr)
        return 1

    _validate_env(env, errors)
    _validate_services(compose, env, errors)
    _validate_prometheus(prometheus, errors)
    _validate_grafana(datasource, provider, errors)
    _validate_scripts(errors)
    _validate_dockerfile(errors)

    external_results = _run_external_checks()
    for status, name, detail in external_results:
        suffix = f": {detail}" if detail else ""
        stream = sys.stderr if status == "FAIL" else sys.stdout
        print(f"[{status}] {name}{suffix}", file=stream)
        if status == "FAIL":
            errors.append(f"{name} failed")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Deployment validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print("[OK] Deployment topology, pins, mounts, GPU policy, and provisioning are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
