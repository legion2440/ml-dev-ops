#!/usr/bin/env python3
"""Verify the Step 7 Triton -> Prometheus -> Grafana monitoring chain."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from client.inference_client import main as client_main
from client.transport import RepositoryController, TransportError
from scripts.validate_monitoring import (
    ALERT_CONTRACTS,
    DASHBOARD_PATH,
    DASHBOARD_UID,
    DATASOURCE_UID,
    GPU_METRIC,
    HASHED_ARTIFACTS,
    PANEL_CONTRACTS,
    QUERY_EVIDENCE_PATH,
    RUNTIME_EVIDENCE_PATH,
    TARGET_MODEL,
    TARGET_VERSION,
    _normalize_expression,
)

PROMETHEUS_PATH = REPOSITORY_ROOT / "monitoring/prometheus/prometheus.yml"
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
CLIENT_CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"
SAMPLE_DIRECTORY = REPOSITORY_ROOT / "client/samples"
CACHE_DIRECTORY = REPOSITORY_ROOT / ".cache/monitoring"
TEMPORARY_LOG_PATH = CACHE_DIRECTORY / "inference-log.jsonl"
GRAFANA_PROXY_PATH = "/api/datasources/proxy/uid/prometheus/api/v1/query"


class MonitoringVerificationError(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MonitoringVerificationError(f"{path.name} must contain a JSON object")
    return value


def _select_env_file(argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
    else:
        local = REPOSITORY_ROOT / ".env"
        path = local if local.is_file() else REPOSITORY_ROOT / ".env.example"
    if not path.is_file():
        raise MonitoringVerificationError(f"Compose environment file is missing: {path}")
    return path.resolve()


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise MonitoringVerificationError(
                f"{path.name}:{line_number} is not a KEY=VALUE assignment"
            )
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _required(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    if not value:
        raise MonitoringVerificationError(f"Required environment value is missing: {key}")
    return value


def _base_url(port: str) -> str:
    return f"http://127.0.0.1:{port}"


def _published_port(env_file: Path, service: str, container_port: int) -> str:
    process = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(REPOSITORY_ROOT),
            "--file",
            str(COMPOSE_PATH),
            "--env-file",
            str(env_file),
            "port",
            service,
            str(container_port),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise MonitoringVerificationError(
            f"Cannot resolve published port for {service}:{container_port}: {detail}"
        )
    endpoint = process.stdout.strip().splitlines()
    if len(endpoint) != 1 or ":" not in endpoint[0]:
        raise MonitoringVerificationError(
            f"Unexpected published port for {service}:{container_port}: {process.stdout!r}"
        )
    return endpoint[0].rsplit(":", 1)[1]


def _basic_auth(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {**(headers or {})}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise MonitoringVerificationError(
            f"HTTP {error.code} for {url}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise MonitoringVerificationError(f"Cannot reach {url}: {error.reason}") from error


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = json.loads(
        _request(url, headers=headers, method=method, payload=payload).decode("utf-8")
    )
    if not isinstance(value, dict):
        raise MonitoringVerificationError(f"{url} did not return a JSON object")
    return value


def _eventually(
    description: str,
    operation: Callable[[], Any],
    predicate: Callable[[Any], bool],
    *,
    timeout: float = 60.0,
    interval: float = 2.0,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        try:
            value = operation()
            if predicate(value):
                return value
        except (OSError, ValueError, KeyError, MonitoringVerificationError) as error:
            last_error = error
        if time.monotonic() >= deadline:
            suffix = f": {last_error}" if last_error else ""
            raise MonitoringVerificationError(f"Timed out waiting for {description}{suffix}")
        time.sleep(interval)


def _scrape_interval_seconds() -> float:
    value = yaml.safe_load(PROMETHEUS_PATH.read_text(encoding="utf-8"))
    interval = value.get("global", {}).get("scrape_interval") if isinstance(value, dict) else None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", str(interval))
    if match is None:
        raise MonitoringVerificationError(
            "Step 7 verifier requires a Prometheus scrape_interval expressed in seconds"
        )
    return float(match.group(1))


def _ready_rows(values: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"model": model, "version": version}
        for model, version in sorted(values, key=lambda item: (item[0], int(item[1])))
    ]


def _validate_restorable_ready_set(initial: set[tuple[str, str]]) -> None:
    contract = _load_json(CLIENT_CONTRACT_PATH)
    for model, entry in contract.get("models", {}).items():
        expected = {str(version) for version in entry.get("versions", [])}
        ready = {version for name, version in initial if name == model}
        if ready and ready != expected:
            raise MonitoringVerificationError(
                f"Initial READY state for {model} is partial and cannot be restored exactly"
            )


def _restore_ready_set(
    controller: RepositoryController, initial: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    initial_models = {model for model, _ in initial}
    current_models = {model for model, _ in controller.ready_set()}
    for model in sorted(current_models - initial_models):
        controller.unload(model)
    for model in sorted(initial_models - current_models):
        controller.load(model)

    deadline = time.monotonic() + 120.0
    while True:
        current = controller.ready_set()
        if current == initial:
            return current
        if time.monotonic() >= deadline:
            raise MonitoringVerificationError(
                f"READY state was not restored: initial={sorted(initial)}, final={sorted(current)}"
            )
        time.sleep(0.5)


def _run_workload(http_port: str, duration_seconds: float) -> dict[str, Any]:
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEMPORARY_LOG_PATH.unlink(missing_ok=True)
    arguments = [
        "classify",
        SAMPLE_DIRECTORY.relative_to(REPOSITORY_ROOT).as_posix(),
        "--model",
        TARGET_MODEL,
        "--version",
        TARGET_VERSION,
        "--batch-size",
        "8",
        "--protocol",
        "http",
        "--http-url",
        f"127.0.0.1:{http_port}",
        "--log-file",
        TEMPORARY_LOG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
    ]
    started = time.monotonic()
    invocations = 0
    while time.monotonic() - started < duration_seconds:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return_code = client_main(arguments)
        if return_code != 0:
            detail = stderr.getvalue().strip() or stdout.getvalue().strip()
            raise MonitoringVerificationError(f"Inference workload failed: {detail}")
        invocations += 1
        time.sleep(0.25)
    elapsed = time.monotonic() - started
    requests = sum(1 for line in TEMPORARY_LOG_PATH.read_text(encoding="utf-8").splitlines() if line)
    return {
        "client": "client/inference_client.py classify",
        "duration_seconds": round(elapsed, 3),
        "minimum_duration_seconds": duration_seconds,
        "invocations": invocations,
        "inference_requests": requests,
        "model": TARGET_MODEL,
        "version": TARGET_VERSION,
        "temporary_log": ".cache/monitoring/inference-log.jsonl",
    }


def _panel_queries_from_dashboard(
    dashboard: dict[str, Any],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for panel in dashboard.get("panels", []):
        if not isinstance(panel, dict):
            continue
        title = panel.get("title")
        targets = panel.get("targets", [])
        contract = PANEL_CONTRACTS.get(title)
        if contract and len(targets) == 1 and isinstance(targets[0], dict):
            result[contract["id"]] = {
                "panel_title": title,
                "expression": str(targets[0].get("expr", "")),
            }
    if set(result) != {contract["id"] for contract in PANEL_CONTRACTS.values()}:
        raise MonitoringVerificationError("Cannot extract all five queries from dashboard JSON")
    return result


def _panel_queries() -> dict[str, dict[str, str]]:
    return _panel_queries_from_dashboard(_load_json(DASHBOARD_PATH))


def _grafana_query(
    grafana_url: str, headers: dict[str, str], expression: str
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"query": expression})
    return _request_json(
        f"{grafana_url}{GRAFANA_PROXY_PATH}?{query}", headers=headers
    )


def _query_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "success":
        raise MonitoringVerificationError("Grafana datasource proxy query did not succeed")
    data = payload.get("data", {})
    if data.get("resultType") != "vector":
        raise MonitoringVerificationError("Grafana datasource proxy did not return a vector")
    samples: list[dict[str, Any]] = []
    for item in data.get("result", []):
        if not isinstance(item, dict):
            continue
        raw_value = item.get("value", [])
        if not isinstance(raw_value, list) or len(raw_value) != 2:
            continue
        value = float(raw_value[1])
        if not math.isfinite(value):
            continue
        metric = item.get("metric", {})
        samples.append(
            {
                "metric": dict(sorted(metric.items())) if isinstance(metric, dict) else {},
                "timestamp": float(raw_value[0]),
                "value": value,
            }
        )
    return samples


def _required_query_data_available(values: dict[str, list[dict[str, Any]]]) -> bool:
    for query_id in ("inference_throughput", "request_rate", "average_request_latency"):
        matching = [
            sample["value"]
            for sample in values.get(query_id, [])
            if sample["metric"].get("model") == TARGET_MODEL
            and sample["metric"].get("version") == TARGET_VERSION
        ]
        if not matching or max(matching) <= 0:
            return False
    return bool(values.get("gpu_utilization")) and bool(values.get("failed_requests"))


def _capture_queries(
    grafana_url: str, headers: dict[str, str], timeout: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definitions = _panel_queries()

    def capture() -> dict[str, list[dict[str, Any]]]:
        return {
            query_id: _query_samples(
                _grafana_query(grafana_url, headers, definition["expression"])
            )
            for query_id, definition in definitions.items()
        }

    values = _eventually(
        "non-empty panel data through the Grafana datasource proxy",
        capture,
        _required_query_data_available,
        timeout=timeout,
        interval=2.0,
    )
    queries = [
        {
            "id": query_id,
            "panel_title": definitions[query_id]["panel_title"],
            "expression": definitions[query_id]["expression"],
            "result_type": "vector",
            "samples": values[query_id],
        }
        for query_id in sorted(definitions)
    ]
    maximum_expression = (
        "max by (UUID, gpu, modelName, pci_bus_id) "
        f"(max_over_time({GPU_METRIC}[2m]))"
    )
    maximum_samples = _query_samples(
        _grafana_query(grafana_url, headers, maximum_expression)
    )
    diagnostic = {
        "expression": maximum_expression,
        "samples": maximum_samples,
        "positive_observed": any(sample["value"] > 0 for sample in maximum_samples),
        "acceptance_gate": False,
    }
    return queries, diagnostic


def _prometheus_targets(prometheus_url: str) -> list[dict[str, Any]]:
    payload = _request_json(f"{prometheus_url}/api/v1/targets")
    rows: list[dict[str, Any]] = []
    for target in payload.get("data", {}).get("activeTargets", []):
        if not isinstance(target, dict):
            continue
        job = target.get("labels", {}).get("job")
        if job in {"triton", "dcgm-exporter"}:
            rows.append(
                {
                    "job": job,
                    "health": target.get("health"),
                    "scrape_url": target.get("scrapeUrl"),
                }
            )
    if {item["job"] for item in rows if item.get("health") == "up"} != {
        "triton",
        "dcgm-exporter",
    }:
        raise MonitoringVerificationError("Triton and DCGM Prometheus targets must be up")
    return sorted(rows, key=lambda item: item["job"])


def _prometheus_alerts(prometheus_url: str) -> list[dict[str, str]]:
    payload = _request_json(f"{prometheus_url}/api/v1/rules?type=alert")
    rows: list[dict[str, str]] = []
    for group in payload.get("data", {}).get("groups", []):
        for rule in group.get("rules", []) if isinstance(group, dict) else []:
            if isinstance(rule, dict) and rule.get("name") in ALERT_CONTRACTS:
                rows.append(
                    {
                        "name": str(rule["name"]),
                        "state": str(rule.get("state", "")),
                        "expression": str(rule.get("query", "")),
                    }
                )
    by_name = {item["name"]: item for item in rows}
    if set(by_name) != set(ALERT_CONTRACTS):
        raise MonitoringVerificationError("Prometheus did not load both Step 7 alert rules")
    for name, contract in ALERT_CONTRACTS.items():
        if _normalize_expression(by_name[name]["expression"]) != _normalize_expression(
            contract["expression"]
        ):
            raise MonitoringVerificationError(
                f"Loaded Prometheus expression differs for {name}"
            )
    return [by_name[name] for name in sorted(by_name)]


def _nvidia_identity() -> dict[str, str]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise MonitoringVerificationError(
            f"nvidia-smi identity query failed: {process.stderr.strip()}"
        )
    rows = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise MonitoringVerificationError("Step 7 requires exactly one selected NVIDIA GPU")
    values = [item.strip() for item in rows[0].split(",", 2)]
    if len(values) != 3:
        raise MonitoringVerificationError("Cannot parse nvidia-smi GPU identity")
    return {"uuid": values[0], "model_name": values[1], "pci_bus_id": values[2]}


def _matching_gpu_sample(
    queries: list[dict[str, Any]], identity: dict[str, str]
) -> dict[str, Any]:
    gpu_query = next(item for item in queries if item["id"] == "gpu_utilization")
    for sample in gpu_query["samples"]:
        metric = sample["metric"]
        if (
            metric.get("UUID") == identity["uuid"]
            and metric.get("modelName") == identity["model_name"]
            and metric.get("pci_bus_id", "").lower() == identity["pci_bus_id"].lower()
        ):
            return sample
    raise MonitoringVerificationError(
        "DCGM GPU series does not match the nvidia-smi GPU identity"
    )


def run(env_file: Path) -> dict[str, Any]:
    env = _load_env(env_file)
    triton_http_port = _published_port(env_file, "triton", 8000)
    prometheus_url = _base_url(_published_port(env_file, "prometheus", 9090))
    grafana_url = _base_url(_published_port(env_file, "grafana", 3000))
    grafana_headers = _basic_auth(
        _required(env, "GRAFANA_ADMIN_USER"),
        _required(env, "GRAFANA_ADMIN_PASSWORD"),
    )
    scrape_interval = _scrape_interval_seconds()
    workload_duration = 2 * scrape_interval + 5.0

    if _request(f"{prometheus_url}/-/healthy").decode("utf-8").strip() != "Prometheus Server is Healthy.":
        raise MonitoringVerificationError("Prometheus health endpoint is not healthy")
    grafana_health = _request_json(f"{grafana_url}/api/health")
    if grafana_health.get("database") != "ok":
        raise MonitoringVerificationError("Grafana health endpoint is not healthy")
    datasource = _request_json(
        f"{grafana_url}/api/datasources/uid/{DATASOURCE_UID}", headers=grafana_headers
    )
    datasource_health = _request_json(
        f"{grafana_url}/api/datasources/uid/{DATASOURCE_UID}/health",
        headers=grafana_headers,
    )
    if datasource_health.get("status") != "OK":
        raise MonitoringVerificationError("Grafana Prometheus datasource is not healthy")
    dashboard_response = _eventually(
        "provisioned Grafana dashboard",
        lambda: _request_json(
            f"{grafana_url}/api/dashboards/uid/{DASHBOARD_UID}",
            headers=grafana_headers,
        ),
        lambda value: value.get("dashboard", {}).get("uid") == DASHBOARD_UID,
        timeout=75.0,
    )
    dashboard = dashboard_response["dashboard"]
    panel_titles = sorted(
        panel.get("title")
        for panel in dashboard.get("panels", [])
        if isinstance(panel, dict) and isinstance(panel.get("title"), str)
    )
    if set(panel_titles) != set(PANEL_CONTRACTS):
        raise MonitoringVerificationError("Provisioned Grafana dashboard has wrong panels")
    repository_queries = _panel_queries()
    provisioned_queries = _panel_queries_from_dashboard(dashboard)
    if {
        query_id: _normalize_expression(item["expression"])
        for query_id, item in provisioned_queries.items()
    } != {
        query_id: _normalize_expression(item["expression"])
        for query_id, item in repository_queries.items()
    }:
        raise MonitoringVerificationError(
            "Provisioned Grafana dashboard queries differ from repository JSON"
        )

    targets = _prometheus_targets(prometheus_url)
    alerts = _prometheus_alerts(prometheus_url)
    identity = _nvidia_identity()
    controller = RepositoryController(f"127.0.0.1:{triton_http_port}", 120.0)
    initial = controller.ready_set()
    _validate_restorable_ready_set(initial)

    workload: dict[str, Any] | None = None
    queries: list[dict[str, Any]] | None = None
    maximum_observation: dict[str, Any] | None = None
    try:
        workload = _run_workload(triton_http_port, workload_duration)
        queries, maximum_observation = _capture_queries(
            grafana_url, grafana_headers, timeout=60.0
        )
        gpu_sample = _matching_gpu_sample(queries, identity)
    finally:
        final = _restore_ready_set(controller, initial)

    if workload is None or queries is None or maximum_observation is None:
        raise MonitoringVerificationError("Monitoring verification did not complete")

    generated_at = _timestamp()
    query_evidence = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "passed": True,
        "access": {
            "kind": "grafana_datasource_proxy",
            "datasource_uid": DATASOURCE_UID,
            "path": GRAFANA_PROXY_PATH,
        },
        "queries": queries,
        "gpu_max_over_time_observation": maximum_observation,
    }
    _write_json(QUERY_EVIDENCE_PATH, query_evidence)

    artifact_sha256 = {
        relative: _sha256(REPOSITORY_ROOT / relative) for relative in HASHED_ARTIFACTS
    }
    runtime_evidence = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "passed": True,
        "runtime": {
            "env_file": env_file.name,
            "scrape_interval_seconds": scrape_interval,
            "minimum_scrape_intervals_observed": 2,
            "workload": workload,
        },
        "prometheus": {
            "health": "healthy",
            "targets": targets,
            "alerts": alerts,
        },
        "grafana": {
            "health": "ok",
            "version": grafana_health.get("version"),
            "datasource": {
                "uid": datasource.get("uid"),
                "url": datasource.get("url"),
                "health": datasource_health.get("status"),
            },
            "dashboard": {
                "uid": dashboard.get("uid"),
                "title": dashboard.get("title"),
                "panels": panel_titles,
                "queries_match_repository": True,
            },
        },
        "gpu": {
            "metric": GPU_METRIC,
            "unit": "percent_0_to_100",
            "identity": identity,
            "observed_value": gpu_sample["value"],
            "numeric_series_present": True,
            "positive_max_over_time_observed": maximum_observation["positive_observed"],
            "positive_max_over_time_is_acceptance_gate": False,
        },
        "ready_state": {
            "initial": _ready_rows(initial),
            "final": _ready_rows(final),
            "restored": final == initial,
        },
        "query_evidence": {
            "path": "docs/evidence/step-7/prometheus-queries.json",
            "sha256": _sha256(QUERY_EVIDENCE_PATH),
        },
        "artifact_sha256": artifact_sha256,
    }
    _write_json(RUNTIME_EVIDENCE_PATH, runtime_evidence)
    return runtime_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Compose environment file.")
    args = parser.parse_args()
    try:
        evidence = run(_select_env_file(args.env_file))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        yaml.YAMLError,
        MonitoringVerificationError,
        TransportError,
    ) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    workload = evidence["runtime"]["workload"]
    print(
        "[OK] Step 7 monitoring runtime passed: "
        f"{workload['inference_requests']} requests across at least two scrapes, "
        "Grafana datasource queries returned data, alerts loaded, READY state restored."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
