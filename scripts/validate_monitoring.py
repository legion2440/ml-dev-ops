"""Validate Step 7 monitoring configuration and committed runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_PATH = REPOSITORY_ROOT / "monitoring/prometheus/prometheus.yml"
ALERTS_PATH = REPOSITORY_ROOT / "monitoring/prometheus/alerts.yml"
DASHBOARD_PATH = REPOSITORY_ROOT / "monitoring/grafana/dashboards/ml-dev-ops.json"
DATASOURCE_PATH = (
    REPOSITORY_ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml"
)
PROVIDER_PATH = (
    REPOSITORY_ROOT / "monitoring/grafana/provisioning/dashboards/provider.yml"
)
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
RUNTIME_EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-7/monitoring-runtime.json"
QUERY_EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-7/prometheus-queries.json"

DASHBOARD_UID = "ml-dev-ops-inference"
DATASOURCE_UID = "prometheus"
GPU_METRIC = "DCGM_FI_DEV_GPU_UTIL"
TARGET_MODEL = "resnet50_onnx"
TARGET_VERSION = "1"

PANEL_CONTRACTS = {
    "Inference Throughput": {
        "id": "inference_throughput",
        "expression": "sum by (model, version) (rate(nv_inference_count[1m]))",
        "unit": "ops",
        "positive": True,
    },
    "Request Rate": {
        "id": "request_rate",
        "expression": (
            "sum by (model, version) "
            "(rate(nv_inference_request_success[1m]))"
        ),
        "unit": "reqps",
        "positive": True,
    },
    "Average Request Latency": {
        "id": "average_request_latency",
        "expression": (
            "sum by (model, version) "
            "(rate(nv_inference_request_duration_us[1m])) / "
            "sum by (model, version) "
            "(rate(nv_inference_request_success[1m])) / 1000"
        ),
        "unit": "ms",
        "positive": True,
    },
    "GPU Utilization": {
        "id": "gpu_utilization",
        "expression": (
            "max by (UUID, gpu, modelName, pci_bus_id) "
            "(DCGM_FI_DEV_GPU_UTIL)"
        ),
        "unit": "percent",
        "positive": False,
    },
    "Failed Requests": {
        "id": "failed_requests",
        "expression": (
            "sum by (model, version) "
            "(rate(nv_inference_request_failure[1m]))"
        ),
        "unit": "reqps",
        "positive": False,
    },
}

ALERT_CONTRACTS = {
    "HighInferenceLatency": {
        "expression": """
            (
              sum by (model, version) (
                rate(nv_inference_request_duration_us[1m])
              )
              /
              sum by (model, version) (
                rate(nv_inference_request_success[1m])
              )
              / 1000
            ) > 100
            and on (model, version)
            (
              sum by (model, version) (
                rate(nv_inference_request_success[1m])
              ) > 0
            )
        """,
        "for": "2m",
    },
    "InferenceRequestFailures": {
        "expression": """
            sum by (model, version) (
              increase(nv_inference_request_failure[5m])
            ) > 0
        """,
        "for": None,
    },
}

HASHED_ARTIFACTS = (
    "docker-compose.yml",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/prometheus/alerts.yml",
    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "monitoring/grafana/provisioning/dashboards/provider.yml",
    "monitoring/grafana/dashboards/ml-dev-ops.json",
    "monitoring/verify_runtime.py",
    "scripts/validate_monitoring.py",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a YAML object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_expression(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"\s+", " ", value).strip()
    normalized = re.sub(r"\(\s+", "(", normalized)
    return re.sub(r"\s+\)", ")", normalized)


def _panel_map(dashboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        panel.get("title"): panel
        for panel in dashboard.get("panels", [])
        if isinstance(panel, dict) and isinstance(panel.get("title"), str)
    }


def _target_expression(panel: dict[str, Any]) -> str:
    targets = panel.get("targets", [])
    if len(targets) != 1 or not isinstance(targets[0], dict):
        return ""
    return _normalize_expression(targets[0].get("expr"))


def _validate_prometheus(
    prometheus: dict[str, Any], compose: dict[str, Any], errors: list[str]
) -> None:
    if prometheus.get("global", {}).get("scrape_interval") != "15s":
        errors.append("Prometheus scrape interval must remain 15s for Step 7")
    rule_files = prometheus.get("rule_files", [])
    if rule_files != ["/etc/prometheus/alerts.yml"]:
        errors.append("Prometheus must load only /etc/prometheus/alerts.yml")

    jobs = {
        item.get("job_name"): item
        for item in prometheus.get("scrape_configs", [])
        if isinstance(item, dict)
    }
    for job, target in {
        "triton": "triton:8002",
        "dcgm-exporter": "dcgm-exporter:9400",
    }.items():
        values = {
            candidate
            for entry in jobs.get(job, {}).get("static_configs", [])
            if isinstance(entry, dict)
            for candidate in entry.get("targets", [])
        }
        if target not in values:
            errors.append(f"Prometheus job {job} must scrape {target}")

    mounts = compose.get("services", {}).get("prometheus", {}).get("volumes", [])
    alerts_mount = next(
        (
            item
            for item in mounts
            if isinstance(item, dict)
            and item.get("target") == "/etc/prometheus/alerts.yml"
        ),
        None,
    )
    if alerts_mount is None:
        errors.append("Compose must mount the Prometheus alert rule file")
    elif (
        alerts_mount.get("source") != "./monitoring/prometheus/alerts.yml"
        or alerts_mount.get("type") != "bind"
        or alerts_mount.get("read_only") is not True
    ):
        errors.append("Prometheus alert rule mount has an invalid contract")


def _validate_alerts(alerts: dict[str, Any], errors: list[str]) -> None:
    groups = alerts.get("groups", [])
    if len(groups) != 1 or not isinstance(groups[0], dict):
        errors.append("Prometheus alerts must contain exactly one rule group")
        return
    if groups[0].get("name") != "ml-dev-ops-inference":
        errors.append("Prometheus alert group must be ml-dev-ops-inference")
    rules = {
        rule.get("alert"): rule
        for rule in groups[0].get("rules", [])
        if isinstance(rule, dict) and isinstance(rule.get("alert"), str)
    }
    if set(rules) != set(ALERT_CONTRACTS):
        errors.append("Prometheus must define exactly the two Step 7 alert rules")
        return
    for name, contract in ALERT_CONTRACTS.items():
        rule = rules[name]
        if _normalize_expression(rule.get("expr")) != _normalize_expression(
            contract["expression"]
        ):
            errors.append(f"Alert {name} has an unexpected expression")
        if rule.get("for") != contract["for"]:
            errors.append(f"Alert {name} has an unexpected for duration")
        if rule.get("labels", {}).get("severity") != "warning":
            errors.append(f"Alert {name} must use warning severity")


def _validate_grafana(
    dashboard: dict[str, Any],
    datasource: dict[str, Any],
    provider: dict[str, Any],
    errors: list[str],
) -> None:
    if dashboard.get("uid") != DASHBOARD_UID:
        errors.append(f"Grafana dashboard UID must be {DASHBOARD_UID}")
    if dashboard.get("title") != "ML DevOps Inference":
        errors.append("Grafana dashboard title is invalid")
    if dashboard.get("refresh") != "15s":
        errors.append("Grafana dashboard refresh must match the 15s scrape interval")

    panels = _panel_map(dashboard)
    if set(panels) != set(PANEL_CONTRACTS):
        errors.append("Grafana dashboard must contain exactly the five Step 7 panels")
    ids = [panel.get("id") for panel in panels.values()]
    if len(ids) != len(set(ids)) or any(not isinstance(value, int) for value in ids):
        errors.append("Grafana panel IDs must be unique integers")

    for title, contract in PANEL_CONTRACTS.items():
        panel = panels.get(title)
        if panel is None:
            continue
        if panel.get("datasource", {}).get("uid") != DATASOURCE_UID:
            errors.append(f"Grafana panel {title} must use datasource UID prometheus")
        targets = panel.get("targets", [])
        if len(targets) != 1 or not isinstance(targets[0], dict):
            errors.append(f"Grafana panel {title} must contain exactly one query")
        elif targets[0].get("datasource", {}).get("uid") != DATASOURCE_UID:
            errors.append(f"Grafana query {title} must use datasource UID prometheus")
        if _target_expression(panel) != _normalize_expression(contract["expression"]):
            errors.append(f"Grafana panel {title} has an unexpected PromQL expression")
        if panel.get("fieldConfig", {}).get("defaults", {}).get("unit") != contract["unit"]:
            errors.append(f"Grafana panel {title} has an invalid unit")

    all_expressions = " ".join(_target_expression(panel) for panel in panels.values())
    if "clamp_min" in all_expressions:
        errors.append("Grafana PromQL must not use clamp_min for request latency")
    if "DCGM_FI_DEV_GPU_UTIL" in all_expressions:
        gpu_panel = panels.get("GPU Utilization", {})
        gpu_expression = _target_expression(gpu_panel)
        if "/ 100" in gpu_expression or "* 100" in gpu_expression:
            errors.append("DCGM GPU utilization is already expressed on a 0..100 scale")

    datasources = datasource.get("datasources", [])
    selected = next(
        (
            item
            for item in datasources
            if isinstance(item, dict) and item.get("uid") == DATASOURCE_UID
        ),
        None,
    )
    if selected is None or selected.get("url") != "http://prometheus:9090":
        errors.append("Grafana Prometheus datasource contract is invalid")

    providers = provider.get("providers", [])
    selected_provider = next(
        (
            item
            for item in providers
            if isinstance(item, dict) and item.get("name") == "ml-dev-ops"
        ),
        None,
    )
    if (
        selected_provider is None
        or selected_provider.get("options", {}).get("path")
        != "/var/lib/grafana/dashboards"
    ):
        errors.append("Existing Grafana dashboard provider contract is invalid")


def validate_config() -> list[str]:
    errors: list[str] = []
    try:
        prometheus = _load_yaml(PROMETHEUS_PATH)
        alerts = _load_yaml(ALERTS_PATH)
        compose = _load_yaml(COMPOSE_PATH)
        dashboard = _load_json(DASHBOARD_PATH)
        datasource = _load_yaml(DATASOURCE_PATH)
        provider = _load_yaml(PROVIDER_PATH)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as error:
        return [f"Cannot load monitoring configuration: {error}"]

    _validate_prometheus(prometheus, compose, errors)
    _validate_alerts(alerts, errors)
    _validate_grafana(dashboard, datasource, provider, errors)
    return errors


def _sample_values(query: dict[str, Any], errors: list[str], query_id: str) -> list[float]:
    samples = query.get("samples", [])
    if not isinstance(samples, list) or not samples:
        errors.append(f"Runtime query {query_id} has no samples")
        return []
    values: list[float] = []
    for sample in samples:
        try:
            value = float(sample.get("value"))
        except (AttributeError, TypeError, ValueError):
            errors.append(f"Runtime query {query_id} contains a non-numeric sample")
            continue
        if not math.isfinite(value):
            errors.append(f"Runtime query {query_id} contains a non-finite sample")
            continue
        values.append(value)
    return values


def validate_evidence() -> list[str]:
    errors: list[str] = []
    try:
        runtime = _load_json(RUNTIME_EVIDENCE_PATH)
        query_evidence = _load_json(QUERY_EVIDENCE_PATH)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        return [f"Cannot load Step 7 evidence: {error}"]

    if runtime.get("schema_version") != 1 or runtime.get("passed") is not True:
        errors.append("Step 7 runtime evidence is not a schema-v1 PASS")
    runtime_contract = runtime.get("runtime", {})
    scrape_interval = runtime_contract.get("scrape_interval_seconds")
    workload = runtime_contract.get("workload", {})
    try:
        observed_duration = float(workload.get("duration_seconds"))
        required_duration = 2 * float(scrape_interval)
    except (TypeError, ValueError):
        errors.append("Step 7 runtime evidence has invalid scrape/workload duration")
    else:
        if observed_duration < required_duration:
            errors.append("Step 7 workload did not span at least two scrape intervals")
    if (
        runtime_contract.get("minimum_scrape_intervals_observed") != 2
        or workload.get("client") != "client/inference_client.py classify"
        or workload.get("temporary_log") != ".cache/monitoring/inference-log.jsonl"
    ):
        errors.append("Step 7 workload contract is stale")
    ready = runtime.get("ready_state", {})
    if ready.get("restored") is not True or ready.get("initial") != ready.get("final"):
        errors.append("Step 7 runtime evidence does not prove READY state restoration")

    targets = {
        item.get("job"): item.get("health")
        for item in runtime.get("prometheus", {}).get("targets", [])
        if isinstance(item, dict)
    }
    if targets.get("triton") != "up" or targets.get("dcgm-exporter") != "up":
        errors.append("Step 7 runtime evidence does not prove both scrape targets are up")

    grafana = runtime.get("grafana", {})
    if grafana.get("health") != "ok":
        errors.append("Step 7 runtime evidence does not prove Grafana health")
    datasource = grafana.get("datasource", {})
    if datasource.get("uid") != DATASOURCE_UID or datasource.get("health") != "OK":
        errors.append("Step 7 runtime evidence does not prove datasource health")
    dashboard = grafana.get("dashboard", {})
    if dashboard.get("uid") != DASHBOARD_UID or set(dashboard.get("panels", [])) != set(
        PANEL_CONTRACTS
    ) or dashboard.get("queries_match_repository") is not True:
        errors.append("Step 7 runtime evidence does not prove the provisioned dashboard")

    runtime_alerts = {
        item.get("name"): item
        for item in runtime.get("prometheus", {}).get("alerts", [])
        if isinstance(item, dict)
    }
    if set(runtime_alerts) != set(ALERT_CONTRACTS):
        errors.append("Step 7 runtime evidence does not prove both alert rules are loaded")
    for name, contract in ALERT_CONTRACTS.items():
        item = runtime_alerts.get(name, {})
        if item.get("state") not in {"inactive", "pending", "firing"}:
            errors.append(f"Runtime alert {name} has an invalid state")
        if _normalize_expression(item.get("expression")) != _normalize_expression(
            contract["expression"]
        ):
            errors.append(f"Runtime alert {name} expression differs from the repo rule")

    query_reference = runtime.get("query_evidence", {})
    if query_reference.get("path") != "docs/evidence/step-7/prometheus-queries.json" or query_reference.get(
        "sha256"
    ) != _sha256(QUERY_EVIDENCE_PATH):
        errors.append("Step 7 runtime query evidence reference is stale")

    artifact_hashes = runtime.get("artifact_sha256", {})
    for relative in HASHED_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        if artifact_hashes.get(relative) != _sha256(path):
            errors.append(f"Step 7 artifact hash is stale: {relative}")

    access = query_evidence.get("access", {})
    if (
        access.get("kind") != "grafana_datasource_proxy"
        or access.get("datasource_uid") != DATASOURCE_UID
    ):
        errors.append("Runtime queries were not captured through the Grafana datasource proxy")
    queries = {
        item.get("id"): item
        for item in query_evidence.get("queries", [])
        if isinstance(item, dict)
    }
    if set(queries) != {contract["id"] for contract in PANEL_CONTRACTS.values()}:
        errors.append("Step 7 query evidence does not contain exactly the five panel queries")

    for title, contract in PANEL_CONTRACTS.items():
        query_id = contract["id"]
        query = queries.get(query_id, {})
        if query.get("panel_title") != title:
            errors.append(f"Runtime query {query_id} has an invalid panel title")
        if _normalize_expression(query.get("expression")) != _normalize_expression(
            contract["expression"]
        ):
            errors.append(f"Runtime query {query_id} differs from the dashboard")
        values = _sample_values(query, errors, query_id)
        if contract["positive"]:
            matching = [
                float(sample["value"])
                for sample in query.get("samples", [])
                if sample.get("metric", {}).get("model") == TARGET_MODEL
                and sample.get("metric", {}).get("version") == TARGET_VERSION
            ]
            if not matching or max(matching) <= 0:
                errors.append(f"Runtime query {query_id} has no positive target-model sample")
        elif query_id == "gpu_utilization" and not values:
            errors.append("GPU utilization must contain a numeric sample; zero is valid")

    gpu = runtime.get("gpu", {})
    identity = gpu.get("identity", {})
    if gpu.get("metric") != GPU_METRIC or not all(
        identity.get(field) for field in ("uuid", "model_name", "pci_bus_id")
    ):
        errors.append("Step 7 runtime evidence has incomplete GPU identity")
    gpu_query = queries.get("gpu_utilization", {})
    if not any(
        sample.get("metric", {}).get("UUID") == identity.get("uuid")
        and sample.get("metric", {}).get("modelName") == identity.get("model_name")
        and sample.get("metric", {}).get("pci_bus_id", "").lower()
        == str(identity.get("pci_bus_id", "")).lower()
        for sample in gpu_query.get("samples", [])
    ):
        errors.append("GPU query samples do not match the verified GPU identity")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate monitoring configuration without requiring runtime evidence.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the tracked generated evidence (default behavior).",
    )
    args = parser.parse_args()

    errors = validate_config()
    if not args.config_only:
        errors.extend(validate_evidence())
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Monitoring validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    if args.config_only:
        print("[OK] Step 7 monitoring configuration is valid.")
    else:
        print("[OK] Step 7 monitoring configuration and runtime evidence are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
