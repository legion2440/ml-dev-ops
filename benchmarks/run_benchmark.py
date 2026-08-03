#!/usr/bin/env python3
"""Measure the formal ResNet optimization pair and publish only a passing bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.aggregate_results import (  # noqa: E402
    AggregationError,
    aggregate,
    parse_raw_csv,
    render_report,
    summarize_paired_measurements,
)
from benchmarks.environment_guard import (  # noqa: E402
    BoundaryClient,
    GuardError,
    HostObserver,
    classify_trial,
    collect_nvidia_sample,
    read_jsonl,
    recompute_guard,
)
from client.transport import HttpTransport, RepositoryController  # noqa: E402

CONFIG_PATH = REPOSITORY_ROOT / "benchmarks/configs/benchmark.json"
PAIR_CONTRACT_PATH = REPOSITORY_ROOT / "shared/benchmark-model-pair.json"
CLIENT_CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarking"
LATEST_CANDIDATE_PATH = CACHE_ROOT / "latest-passed.json"
RUNTIME_SOURCE_PATHS = (
    "benchmarks/run_benchmark.py",
    "benchmarks/aggregate_results.py",
    "benchmarks/clock_guard.c",
    "benchmarks/environment_guard.py",
    "benchmarks/configs/benchmark.json",
    "schemas/benchmark-config.schema.json",
    "schemas/benchmark-evidence.schema.json",
    "schemas/benchmark-host-telemetry.schema.json",
    "schemas/benchmark-model-pair.schema.json",
    "scripts/validate_benchmark.py",
    "scripts/validate_benchmark_evidence.py",
    "shared/benchmark-model-pair.json",
    "docker-compose.yml",
    ".env.example",
)
# Backward-compatible name for callers that only need the tracked source set.
SOURCE_FINGERPRINT_PATHS = RUNTIME_SOURCE_PATHS
GPU_METRIC = re.compile(
    r'^nv_gpu_(?P<name>utilization|memory_total_bytes|memory_used_bytes)'
    r'\{gpu_uuid="(?P<uuid>[^"]+)"\}\s+(?P<value>[-+0-9.eE]+)$'
)
STABILITY_PASS = re.compile(
    r"Pass \[(?P<attempt>[0-9]+)\] throughput: "
    r"(?P<throughput>[0-9]+(?:\.[0-9]+)?) infer/sec\. p95 latency: "
    r"(?P<p95>[0-9]+) usec"
)
CLIENT_REQUEST_COUNT = re.compile(r"Client:\s*\n\s*Request count: (?P<count>[0-9]+)")
STATISTIC_COUNT_FIELDS = (
    "request_count",
    "inference_count",
    "execution_count",
)
STATISTIC_DURATION_FIELDS = (
    "request_duration_ns",
    "queue_duration_ns",
    "compute_input_duration_ns",
    "compute_infer_duration_ns",
    "compute_output_duration_ns",
)


class BenchmarkError(RuntimeError):
    """The benchmark could not produce a publishable passing result."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path.name} must contain a JSON object")
    return value


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(content, bytes):
        temporary.write_bytes(content)
    else:
        temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_source_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """Return the benchmark source manifest captured with a runtime candidate."""
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in RUNTIME_SOURCE_PATHS
    }


def source_fingerprint(root: Path = REPOSITORY_ROOT) -> str:
    """Hash exact benchmark source bytes; retained as a historical run identity."""
    digest = hashlib.sha256()
    for relative in RUNTIME_SOURCE_PATHS:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def replacement_decision(
    classification: str,
    slot_id: str,
    slot_attempt: int,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Return the production control-flow decision for one formal slot attempt."""
    if classification == "VALID":
        return {
            "action": "accept",
            "slot_id": slot_id,
            "accepted_attempt": slot_attempt,
        }
    if classification == "CONTAMINATED":
        if slot_attempt >= maximum_attempts:
            return {
                "action": "abort_environment",
                "slot_id": slot_id,
                "attempt": slot_attempt,
            }
        return {
            "action": "retry_same_slot",
            "slot_id": slot_id,
            "next_attempt": slot_attempt + 1,
        }
    return {
        "action": "abort_error",
        "slot_id": slot_id,
        "attempt": slot_attempt,
    }


def _perf_analyzer_semantic_probe(
    config: dict[str, Any], pair: dict[str, Any]
) -> list[dict[str, Any]]:
    input_path = Path("__compat_input__")
    csv_path = Path("__compat_output__.csv")
    probes: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        for role in ("baseline", "optimized"):
            command = build_perf_analyzer_command(
                config, pair, role, scenario, input_path, csv_path
            )
            probes.append(
                {
                    "scenario": scenario["id"],
                    "role": role,
                    "command": [
                        "<input>"
                        if item.replace("\\", "/") == input_path.as_posix()
                        else (
                            "<csv>"
                            if item.replace("\\", "/") == csv_path.as_posix()
                            else item.replace("\\", "/")
                        )
                        for item in command
                    ],
                }
            )
    return probes


def _aggregation_semantic_probe(config: dict[str, Any]) -> dict[str, Any]:
    def metrics(*, latency: float, throughput: float) -> dict[str, float]:
        return {"avg_latency_ms": latency, "infer_per_sec": throughput}

    inputs = {
        "latency": {
            "primary_metric": "mean_client_latency_ms",
            "baseline": [metrics(latency=100.0, throughput=100.0)] * 4,
            "optimized": [
                metrics(latency=value, throughput=100.0)
                for value in (80.0, 90.0, 110.0, 70.0)
            ],
        },
        "throughput": {
            "primary_metric": "infer_per_sec",
            "baseline": [metrics(latency=100.0, throughput=100.0)] * 4,
            "optimized": [
                metrics(latency=100.0, throughput=value)
                for value in (120.0, 90.0, 130.0, 100.0)
            ],
        },
    }
    output: dict[str, Any] = {}
    for probe_id, probe in inputs.items():
        summary = summarize_paired_measurements(
            probe["primary_metric"],
            config["execution_order"],
            probe["baseline"],
            probe["optimized"],
            config["acceptance"],
        )
        output[probe_id] = {
            "paired_improvement_pct": [
                item["paired_improvement_pct"] for item in summary["pairs"]
            ],
            "directional_improvement": [
                item["directional_improvement"] for item in summary["pairs"]
            ],
            "median_paired_improvement_pct": summary[
                "median_paired_improvement_pct"
            ],
            "improved_pair_count": summary["improved_pair_count"],
            "strength": summary["strength"],
            "gate_passed": summary["gate_passed"],
        }
    return output


def _guard_semantic_probe(config: dict[str, Any]) -> dict[str, Any]:
    guard = config["environment_guard"]
    boundary = {
        "baseline_start_seq": 1,
        "baseline_end_seq": 5,
        "guard_start_seq": 5,
        "guard_end_seq": 7,
    }

    def engine(
        pid: int,
        name: str,
        utilization: float,
        engine_type: str = "3D",
    ) -> dict[str, Any]:
        return {
            "pid": pid,
            "process_name": name,
            "engine_type": engine_type,
            "utilization_percent": utilization,
        }

    def sample(sequence: int, engines: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "sequence": sequence,
            "host_monotonic_ns": sequence * 1_000_000_000,
            "collection_ok": True,
            "gpu_engine_inventory": [
                {
                    "pid": item["pid"],
                    "process_name": item["process_name"],
                    "engine_types": [item["engine_type"]],
                }
                for item in engines
            ],
            "gpu_engines": engines,
        }

    owned = engine(10, "vmmemWSL.exe", 10.0)
    cases: dict[str, list[dict[str, Any]]] = {
        "clean_owned_activity": [sample(index, [owned]) for index in range(1, 8)],
        "forbidden_process_activity": [
            *[
                sample(index, [owned, engine(20, "chrome.exe", 0.0)])
                for index in range(1, 6)
            ],
            sample(6, [owned, engine(20, "chrome.exe", 1.0)]),
            sample(7, [owned, engine(20, "chrome.exe", 0.0)]),
        ],
        "new_gpu_process_activity": [
            *[sample(index, [owned]) for index in range(1, 6)],
            sample(6, [owned, engine(30, "new-helper.exe", 1.0)]),
            sample(7, [owned]),
        ],
        "baseline_idle_process_became_active": [
            *[
                sample(index, [owned, engine(4, "System", 0.0, "Copy")])
                for index in range(1, 6)
            ],
            sample(6, [owned, engine(4, "System", 1.0, "Copy")]),
            sample(7, [owned, engine(4, "System", 0.0, "Copy")]),
        ],
    }
    gap = [sample(index, [owned]) for index in (1, 2, 3, 5, 6, 7)]
    cases["telemetry_sequence_gap"] = gap
    attribution: dict[str, Any] = {}
    for case_id, telemetry in cases.items():
        result = recompute_guard(telemetry, boundary, guard)
        reasons = result.get("reasons", [])
        attribution[case_id] = {
            "classification": result["classification"],
            "reasons": [
                {
                    key: item[key]
                    for key in (
                        "pid",
                        "process_name",
                        "engine_type",
                        "reason",
                        "first_sequence",
                    )
                }
                if isinstance(item, dict)
                else item
                for item in reasons
            ],
        }
    trial_classification = {
        case_id: classify_trial(**arguments)
        for case_id, arguments in {
            "valid": {
                "scenario_status": "formal",
                "runtime_error": False,
                "guard_classification": "CLEAN",
                "measurement_valid": True,
            },
            "contaminated": {
                "scenario_status": "formal",
                "runtime_error": False,
                "guard_classification": "CONTAMINATED",
                "measurement_valid": True,
            },
            "runtime_error": {
                "scenario_status": "formal",
                "runtime_error": True,
                "guard_classification": "CLEAN",
                "measurement_valid": True,
            },
            "guard_error": {
                "scenario_status": "formal",
                "runtime_error": False,
                "guard_classification": "ERROR",
                "measurement_valid": True,
            },
            "non_formal_scenario": {
                "scenario_status": "diagnostic",
                "runtime_error": False,
                "guard_classification": "CLEAN",
                "measurement_valid": True,
            },
            "invalid_measurement": {
                "scenario_status": "formal",
                "runtime_error": False,
                "guard_classification": "CLEAN",
                "measurement_valid": False,
            },
        }.items()
    }
    maximum = int(guard["max_contaminated_attempts_per_slot"])
    replacement = {
        case_id: replacement_decision(classification, "canonical-slot", attempt, maximum)
        for case_id, classification, attempt in (
            ("valid", "VALID", 1),
            ("contaminated_retry", "CONTAMINATED", 1),
            ("contaminated_limit", "CONTAMINATED", maximum),
            ("error", "ERROR", 1),
        )
    }
    return {
        "attribution": attribution,
        "trial_classification": trial_classification,
        "replacement": replacement,
    }


def benchmark_compatibility_projection(
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Project only semantics that can change interpretation of a measurement."""
    config = _json(root / "benchmarks/configs/benchmark.json")
    pair = _json(root / "shared/benchmark-model-pair.json")
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    if not isinstance(compose, dict) or not isinstance(compose.get("services"), dict):
        raise BenchmarkError("docker-compose.yml must contain a services object")
    services = compose["services"]
    triton = services.get("triton")
    runner = services.get("benchmark-runner")
    if not isinstance(triton, dict) or not isinstance(runner, dict):
        raise BenchmarkError("benchmark Compose services are missing")
    env = _load_env_values(root / ".env.example")
    triton_command_prefixes = (
        "--model-repository=",
        "--model-control-mode=",
        "--disable-auto-complete-config",
        "--allow-http=",
        "--http-port=",
        "--allow-grpc=",
        "--grpc-port=",
        "--allow-metrics=",
        "--metrics-port=",
    )
    triton_projection = {
        "build": triton.get("build"),
        "image": triton.get("image"),
        "command": [
            item
            for item in triton.get("command", [])
            if isinstance(item, str) and item.startswith(triton_command_prefixes)
        ],
        "model_repository_mount": [
            item
            for item in triton.get("volumes", [])
            if isinstance(item, dict) and item.get("target") == "/models"
        ],
        "gpu_reservation": triton.get("deploy"),
        "networks": triton.get("networks"),
    }
    runner_projection = {
        field: runner.get(field)
        for field in (
            "image",
            "profiles",
            "working_dir",
            "command",
            "environment",
            "volumes",
            "networks",
        )
    }
    pair_projection = {
        field: pair[field]
        for field in (
            "schema_version",
            "pair_id",
            "logical_model_id",
            "baseline",
            "optimized",
            "common_contract",
            "common_contract_sha256",
            "weights_sha256",
            "parity",
            "declared_build_target",
        )
    }
    return {
        "schema_version": 2,
        "methodology": config,
        "model_pair": pair_projection,
        "aggregation_semantic_probe": _aggregation_semantic_probe(config),
        "perf_analyzer_command_probes": _perf_analyzer_semantic_probe(config, pair),
        "guard_semantic_probe": _guard_semantic_probe(config),
        "deployment": {
            "images": {
                "triton": env.get("TRITON_IMAGE"),
                "sdk": env.get("TRITON_SDK_IMAGE"),
            },
            "triton": triton_projection,
            "benchmark_runner": runner_projection,
            "backend_network": compose.get("networks", {}).get("backend"),
        },
    }


def _http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30.0) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise BenchmarkError(f"{url} did not return a JSON object")
    return value


def _http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30.0) as response:
        return response.read().decode("utf-8")


def _model_metric_snapshot(endpoint: str, model: str, version: str) -> dict[str, Any]:
    """Read cumulative Triton statistics for exactly one model version."""
    value = _http_json(
        f"http://{endpoint}/v2/models/{model}/versions/{version}/stats"
    )
    rows = value.get("model_stats")
    if not isinstance(rows, list):
        raise BenchmarkError("Triton model statistics response has no model_stats list")
    matching = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("name") == model
        and str(row.get("version")) == version
    ]
    if len(matching) != 1:
        raise BenchmarkError(
            f"Expected one statistics row for {model} version {version}, got {len(matching)}"
        )
    row = matching[0]
    inference_stats = row.get("inference_stats")
    if not isinstance(inference_stats, dict):
        raise BenchmarkError("Triton model statistics has no inference_stats object")

    def statistic(name: str) -> dict[str, Any]:
        item = inference_stats.get(name)
        if not isinstance(item, dict) or "count" not in item or "ns" not in item:
            raise BenchmarkError(f"Triton model statistics is missing {name}")
        return item

    success = statistic("success")
    queue = statistic("queue")
    compute_input = statistic("compute_input")
    compute_infer = statistic("compute_infer")
    compute_output = statistic("compute_output")
    try:
        snapshot = {
            "model": model,
            "version": version,
            "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "request_count": int(success["count"]),
            "inference_count": int(row["inference_count"]),
            "execution_count": int(row["execution_count"]),
            "request_duration_ns": int(success["ns"]),
            "queue_duration_ns": int(queue["ns"]),
            "compute_input_duration_ns": int(compute_input["ns"]),
            "compute_infer_duration_ns": int(compute_infer["ns"]),
            "compute_output_duration_ns": int(compute_output["ns"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkError("Triton model statistics contains invalid counters") from error
    if any(snapshot[field] < 0 for field in (*STATISTIC_COUNT_FIELDS, *STATISTIC_DURATION_FIELDS)):
        raise BenchmarkError("Triton model statistics contains a negative counter")
    return snapshot


def _model_metric_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Calculate a self-contained, validator-recomputable per-pass delta."""
    if (before.get("model"), before.get("version")) != (
        after.get("model"),
        after.get("version"),
    ):
        raise BenchmarkError("Triton statistics snapshots identify different model versions")
    delta: dict[str, Any] = {}
    for field in (*STATISTIC_COUNT_FIELDS, *STATISTIC_DURATION_FIELDS):
        value = int(after[field]) - int(before[field])
        if value < 0:
            raise BenchmarkError(f"Triton statistics counter decreased: {field}")
        delta[field] = value
    request_count = delta["request_count"]
    if request_count <= 0:
        raise BenchmarkError("Triton statistics pass delta has no successful requests")
    delta["per_request_us"] = {
        field.removesuffix("_duration_ns"): round(
            delta[field] / request_count / 1000.0, 6
        )
        for field in STATISTIC_DURATION_FIELDS
    }
    return delta


def _wait_health(transport: HttpTransport, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, bool] = {}
    while True:
        last = transport.health()
        if last == {"live": True, "ready": True}:
            return
        if time.monotonic() >= deadline:
            raise BenchmarkError(f"Triton health timeout: {last}")
        time.sleep(0.25)


def _wait_ready_set(
    controller: RepositoryController,
    expected: set[tuple[str, str]],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    current: set[tuple[str, str]] = set()
    while True:
        current = controller.ready_set()
        if current == expected:
            return
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"READY state timeout: expected={sorted(expected)}, actual={sorted(current)}"
            )
        time.sleep(0.25)


def _ready_rows(values: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"model": model, "version": version}
        for model, version in sorted(values, key=lambda item: (item[0], int(item[1])))
    ]


def _validate_restorable_initial_state(
    initial: set[tuple[str, str]], client_contract: dict[str, Any]
) -> None:
    known = client_contract["models"]
    for model in {name for name, _ in initial}:
        if model not in known:
            raise BenchmarkError(f"Initial READY state contains unknown model: {model}")
        actual = {version for name, version in initial if name == model}
        expected = set(known[model]["versions"])
        if actual != expected:
            raise BenchmarkError(
                f"Initial READY state for {model} is partial: {sorted(actual)}"
            )


def _unload_all(
    controller: RepositoryController,
    ready: set[tuple[str, str]],
    timeout_seconds: float,
) -> None:
    for model in sorted({name for name, _ in ready}):
        controller.unload(model)
    _wait_ready_set(controller, set(), timeout_seconds)


def _load_role(
    controller: RepositoryController,
    pair: dict[str, Any],
    role: str,
    timeout_seconds: float,
) -> set[tuple[str, str]]:
    entry = pair[role]
    controller.load(entry["model"])
    expected = {(entry["model"], version) for version in entry["available_versions"]}
    _wait_ready_set(controller, expected, timeout_seconds)
    return expected


def _restore(
    controller: RepositoryController,
    initial: set[tuple[str, str]],
    timeout_seconds: float,
) -> None:
    current = controller.ready_set()
    _unload_all(controller, current, timeout_seconds)
    for model in sorted({name for name, _ in initial}):
        controller.load(model)
    _wait_ready_set(controller, initial, timeout_seconds)


def _normalize_runtime_contract(
    metadata: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    def tensor(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": item["name"],
                "datatype": item["datatype"],
                "shape": item["shape"],
            }
            for item in items
        ]

    instance_groups = [
        {
            "kind": item.get("kind"),
            "count": int(item.get("count", 1)),
            "gpus": [int(gpu) for gpu in item.get("gpus", [])],
        }
        for item in config.get("instance_group", [])
    ]
    return {
        "inputs": tensor(metadata["inputs"]),
        "outputs": tensor(metadata["outputs"]),
        "max_batch_size": int(config["max_batch_size"]),
        "dynamic_batching": config.get("dynamic_batching", {}),
        # Instance names are model-specific and do not affect fairness. Only the
        # scheduling/resource fields are part of the comparable runtime contract.
        "instance_group": instance_groups,
    }


def _preflight_contracts(
    controller: RepositoryController,
    pair: dict[str, Any],
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for role in ("baseline", "optimized"):
        entry = pair[role]
        _load_role(controller, pair, role, timeout_seconds)
        try:
            base = f"http://{endpoint}/v2/models/{entry['model']}"
            metadata = _http_json(f"{base}/versions/{entry['version']}")
            config = _http_json(f"{base}/config")
            snapshots[role] = {
                "model": entry["model"],
                "version": entry["version"],
                "runtime_contract": _normalize_runtime_contract(metadata, config),
            }
        finally:
            controller.unload(entry["model"])
            _wait_ready_set(controller, set(), timeout_seconds)
    baseline = snapshots["baseline"]["runtime_contract"]
    optimized = snapshots["optimized"]["runtime_contract"]
    if baseline != optimized:
        raise BenchmarkError("ONNX and TensorRT runtime contracts differ")
    common = pair["common_contract"]
    expected_inputs = [
        {
            "name": common["input"]["name"],
            "datatype": common["input"]["dtype"],
            "shape": common["input"]["shape"],
        }
    ]
    expected_outputs = [
        {
            "name": common["output"]["name"],
            "datatype": common["output"]["dtype"],
            "shape": common["output"]["shape"],
        }
    ]
    if baseline["inputs"] != expected_inputs or baseline["outputs"] != expected_outputs:
        raise BenchmarkError("Live tensor metadata differs from the shared pair contract")
    if baseline["max_batch_size"] != common["max_batch_size"]:
        raise BenchmarkError("Live max_batch_size differs from the shared pair contract")
    expected_dynamic = common["scheduling"]["dynamic_batching"]
    actual_dynamic = baseline["dynamic_batching"]
    if int(actual_dynamic.get("max_queue_delay_microseconds", -1)) != int(
        expected_dynamic["max_queue_delay_microseconds"]
    ):
        raise BenchmarkError("Live dynamic batching queue delay differs from the pair contract")
    actual_preferred = [int(value) for value in actual_dynamic.get("preferred_batch_size", [])]
    if actual_preferred != expected_dynamic["preferred_batch_sizes"]:
        raise BenchmarkError("Live preferred batch sizes differ from the pair contract")
    snapshots["common_contract_sha256"] = pair["common_contract_sha256"]
    return snapshots


def _gpu_runtime(metrics_url: str) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {}
    for raw_line in _http_text(metrics_url).splitlines():
        match = GPU_METRIC.fullmatch(raw_line.strip())
        if match:
            metrics.setdefault(match.group("uuid"), {})[match.group("name")] = float(
                match.group("value")
            )
    complete = {
        gpu_uuid: values
        for gpu_uuid, values in metrics.items()
        if {"utilization", "memory_total_bytes", "memory_used_bytes"} <= set(values)
    }
    if len(complete) != 1:
        raise BenchmarkError("Expected exactly one complete GPU metric identity")
    gpu_uuid, values = next(iter(complete.items()))
    return {
        "gpu_uuid": gpu_uuid,
        "gpu_utilization_fraction_before": values["utilization"],
        "gpu_memory_total_bytes": int(values["memory_total_bytes"]),
        "gpu_memory_used_bytes_before": int(values["memory_used_bytes"]),
    }


def build_perf_analyzer_command(
    config: dict[str, Any],
    pair: dict[str, Any],
    role: str,
    scenario: dict[str, Any],
    input_directory: Path,
    csv_path: Path,
) -> list[str]:
    measurement = config["measurement"]
    entry = pair[role]
    command = [
        "perf_analyzer",
        "-v",
        "--service-kind",
        config["service_kind"],
        "-m",
        entry["model"],
        "-x",
        entry["version"],
        "-u",
        config["endpoints"]["http"],
        "-i",
        config["protocol"],
        "-b",
        str(scenario["batch_size"]),
        "--concurrency-range",
        str(scenario["concurrency"]),
        "--measurement-mode",
        measurement["mode"],
        "--measurement-request-count",
        str(measurement["request_count_per_window"]),
        "--warmup-request-count",
        str(measurement["warmup_request_count"]),
        "--stability-percentage",
        str(scenario["perf_analyzer_completion_tolerance_pct"]),
        "--percentile",
        str(measurement["reporting_percentile"]),
        "--max-trials",
        str(measurement["max_measurement_windows"]),
        "--input-data",
        str(input_directory),
        "--input-tensor-format",
        "binary",
        "--output-tensor-format",
        "binary",
        "--collect-metrics",
        "--verbose-csv",
        "--metrics-url",
        config["endpoints"]["metrics"],
        "--metrics-interval",
        str(measurement["metrics_interval_ms"]),
    ]
    command.extend(["-f", str(csv_path)])
    return command


def _sanitized_command(command: list[str], run_root: Path) -> list[str]:
    rendered: list[str] = []
    for value in command:
        value = value.replace(str(run_root), "<run>")
        value = value.replace(str(REPOSITORY_ROOT), "<repository>")
        rendered.append(value.replace("\\", "/"))
    return rendered


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGCONT)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def _stream_perf_analyzer(
    command: list[str],
    timeout_seconds: float,
    environment: dict[str, str],
    boundary_client: BoundaryClient,
    trial_id: str,
    start_ack: dict[str, Any],
    endpoint: str,
    model: str,
    version: str,
    warmup_request_count: int,
) -> tuple[int, str, list[dict[str, Any]]]:
    """Run PA and freeze it after every reported pass for diagnostic snapshots."""
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise BenchmarkError("stdbuf is required for pass-boundary diagnostics")
    before = _model_metric_snapshot(endpoint, model, version)
    previous_boundary = {
        "sequence": int(start_ack["guard_start_seq"]),
        "observed_at_utc": start_ack["observed_at_utc"],
        "host_monotonic_ns": int(start_ack["host_monotonic_ns"]),
    }
    diagnostics: list[dict[str, Any]] = []
    output: list[str] = []
    process = subprocess.Popen(
        [stdbuf, "-oL", "-eL", *command],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=environment,
    )
    if process.stdout is None:
        _stop_process_group(process)
        raise BenchmarkError("Perf Analyzer output pipe was not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds

    def observe_line(line: str) -> None:
        nonlocal before, previous_boundary
        output.append(line)
        matches = list(STABILITY_PASS.finditer(line))
        for match in matches:
            attempt = int(match.group("attempt"))
            if attempt != len(diagnostics) + 1:
                raise BenchmarkError("Perf Analyzer pass numbers are not consecutive")
            paused = False
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                    paused = True
                except ProcessLookupError:
                    pass
            try:
                pass_ack = boundary_client.checkpoint(trial_id, attempt)
                after = _model_metric_snapshot(endpoint, model, version)
                delta = _model_metric_delta(before, after)
                diagnostics.append(
                    {
                        "attempt": attempt,
                        "infer_per_sec": float(match.group("throughput")),
                        "p95_latency_us": int(match.group("p95")),
                        "includes_initial_warmup": attempt == 1
                        and warmup_request_count > 0,
                        "guard_boundary": {
                            "guard_start_seq": previous_boundary["sequence"],
                            "guard_end_seq": int(pass_ack["sequence"]),
                            "guard_started_at_utc": previous_boundary[
                                "observed_at_utc"
                            ],
                            "guard_ended_at_utc": pass_ack["observed_at_utc"],
                            "guard_started_monotonic_ns": previous_boundary[
                                "host_monotonic_ns"
                            ],
                            "guard_ended_monotonic_ns": int(
                                pass_ack["host_monotonic_ns"]
                            ),
                        },
                        "triton_statistics": {
                            "before": before,
                            "after": after,
                            "delta": delta,
                        },
                    }
                )
                before = after
                previous_boundary = {
                    "sequence": int(pass_ack["sequence"]),
                    "observed_at_utc": pass_ack["observed_at_utc"],
                    "host_monotonic_ns": int(pass_ack["host_monotonic_ns"]),
                }
            finally:
                if paused and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass

    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(timeout=min(0.25, remaining))
            if events:
                line = process.stdout.readline()
                if line:
                    observe_line(line)
                    continue
            if process.poll() is not None:
                remainder = process.stdout.read()
                for line in remainder.splitlines(keepends=True):
                    observe_line(line)
                break
        if timed_out:
            output.append("command timed out\n")
            return_code = 124
        else:
            return_code = process.wait(timeout=5.0)
    finally:
        selector.close()
        if process.poll() is None:
            _stop_process_group(process)
    return return_code, "".join(output), diagnostics


def _run_perf(
    run_root: Path,
    publish_root: Path,
    config: dict[str, Any],
    pair: dict[str, Any],
    role: str,
    scenario: dict[str, Any],
    repetition: int,
    order_position: int,
    input_directory: Path,
    input_sha256: str,
    clock_guard_path: Path,
    clock_guard_sha256: str,
    boundary_client: BoundaryClient,
    slot_id: str,
    slot_attempt: int,
) -> dict[str, Any]:
    trial_id = f"{slot_id}-attempt-{slot_attempt:02d}"
    work_directory = run_root / "work" / trial_id
    work_directory.mkdir(parents=True, exist_ok=True)
    csv_path = work_directory / "result.csv"
    command = build_perf_analyzer_command(
        config, pair, role, scenario, input_directory, csv_path
    )
    start_ack = boundary_client.start(
        trial_id,
        {
            "slot_id": slot_id,
            "slot_attempt": slot_attempt,
            "scenario": scenario["id"],
            "repetition": repetition,
            "order_position": order_position,
            "role": role,
        },
    )
    started = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    pass_diagnostics: list[dict[str, Any]] = []
    try:
        exit_code, combined, pass_diagnostics = _stream_perf_analyzer(
            command,
            float(config["measurement"]["command_timeout_seconds"]),
            {**os.environ, "LD_PRELOAD": str(clock_guard_path)},
            boundary_client,
            trial_id,
            start_ack,
            config["endpoints"]["http"],
            pair[role]["model"],
            pair[role]["version"],
            int(config["measurement"]["warmup_request_count"]),
        )
    except (OSError, ValueError, TypeError, KeyError, BenchmarkError, GuardError) as error:
        exit_code = 125
        combined = f"cb::Error: pass diagnostic failed: {error}\n"
    finished = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    end_ack = boundary_client.end(trial_id)
    boundary = {
        "baseline_start_seq": start_ack["baseline_start_seq"],
        "baseline_end_seq": start_ack["baseline_end_seq"],
        "guard_start_seq": start_ack["guard_start_seq"],
        "guard_end_seq": end_ack["guard_end_seq"],
        "guard_started_at_utc": start_ack["observed_at_utc"],
        "guard_ended_at_utc": end_ack["observed_at_utc"],
        "guard_started_monotonic_ns": start_ack["host_monotonic_ns"],
        "guard_ended_monotonic_ns": end_ack["host_monotonic_ns"],
    }
    telemetry = read_jsonl(run_root / "guard/telemetry.jsonl")
    guard_result = recompute_guard(
        telemetry, boundary, config["environment_guard"]
    )
    sanitized_log = combined.replace(str(run_root), "<run>").replace(
        str(REPOSITORY_ROOT), "<repository>"
    )
    measurement_windows = [
        {
            "attempt": int(match.group("attempt")),
            "infer_per_sec": float(match.group("throughput")),
            "p95_latency_us": int(match.group("p95")),
        }
        for match in STABILITY_PASS.finditer(combined)
    ]
    windows_are_complete = [item["attempt"] for item in measurement_windows] == list(
        range(1, len(measurement_windows) + 1)
    )
    diagnostics_are_complete = [
        {
            "attempt": item["attempt"],
            "infer_per_sec": item["infer_per_sec"],
            "p95_latency_us": item["p95_latency_us"],
        }
        for item in pass_diagnostics
    ] == measurement_windows
    pa_reported_stable = (
        exit_code == 0
        and csv_path.is_file()
        and bool(measurement_windows)
        and windows_are_complete
        and diagnostics_are_complete
        and "Failed to obtain stable measurement" not in combined
    )
    csv_parsed = False
    if exit_code == 0 and csv_path.is_file():
        parse_raw_csv(csv_path)
        csv_parsed = True
    client_request_counts = [
        int(match.group("count")) for match in CLIENT_REQUEST_COUNT.finditer(combined)
    ]
    client_request_count = client_request_counts[-1] if client_request_counts else 0
    request_count_complete = client_request_count >= (
        len(measurement_windows)
        * int(config["measurement"]["request_count_per_window"])
    )
    measurement_completed = (
        exit_code == 0
        and csv_parsed
        and bool(measurement_windows)
        and windows_are_complete
        and diagnostics_are_complete
        and request_count_complete
    )
    runtime_error = (
        "cb::Error:" in combined
        or "command timed out" in combined
        or not measurement_completed
    )
    classification = classify_trial(
        scenario_status=scenario["status"],
        runtime_error=runtime_error,
        guard_classification=guard_result["classification"],
        measurement_valid=measurement_completed,
    )
    if classification == "VALID":
        relative_directory = Path(
            f"benchmarks/results/raw/valid/{scenario['id']}/{slot_id}"
        )
    elif classification == "CONTAMINATED":
        relative_directory = Path(
            f"benchmarks/results/raw/contaminated/{scenario['id']}/{slot_id}/attempt-{slot_attempt:02d}"
        )
    else:
        relative_directory = Path(f"diagnostics/{classification.lower()}/{trial_id}")
    artifact_root = publish_root if classification in {
        "VALID",
        "CONTAMINATED",
    } else run_root
    target_directory = artifact_root / relative_directory
    target_directory.mkdir(parents=True, exist_ok=True)
    csv_relative = (relative_directory / f"{role}.csv").as_posix()
    log_relative = (relative_directory / f"{role}.log").as_posix()
    sidecar_relative = (relative_directory / f"{role}.json").as_posix()
    if csv_path.is_file():
        shutil.copy2(csv_path, artifact_root / csv_relative)
    _write(artifact_root / log_relative, sanitized_log.replace("\r\n", "\n"))
    recorded_csv_relative = csv_relative if csv_path.is_file() else None
    sidecar = {
        "schema_version": 1,
        "trial_id": trial_id,
        "slot_id": slot_id,
        "slot_attempt": slot_attempt,
        "classification": classification,
        "guard_classification": guard_result["classification"],
        "guard_reasons": guard_result.get("reasons", []),
        "guard_baseline_processes": guard_result.get("baseline_processes", []),
        "guard_baseline_active_processes": guard_result.get(
            "baseline_active_processes", []
        ),
        "guard_baseline_engines": guard_result.get("baseline_engines", []),
        "guard_boundary": boundary,
        "scenario": scenario["id"],
        "scenario_status": scenario["status"],
        "perf_analyzer_completion_tolerance_pct": scenario[
            "perf_analyzer_completion_tolerance_pct"
        ],
        "measurement_windows_used": len(measurement_windows),
        "measurement_windows": measurement_windows,
        "client_request_count": client_request_count,
        "pass_diagnostics": pass_diagnostics,
        "repetition": repetition,
        "order_position": order_position,
        "role": role,
        "model": pair[role]["model"],
        "version": pair[role]["version"],
        "batch_size": scenario["batch_size"],
        "concurrency": scenario["concurrency"],
        "protocol": config["protocol"],
        "input_sha256": input_sha256,
        "clock_guard_sha256": clock_guard_sha256,
        "container_perf_started_at_utc": started,
        "container_perf_ended_at_utc": finished,
        "command": _sanitized_command(command, run_root),
        "exit_code": exit_code,
        "measurement_completed": measurement_completed,
        "pa_reported_stable": pa_reported_stable,
        "errors": 1 if classification == "ERROR" else 0,
        "csv_path": recorded_csv_relative,
        "log_path": log_relative,
    }
    _write(artifact_root / sidecar_relative, _canonical(sidecar))
    return {
        "trial_id": trial_id,
        "slot_id": slot_id,
        "slot_attempt": slot_attempt,
        "classification": classification,
        "guard_classification": guard_result["classification"],
        "guard_reasons": guard_result.get("reasons", []),
        "guard_baseline_processes": guard_result.get("baseline_processes", []),
        "guard_baseline_active_processes": guard_result.get(
            "baseline_active_processes", []
        ),
        "guard_baseline_engines": guard_result.get("baseline_engines", []),
        "guard_boundary": boundary,
        "scenario": scenario["id"],
        "scenario_status": scenario["status"],
        "perf_analyzer_completion_tolerance_pct": scenario[
            "perf_analyzer_completion_tolerance_pct"
        ],
        "measurement_windows_used": len(measurement_windows),
        "measurement_windows": measurement_windows,
        "client_request_count": client_request_count,
        "pass_diagnostics": pass_diagnostics,
        "measurement_completed": measurement_completed,
        "pa_reported_stable": pa_reported_stable,
        "errors": 1 if classification == "ERROR" else 0,
        "repetition": repetition,
        "order_position": order_position,
        "role": role,
        "model": pair[role]["model"],
        "version": pair[role]["version"],
        "input_sha256": input_sha256,
        "clock_guard_sha256": clock_guard_sha256,
        "csv_path": recorded_csv_relative,
        "csv_sha256": sha256(artifact_root / csv_relative) if csv_path.is_file() else None,
        "log_path": log_relative,
        "log_sha256": sha256(artifact_root / log_relative),
        "sidecar_path": sidecar_relative,
        "sidecar_sha256": sha256(artifact_root / sidecar_relative),
    }


def _perf_analyzer_version() -> str:
    process = subprocess.run(
        ["perf_analyzer", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    if process.returncode != 0 or not output:
        raise BenchmarkError("perf_analyzer --version failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1]


def _compile_clock_guard(run_root: Path) -> tuple[Path, str]:
    source = REPOSITORY_ROOT / "benchmarks/clock_guard.c"
    output = run_root / "clock_guard.so"
    process = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-pthread", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not output.is_file():
        raise BenchmarkError("clock guard compilation failed: " + process.stderr.strip())
    return output, sha256(output)


def _runtime_summary(
    config: dict[str, Any], result: dict[str, Any], gpu: dict[str, Any]
) -> dict[str, Any]:
    utilizations = [
        values["gpu_utilization_fraction"]
        for role in result["aggregates"].values()
        for values in role.values()
        if values["gpu_utilization_fraction"] is not None
    ]
    memories = [
        values["gpu_memory_used_bytes"]
        for role in result["aggregates"].values()
        for values in role.values()
        if values["gpu_memory_used_bytes"] is not None
    ]
    server = _http_json(f"http://{config['endpoints']['http']}/v2")
    sdk_image = os.environ.get("TRITON_SDK_IMAGE", "")
    triton_image = os.environ.get("TRITON_IMAGE", "")
    if not sdk_image or not triton_image:
        raise BenchmarkError("TRITON_IMAGE and TRITON_SDK_IMAGE must be provided")
    sdk_tag = sdk_image.rsplit(":", 1)[-1]
    return {
        **gpu,
        "gpu_utilization_fraction_median": float(np.median(utilizations)),
        "gpu_memory_used_bytes_median": int(np.median(memories)),
        "triton_name": str(server["name"]),
        "triton_version": str(server["version"]),
        "triton_image": triton_image,
        "sdk_version": sdk_tag.removesuffix("-py3-sdk"),
        "sdk_image": sdk_image,
        "perf_analyzer_version": _perf_analyzer_version(),
    }


def _artifact(path: str, publish_root: Path) -> dict[str, str]:
    return {"path": path, "sha256": sha256(publish_root / path)}


def _create_evidence(
    publish_root: Path,
    config: dict[str, Any],
    pair: dict[str, Any],
    runtime: dict[str, Any],
    result: dict[str, Any],
    input_sha256: str,
    initial: set[tuple[str, str]],
    final: set[tuple[str, str]],
    raw_index: dict[str, Any],
) -> dict[str, Any]:
    latency = result["comparisons"]["latency"]
    throughput = result["comparisons"]["throughput"]

    def evidence_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
        return {
            "primary_metric": comparison["primary_metric"],
            "median_paired_improvement_pct": comparison[
                "median_paired_improvement_pct"
            ],
            "improved_pair_count": comparison["improved_pair_count"],
            "strength": comparison["strength"],
            "gate_passed": comparison["gate_passed"],
            "pairs": [
                {
                    "repetition": item["repetition"],
                    "execution_order": item["execution_order"],
                    "baseline": {
                        "mean_client_latency_ms": item["baseline_metrics"][
                            "avg_latency_ms"
                        ],
                        "p50_latency_ms": item["baseline_metrics"][
                            "p50_latency_ms"
                        ],
                        "p95_latency_ms": item["baseline_metrics"][
                            "p95_latency_ms"
                        ],
                        "infer_per_sec": item["baseline_metrics"]["infer_per_sec"],
                    },
                    "optimized": {
                        "mean_client_latency_ms": item["optimized_metrics"][
                            "avg_latency_ms"
                        ],
                        "p50_latency_ms": item["optimized_metrics"][
                            "p50_latency_ms"
                        ],
                        "p95_latency_ms": item["optimized_metrics"][
                            "p95_latency_ms"
                        ],
                        "infer_per_sec": item["optimized_metrics"][
                            "infer_per_sec"
                        ],
                    },
                    "paired_improvement_pct": item["paired_improvement_pct"],
                    "directional_improvement": item["directional_improvement"],
                }
                for item in comparison["pairs"]
            ],
        }

    source_hashes = runtime_source_hashes()
    compatibility_projection = benchmark_compatibility_projection()
    return {
        "schema_version": 3,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "benchmark_config_sha256": sha256(CONFIG_PATH),
        "runtime_source_fingerprint_sha256": source_fingerprint(),
        "runtime_source_hashes": source_hashes,
        "runtime_source_manifest_sha256": canonical_sha256(source_hashes),
        "runtime_compatibility_projection": compatibility_projection,
        "runtime_compatibility_projection_sha256": canonical_sha256(
            compatibility_projection
        ),
        "pair_contract_sha256": sha256(PAIR_CONTRACT_PATH),
        "input_sha256": input_sha256,
        "pair": {
            "baseline": {
                "model": pair["baseline"]["model"],
                "version": pair["baseline"]["version"],
            },
            "optimized": {
                "model": pair["optimized"]["model"],
                "version": pair["optimized"]["version"],
            },
            "weights_sha256": pair["weights_sha256"],
            "common_contract_sha256": pair["common_contract_sha256"],
        },
        "runtime": runtime,
        "declared_build_target": pair["declared_build_target"],
        "measurement": {
            "scenario_count": len(config["scenarios"]),
            "repetitions": config["measurement"]["repetitions"],
            "run_count": len(config["scenarios"])
            * config["measurement"]["repetitions"]
            * 2,
            "valid_run_count": sum(
                run["classification"] == "VALID" for run in raw_index["runs"]
            ),
            "errors": 0,
            "scenarios": result["scenario_diagnostics"],
            "clock_guard_source_sha256": sha256(
                REPOSITORY_ROOT / "benchmarks/clock_guard.c"
            ),
        },
        "environment_guard": {
            "host_observer": config["environment_guard"]["host_observer"],
            "sample_interval_ms": config["environment_guard"]["sample_interval_ms"],
            "baseline_seconds": config["environment_guard"]["baseline_seconds"],
            "minimum_consecutive_baseline_samples": config["environment_guard"][
                "minimum_consecutive_baseline_samples"
            ],
            "maximum_sample_gap_ms": config["environment_guard"]["maximum_sample_gap_ms"],
            "gpu_engine_activity_threshold_percent": config["environment_guard"][
                "gpu_engine_activity_threshold_percent"
            ],
            "benchmark_owned_processes": config["environment_guard"][
                "benchmark_owned_processes"
            ],
            "forbidden_processes": config["environment_guard"][
                "forbidden_processes"
            ],
            "max_contaminated_attempts_per_slot": config["environment_guard"][
                "max_contaminated_attempts_per_slot"
            ],
            "formal_slot_count": 16,
            "valid_run_count": sum(
                run["classification"] == "VALID" for run in raw_index["runs"]
            ),
            "contaminated_run_count": len(raw_index["contaminated_runs"]),
            "telemetry": _artifact(
                "benchmarks/results/raw/environment-telemetry.jsonl", publish_root
            ),
            "actions": _artifact(
                "benchmarks/results/raw/environment-actions.jsonl", publish_root
            ),
            "contamination_records": [
                {
                    "slot_id": run["slot_id"],
                    "slot_attempt": run["slot_attempt"],
                    "scenario": run["scenario"],
                    "role": run["role"],
                    "guard_boundary": run["guard_boundary"],
                    "reasons": run["guard_reasons"],
                }
                for run in raw_index["contaminated_runs"]
            ],
        },
        "artifacts": {
            "raw_index": _artifact(
                "benchmarks/results/raw/index.json", publish_root
            ),
            "baseline": _artifact("benchmarks/results/baseline.csv", publish_root),
            "optimized": _artifact("benchmarks/results/optimized.csv", publish_root),
            "comparison": _artifact(
                "benchmarks/results/comparison.csv", publish_root
            ),
            "report": _artifact("benchmarks/report.md", publish_root),
        },
        "acceptance": {
            "rule": config["acceptance"],
            "latency": evidence_comparison(latency),
            "throughput": evidence_comparison(throughput),
            "passed": result["acceptance_passed"],
        },
        "initial_ready": _ready_rows(initial),
        "final_ready": _ready_rows(final),
        "runtime_state_restored": initial == final,
    }


def measure() -> None:
    config = _json(CONFIG_PATH)
    pair = _json(PAIR_CONTRACT_PATH)
    client_contract = _json(CLIENT_CONTRACT_PATH)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("BENCHMARK_RUN_ID", "")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", run_id):
        raise BenchmarkError(
            "BENCHMARK_RUN_ID is required; run the host orchestrator instead of measure directly"
        )
    run_root = CACHE_ROOT / f"run-{run_id}"
    publish_root = run_root / "publish"
    input_directory = run_root / "input"
    input_name = pair["common_contract"]["input"]["name"]
    input_shape = tuple(int(value) for value in pair["common_contract"]["input"]["shape"][1:])
    rng = np.random.default_rng(int(config["seed"]))
    input_tensor = rng.random(input_shape, dtype=np.float32)
    input_path = input_directory / input_name
    _write(input_path, input_tensor.tobytes(order="C"))
    input_sha256 = sha256(input_path)
    clock_guard_path, clock_guard_sha256 = _compile_clock_guard(run_root)
    endpoint = config["endpoints"]["http"]
    timeout_seconds = float(config["measurement"]["ready_timeout_seconds"])
    controller = RepositoryController(endpoint, timeout_seconds)
    transport = HttpTransport(endpoint, timeout_seconds)
    _wait_health(transport, timeout_seconds)
    initial = controller.ready_set()
    _validate_restorable_initial_state(initial, client_contract)
    result: dict[str, Any] | None = None
    runtime_contract: dict[str, Any] | None = None
    gpu: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    contaminated_runs: list[dict[str, Any]] = []
    boundary_client = BoundaryClient(run_root, config["environment_guard"])
    observer_stopped = False
    try:
        _unload_all(controller, initial, timeout_seconds)
        runtime_contract = _preflight_contracts(
            controller, pair, endpoint, timeout_seconds
        )
        runtime_contract_path = (
            publish_root / "benchmarks/results/raw/runtime-contract.json"
        )
        _write(runtime_contract_path, _canonical(runtime_contract))
        gpu = _gpu_runtime(f"http://{config['endpoints']['metrics']}")
        for scenario in config["scenarios"]:
            scenario_id = scenario["id"]
            for repetition, order in enumerate(config["execution_order"], 1):
                for position, role in enumerate(order, 1):
                    slot_id = (
                        f"slot-{scenario_id}-{repetition:02d}-{position:02d}-{role}"
                    )
                    slot_attempt = 1
                    while True:
                        _load_role(controller, pair, role, timeout_seconds)
                        try:
                            run = _run_perf(
                                run_root,
                                publish_root,
                                config,
                                pair,
                                role,
                                scenario,
                                repetition,
                                position,
                                input_directory,
                                input_sha256,
                                clock_guard_path,
                                clock_guard_sha256,
                                boundary_client,
                                slot_id,
                                slot_attempt,
                            )
                        finally:
                            controller.unload(pair[role]["model"])
                            _wait_ready_set(controller, set(), timeout_seconds)
                        classification = run["classification"]
                        decision = replacement_decision(
                            classification,
                            slot_id,
                            slot_attempt,
                            int(
                                config["environment_guard"][
                                    "max_contaminated_attempts_per_slot"
                                ]
                            ),
                        )
                        if decision["action"] == "accept":
                            runs.append(run)
                            break
                        if classification == "CONTAMINATED":
                            contaminated_runs.append(run)
                            if decision["action"] == "abort_environment":
                                raise BenchmarkError(
                                    "ENVIRONMENT_NOT_SUITABLE: maximum contaminated "
                                    f"attempts reached for {slot_id}"
                                )
                            slot_attempt = int(decision["next_attempt"])
                            time.sleep(
                                float(config["measurement"]["cooldown_seconds"])
                            )
                            continue
                        raise BenchmarkError(
                            f"benchmark runtime/telemetry error in {slot_id}"
                        )
                    time.sleep(float(config["measurement"]["cooldown_seconds"]))
        boundary_client.stop()
        observer_stopped = True
        telemetry_source = run_root / "guard/telemetry.jsonl"
        actions_source = run_root / "guard/actions.jsonl"
        if not actions_source.exists():
            _write(actions_source, "")
        telemetry_relative = Path(
            "benchmarks/results/raw/environment-telemetry.jsonl"
        )
        actions_relative = Path("benchmarks/results/raw/environment-actions.jsonl")
        telemetry_target = publish_root / telemetry_relative
        actions_target = publish_root / actions_relative
        telemetry_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(telemetry_source, telemetry_target)
        shutil.copy2(actions_source, actions_target)
        raw_index = {
            "schema_version": 1,
            "input_sha256": input_sha256,
            "clock_guard_source_sha256": sha256(
                REPOSITORY_ROOT / "benchmarks/clock_guard.c"
            ),
            "clock_guard_sha256": clock_guard_sha256,
            "runtime_contract_path": "benchmarks/results/raw/runtime-contract.json",
            "runtime_contract_sha256": sha256(runtime_contract_path),
            "environment_guard": {
                "telemetry_path": telemetry_relative.as_posix(),
                "telemetry_sha256": sha256(telemetry_target),
                "actions_path": actions_relative.as_posix(),
                "actions_sha256": sha256(actions_target),
                "sample_count": len(read_jsonl(telemetry_target)),
            },
            "runs": runs,
            "contaminated_runs": contaminated_runs,
        }
        _write(
            publish_root / "benchmarks/results/raw/index.json",
            _canonical(raw_index),
        )
        result = aggregate(publish_root, raw_index, config)
    finally:
        if not observer_stopped:
            try:
                boundary_client.stop()
            except (OSError, ValueError, KeyError, json.JSONDecodeError, GuardError):
                pass
        _restore(controller, initial, timeout_seconds)
        _wait_health(transport, timeout_seconds)
    final = controller.ready_set()
    if final != initial:
        raise BenchmarkError("Benchmark did not restore the initial READY state")
    if result is None or runtime_contract is None or gpu is None:
        raise BenchmarkError("Benchmark did not produce a complete candidate")
    runtime = _runtime_summary(config, result, gpu)
    report = render_report(config, pair, runtime, result, input_sha256)
    _write(publish_root / "benchmarks/report.md", report)
    if not result["acceptance_passed"]:
        raise BenchmarkError(
            f"Acceptance failed; diagnostic candidate retained at {run_root.name}"
        )
    evidence = _create_evidence(
        publish_root,
        config,
        pair,
        runtime,
        result,
        input_sha256,
        initial,
        final,
        raw_index,
    )
    _write(
        publish_root / "docs/evidence/step-6/benchmark-runtime.json",
        _canonical(evidence),
    )
    _write(
        LATEST_CANDIDATE_PATH,
        _canonical(
            {
                "schema_version": 1,
                "run_id": run_id,
                "publish_path": f".cache/benchmarking/run-{run_id}/publish",
            }
        ),
    )
    print(
        "[OK] Passing benchmark candidate staged: "
        "latency="
        f"{evidence['acceptance']['latency']['median_paired_improvement_pct']:.2f}% "
        "throughput="
        f"{evidence['acceptance']['throughput']['median_paired_improvement_pct']:.2f}%"
    )


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish() -> None:
    pointer = _json(LATEST_CANDIDATE_PATH)
    publish_root = REPOSITORY_ROOT / pointer["publish_path"]
    if not publish_root.is_dir() or publish_root.parent.parent != CACHE_ROOT:
        raise BenchmarkError("Candidate publish path is outside .cache/benchmarking")
    from scripts.validate_benchmark_evidence import validate

    candidate_errors = validate(publish_root)
    if candidate_errors:
        raise BenchmarkError("candidate validation failed: " + "; ".join(candidate_errors))
    run_root = publish_root.parent
    backup_root = run_root / "previous"
    targets = [
        "benchmarks/results/raw",
        "benchmarks/results/baseline.csv",
        "benchmarks/results/optimized.csv",
        "benchmarks/results/comparison.csv",
        "benchmarks/report.md",
        "docs/evidence/step-6",
    ]
    moved: list[str] = []
    published: list[str] = []
    try:
        for relative in targets:
            target = REPOSITORY_ROOT / relative
            backup = backup_root / relative
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                moved.append(relative)
        for relative in targets:
            source = publish_root / relative
            target = REPOSITORY_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            published.append(relative)
        canonical_errors = validate(REPOSITORY_ROOT)
        if canonical_errors:
            raise BenchmarkError(
                "published bundle validation failed: " + "; ".join(canonical_errors)
            )
    except Exception:
        for relative in reversed(published):
            _remove(REPOSITORY_ROOT / relative)
        for relative in reversed(moved):
            backup = backup_root / relative
            target = REPOSITORY_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(target)
        raise
    print(f"[OK] Published passing benchmark bundle from {pointer['run_id']}.")


def orchestrate(env_file: Path) -> None:
    """Run the SDK container while Windows owns process-attributed GPU telemetry."""
    if os.name != "nt":
        raise BenchmarkError("the Step 6 host orchestrator requires Windows")
    if not env_file.is_file():
        raise BenchmarkError(f"Compose env file does not exist: {env_file}")
    sample = collect_nvidia_sample(1)
    if sample.get("collection_ok") is not True:
        raise BenchmarkError(f"nvidia-smi host observer preflight failed: {sample.get('error')}")
    config = _json(CONFIG_PATH)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = CACHE_ROOT / f"run-{run_id}"
    run_root.mkdir(parents=True, exist_ok=False)
    observer = HostObserver(run_root, config["environment_guard"])
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(REPOSITORY_ROOT),
        "--file",
        str(REPOSITORY_ROOT / "docker-compose.yml"),
        "--env-file",
        str(env_file.resolve()),
        "--profile",
        "benchmark",
        "run",
        "--rm",
        "-e",
        f"BENCHMARK_RUN_ID={run_id}",
        "benchmark-runner",
    ]
    observer.start()
    try:
        process = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    finally:
        observer.force_stop()
        observer.join(timeout=30.0)
    if process.returncode != 0:
        raise BenchmarkError(
            f"benchmark SDK container failed with exit code {process.returncode}; "
            f"diagnostics retained at {run_root.name}"
        )
    publish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "prepare-cache", "measure", "publish"),
        default="run",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env") if Path(".env").is_file() else Path(".env.example"),
        help="Compose environment file used by the Windows host orchestrator",
    )
    validation_modes = parser.add_mutually_exclusive_group()
    validation_modes.add_argument(
        "--check",
        action="store_true",
        help="validate the currently published evidence without running Triton",
    )
    validation_modes.add_argument(
        "--historical-only",
        action="store_true",
        help="validate only historical evidence integrity without running Triton",
    )
    args = parser.parse_args()
    try:
        if args.check or args.historical_only:
            from scripts.validate_benchmark_evidence import validate

            errors = validate(
                REPOSITORY_ROOT, historical_only=args.historical_only
            )
            if errors:
                raise BenchmarkError("; ".join(errors))
            return 0
        elif args.command == "prepare-cache":
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            return 0
        if args.command == "run":
            orchestrate(args.env_file)
        elif args.command == "publish":
            publish()
        else:
            measure()
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, AggregationError, BenchmarkError) as error:
        print(f"[FAIL] Benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
