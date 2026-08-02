#!/usr/bin/env python3
"""Validate the step 6 benchmark contract without contacting Triton."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.run_benchmark import build_perf_analyzer_command
from scripts.model_preparation.prepare_models import (
    MANIFEST_PATH,
    canonical_json,
    render_benchmark_pair_contract,
)

CONFIG_PATH = REPOSITORY_ROOT / "benchmarks/configs/benchmark.json"
CONFIG_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/benchmark-config.schema.json"
PAIR_PATH = REPOSITORY_ROOT / "shared/benchmark-model-pair.json"
PAIR_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/benchmark-model-pair.schema.json"
TELEMETRY_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/benchmark-host-telemetry.schema.json"
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
BENCHMARK_SOURCES = (
    REPOSITORY_ROOT / "benchmarks/run_benchmark.py",
    REPOSITORY_ROOT / "benchmarks/aggregate_results.py",
    REPOSITORY_ROOT / "benchmarks/clock_guard.c",
    REPOSITORY_ROOT / "benchmarks/environment_guard.py",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _schema_errors(value: Any, schema_path: Path, label: str) -> list[str]:
    schema = _json(schema_path)
    Draft202012Validator.check_schema(schema)
    return [
        f"{label}.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda item: list(item.path),
        )
    ]


def _flag_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _validate_command(config: dict[str, Any], pair: dict[str, Any], errors: list[str]) -> None:
    scenario = config["scenarios"][0]
    with tempfile.TemporaryDirectory() as directory:
        command = build_perf_analyzer_command(
            config,
            pair,
            "baseline",
            scenario,
            Path(directory) / "input",
            Path(directory) / "raw.csv",
        )
    measurement = config["measurement"]
    expected = {
        "--service-kind": config["service_kind"],
        "-m": pair["baseline"]["model"],
        "-x": pair["baseline"]["version"],
        "-u": config["endpoints"]["http"],
        "-i": config["protocol"],
        "-b": str(scenario["batch_size"]),
        "--concurrency-range": str(scenario["concurrency"]),
        "--measurement-mode": measurement["mode"],
        "--measurement-request-count": str(measurement["request_count_per_window"]),
        "--warmup-request-count": str(measurement["warmup_request_count"]),
        "--stability-percentage": str(
            scenario["perf_analyzer_completion_tolerance_pct"]
        ),
        "--percentile": str(measurement["reporting_percentile"]),
        "--max-trials": str(measurement["max_measurement_windows"]),
        "--metrics-url": config["endpoints"]["metrics"],
        "--metrics-interval": str(measurement["metrics_interval_ms"]),
    }
    for flag, value in expected.items():
        if _flag_value(command, flag) != value:
            errors.append(f"Perf Analyzer command does not source {flag} from benchmark config")
    for flag in ("--collect-metrics", "--verbose-csv"):
        if flag not in command:
            errors.append(f"Perf Analyzer command is missing {flag}")


def _validate_compose(errors: list[str]) -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose.get("services", {}).get("benchmark-runner", {})
    if not service:
        errors.append("Compose is missing benchmark-runner")
        return
    if service.get("image") != "${TRITON_SDK_IMAGE:?TRITON_SDK_IMAGE is required}":
        errors.append("benchmark-runner must use the pinned TRITON_SDK_IMAGE")
    if service.get("profiles") != ["benchmark"]:
        errors.append("benchmark-runner must be isolated behind the benchmark profile")
    if service.get("command") != ["python", "benchmarks/run_benchmark.py", "measure"]:
        errors.append("benchmark-runner has an unexpected command")
    if service.get("working_dir") != "/workspace" or service.get("restart") != "no":
        errors.append("benchmark-runner lifecycle contract is invalid")
    if service.get("ports"):
        errors.append("benchmark-runner must not publish ports")
    if "backend" not in service.get("networks", []):
        errors.append("benchmark-runner must join the backend network")
    mounts = {
        item.get("target"): item
        for item in service.get("volumes", [])
        if isinstance(item, dict)
    }
    repository_mount = mounts.get("/workspace", {})
    cache_mount = mounts.get("/workspace/.cache/benchmarking", {})
    if repository_mount.get("source") != "." or repository_mount.get("read_only") is not True:
        errors.append("benchmark-runner repository mount must be read-only")
    if cache_mount.get("source") != "./.cache/benchmarking" or cache_mount.get("read_only") is True:
        errors.append("benchmark-runner may write only to .cache/benchmarking")
    if set(service.get("environment", {})) != {"TRITON_IMAGE", "TRITON_SDK_IMAGE"}:
        errors.append("benchmark-runner image identity environment is incomplete")


def validate() -> list[str]:
    errors: list[str] = []
    config = _json(CONFIG_PATH)
    pair = _json(PAIR_PATH)
    errors.extend(_schema_errors(config, CONFIG_SCHEMA_PATH, "config"))
    errors.extend(_schema_errors(pair, PAIR_SCHEMA_PATH, "pair"))
    Draft202012Validator.check_schema(_json(TELEMETRY_SCHEMA_PATH))
    manifest = _json(MANIFEST_PATH)
    expected_pair = render_benchmark_pair_contract(manifest)
    if canonical_json(pair) != canonical_json(expected_pair):
        errors.append("shared benchmark model pair is stale")
    if pair.get("baseline", {}).get("model") == pair.get("optimized", {}).get("model"):
        errors.append("benchmark roles must reference different models")
    if pair.get("baseline", {}).get("io_precision") != pair.get("optimized", {}).get(
        "io_precision"
    ):
        errors.append("benchmark pair must use equal I/O precision")
    if pair.get("parity", {}).get("status") != "passed":
        errors.append("benchmark pair requires a passed numerical parity result")
    scenarios = {item["id"]: item for item in config.get("scenarios", [])}
    if set(scenarios) != {"latency", "throughput"}:
        errors.append("benchmark scenarios must be exactly latency and throughput")
    expected_scenarios = {
        "latency": (1, 1, "mean_client_latency_ms", "formal", 999),
        "throughput": (8, 4, "infer_per_sec", "formal", 999),
    }
    for scenario_id, expected in expected_scenarios.items():
        item = scenarios.get(scenario_id, {})
        actual = (
            item.get("batch_size"),
            item.get("concurrency"),
            item.get("primary_metric"),
            item.get("status"),
            item.get("perf_analyzer_completion_tolerance_pct"),
        )
        if actual != expected:
            errors.append(f"{scenario_id} scenario differs from the formal contract")
        if int(item.get("batch_size", 0)) > int(pair["common_contract"]["max_batch_size"]):
            errors.append(f"{scenario_id} batch exceeds the shared model contract")
    measurement = config.get("measurement", {})
    fixed_measurement = {
        "mode": "count_windows",
        "request_count_per_window": 500,
        "warmup_request_count": 100,
        "reporting_percentile": 95,
        "max_measurement_windows": 3,
        "repetitions": 4,
        "cooldown_seconds": 2,
        "metrics_interval_ms": 100,
    }
    for field, expected in fixed_measurement.items():
        if measurement.get(field) != expected:
            errors.append(f"measurement.{field} must remain {expected!r}")
    if config.get("acceptance") != {
        "minimum_median_paired_improvement_pct_exclusive": 0.0,
        "minimum_directional_pairs": 3,
        "strong_improvement_threshold_pct": 5.0,
    }:
        errors.append("benchmark paired acceptance rule differs from the Step 6 contract")
    if config.get("load_generator") != {
        "realtime_clock_guard": True,
        "verbose_windows": True,
    }:
        errors.append("load generator monotonic clock guard and verbose windows must remain enabled")
    guard = config.get("environment_guard", {})
    fixed_guard = {
        "host_observer": "windows_gpu_engine_nvidia_smi",
        "sample_interval_ms": 1000,
        "baseline_seconds": 5,
        "minimum_consecutive_baseline_samples": 5,
        "maximum_sample_gap_ms": 3000,
        "ack_timeout_seconds": 30,
        "max_contaminated_attempts_per_slot": 3,
        "gpu_engine_activity_threshold_percent": 0.1,
    }
    for field, expected in fixed_guard.items():
        if guard.get(field) != expected:
            errors.append(f"environment_guard.{field} must remain {expected!r}")
    if not guard.get("benchmark_owned_processes"):
        errors.append("environment guard requires predeclared benchmark-owned processes")
    if not guard.get("forbidden_processes"):
        errors.append("environment guard requires predeclared forbidden process names")
    overlap = {
        name.lower() for name in guard.get("benchmark_owned_processes", [])
    } & {name.lower() for name in guard.get("forbidden_processes", [])}
    if overlap:
        errors.append("owned and forbidden environment guard processes overlap")
    _validate_command(config, pair, errors)
    _validate_compose(errors)
    for source in BENCHMARK_SOURCES:
        content = source.read_text(encoding="utf-8")
        if (
            'REPOSITORY_ROOT / "models' in content
            or "REPOSITORY_ROOT / 'models" in content
            or "scripts.model_preparation" in content
        ):
            errors.append(f"benchmark implementation bypasses shared contracts: {source.name}")
        if b"\r\n" in source.read_bytes():
            errors.append(f"benchmark source is not LF-only: {source.name}")
    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Benchmark validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    print("[OK] Benchmark configuration, pair, command, and Compose contract are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
