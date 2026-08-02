#!/usr/bin/env python3
"""Independently recompute and validate the complete Step 6 evidence bundle."""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.aggregate_results import AggregationError, aggregate, render_report
from benchmarks.environment_guard import GuardError, read_jsonl, recompute_guard
from benchmarks.run_benchmark import (
    CONFIG_PATH,
    CLIENT_REQUEST_COUNT,
    PAIR_CONTRACT_PATH,
    STATISTIC_COUNT_FIELDS,
    STATISTIC_DURATION_FIELDS,
    STABILITY_PASS,
    sha256,
    source_fingerprint,
)

SCHEMA_PATH = REPOSITORY_ROOT / "schemas/benchmark-evidence.schema.json"
TELEMETRY_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/benchmark-host-telemetry.schema.json"
EVIDENCE_RELATIVE = Path("docs/evidence/step-6/benchmark-runtime.json")
INDEX_RELATIVE = Path("benchmarks/results/raw/index.json")
WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)")
POSIX_HOST_PATH = re.compile(r"(?<![\w:/])/(?:home|mnt|users|tmp|var)/")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact path escapes evidence root: {relative}")
    return path


def _artifact_errors(root: Path, path: Path, label: str) -> list[str]:
    if not path.is_file():
        return [f"missing benchmark artifact: {label}"]
    errors: list[str] = []
    data = path.read_bytes()
    if b"\r\n" in data:
        errors.append(f"benchmark artifact is not LF-only: {label}")
    content = data.decode("utf-8", errors="replace")
    if WINDOWS_PATH.search(content) or POSIX_HOST_PATH.search(content):
        errors.append(f"host-specific path leaked into {label}")
    lowered = content.lower()
    if any(token in lowered for token in ("api_key=", "password=", "authorization:")):
        errors.append(f"secret-like value leaked into {label}")
    return errors


def _telemetry_device_source_errors(
    telemetry: list[dict[str, Any]],
) -> list[str]:
    """Ensure event-driven samples reuse, and identify, real periodic device data."""
    errors: list[str] = []
    samples_by_sequence = {sample.get("sequence"): sample for sample in telemetry}
    for sample_index, sample in enumerate(telemetry, 1):
        source_sequence = sample.get("device_metrics_source_sequence")
        if sample.get("sample_kind") == "periodic":
            if source_sequence != sample.get("sequence"):
                errors.append(
                    f"periodic telemetry sample {sample_index} has stale device source"
                )
        elif sample.get("sample_kind") == "boundary":
            source = samples_by_sequence.get(source_sequence)
            if (
                not isinstance(source_sequence, int)
                or source_sequence >= sample.get("sequence", 0)
                or not isinstance(source, dict)
                or source.get("sample_kind") != "periodic"
                or sample.get("gpu") != source.get("gpu")
                or sample.get("nvidia_processes") != source.get("nvidia_processes")
            ):
                errors.append(
                    f"boundary telemetry sample {sample_index} has invalid device source"
                )
    return errors


def _expected_runs(config: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    return [
        (scenario["id"], repetition, position, role)
        for scenario in config["scenarios"]
        for repetition, order in enumerate(config["execution_order"], 1)
        for position, role in enumerate(order, 1)
    ]


def _slot_numbering_errors(
    runs: list[dict[str, Any]],
    contaminated_runs: list[dict[str, Any]],
    maximum: int,
) -> list[str]:
    """Prove replacements are consecutive and only replace an attributed attempt."""
    errors: list[str] = []
    if len(runs) != 16:
        errors.append(f"valid formal run count is {len(runs)}, expected 16")
    formal_slots = {run.get("slot_id"): run for run in runs}
    if len(formal_slots) != 16:
        errors.append("formal valid slots are not unique")
    if any(run.get("classification") != "VALID" for run in runs):
        errors.append("contaminated or invalid trial entered formal aggregates")
    contamination_slots = {run.get("slot_id") for run in contaminated_runs}
    if not contamination_slots <= set(formal_slots):
        errors.append("contaminated run does not belong to a published formal slot")
    if any(
        run.get("scenario") not in {"latency", "throughput"}
        for run in contaminated_runs
    ):
        errors.append("contaminated replacement has a non-formal scenario")
    for slot_id, valid in formal_slots.items():
        contaminated = sorted(
            (run for run in contaminated_runs if run.get("slot_id") == slot_id),
            key=lambda item: item.get("slot_attempt", 0),
        )
        attempts = [run.get("slot_attempt") for run in contaminated]
        if any(
            (run.get("scenario"), run.get("role"))
            != (valid.get("scenario"), valid.get("role"))
            for run in contaminated
        ):
            errors.append(f"contaminated replacement changes formal slot identity: {slot_id}")
        if attempts != list(range(1, len(contaminated) + 1)):
            errors.append(f"contaminated replacement numbering is invalid: {slot_id}")
        if valid.get("slot_attempt") != len(contaminated) + 1:
            errors.append(f"valid run skips or cherry-picks slot attempts: {slot_id}")
        if valid.get("slot_attempt", maximum + 1) > maximum:
            errors.append(f"valid run exceeds contaminated attempt limit: {slot_id}")
    return errors


def _pass_diagnostic_errors(
    run: dict[str, Any],
    attempts: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    model: str,
    version: str,
    warmup_request_count: int,
    request_count_per_window: int = 0,
) -> list[str]:
    """Recompute pass boundaries, server deltas, and minimum request coverage."""
    errors: list[str] = []
    diagnostics = run.get("pass_diagnostics")
    label = run.get("sidecar_path")
    if not isinstance(diagnostics, list) or len(diagnostics) != len(attempts):
        return [f"pass diagnostics do not cover every PA pass: {label}"]
    samples_by_sequence = {
        int(sample["sequence"]): sample
        for sample in telemetry
        if isinstance(sample.get("sequence"), int)
    }
    trial_boundary = run.get("guard_boundary", {})
    expected_start = trial_boundary.get("guard_start_seq")
    previous_after: dict[str, Any] | None = None
    for expected_attempt, (attempt, diagnostic) in enumerate(
        zip(attempts, diagnostics, strict=True), 1
    ):
        if not isinstance(diagnostic, dict):
            errors.append(f"pass diagnostic is not an object: {label}")
            continue
        if {
            "attempt": diagnostic.get("attempt"),
            "infer_per_sec": diagnostic.get("infer_per_sec"),
            "p95_latency_us": diagnostic.get("p95_latency_us"),
        } != attempt:
            errors.append(f"pass diagnostic differs from PA output: {label}")
        expected_warmup = expected_attempt == 1 and warmup_request_count > 0
        if diagnostic.get("includes_initial_warmup") is not expected_warmup:
            errors.append(f"pass diagnostic warmup marker is stale: {label}")
        boundary = diagnostic.get("guard_boundary", {})
        start_sequence = boundary.get("guard_start_seq")
        end_sequence = boundary.get("guard_end_seq")
        if (
            not isinstance(start_sequence, int)
            or not isinstance(end_sequence, int)
            or start_sequence != expected_start
            or end_sequence <= start_sequence
            or end_sequence > trial_boundary.get("guard_end_seq", -1)
        ):
            errors.append(f"pass guard sequence boundary is invalid: {label}")
        else:
            for prefix, sequence in (("started", start_sequence), ("ended", end_sequence)):
                sample = samples_by_sequence.get(sequence)
                if sample is None:
                    errors.append(f"pass guard sequence has no telemetry sample: {label}")
                    continue
                if boundary.get(f"guard_{prefix}_at_utc") != sample.get(
                    "observed_at_utc"
                ) or boundary.get(f"guard_{prefix}_monotonic_ns") != sample.get(
                    "host_monotonic_ns"
                ):
                    errors.append(f"pass guard timestamp does not match telemetry: {label}")
            expected_start = end_sequence
        statistics = diagnostic.get("triton_statistics", {})
        before = statistics.get("before")
        after = statistics.get("after")
        delta = statistics.get("delta")
        if not all(isinstance(item, dict) for item in (before, after, delta)):
            errors.append(f"pass Triton statistics are incomplete: {label}")
            continue
        if previous_after is not None and before != previous_after:
            errors.append(f"pass Triton snapshots are not contiguous: {label}")
        previous_after = after
        if any(
            snapshot.get("model") != model or str(snapshot.get("version")) != version
            for snapshot in (before, after)
        ):
            errors.append(f"pass Triton snapshot identifies another model/version: {label}")
        recomputed: dict[str, int] = {}
        try:
            for field in (*STATISTIC_COUNT_FIELDS, *STATISTIC_DURATION_FIELDS):
                before_value = before[field]
                after_value = after[field]
                if (
                    not isinstance(before_value, int)
                    or isinstance(before_value, bool)
                    or not isinstance(after_value, int)
                    or isinstance(after_value, bool)
                    or before_value < 0
                    or after_value < before_value
                ):
                    raise ValueError(field)
                recomputed[field] = after_value - before_value
        except (KeyError, ValueError, TypeError):
            errors.append(f"pass Triton cumulative counters are invalid: {label}")
            continue
        if recomputed["request_count"] <= 0 or recomputed["execution_count"] <= 0:
            errors.append(f"pass Triton counter delta is empty: {label}")
            continue
        per_request = {
            field.removesuffix("_duration_ns"): round(
                recomputed[field] / recomputed["request_count"] / 1000.0, 6
            )
            for field in STATISTIC_DURATION_FIELDS
        }
        recomputed_value: dict[str, Any] = {**recomputed, "per_request_us": per_request}
        if delta != recomputed_value:
            errors.append(f"pass Triton statistic delta is stale: {label}")
    return errors


def _validate_runs(
    artifact_root: Path,
    index: dict[str, Any],
    config: dict[str, Any],
    pair: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    runs = index.get("runs", [])
    contaminated_runs = index.get("contaminated_runs", [])
    if not isinstance(runs, list) or not isinstance(contaminated_runs, list):
        return ["raw index run collections are invalid"]
    guard_config = config["environment_guard"]
    guard_index = index.get("environment_guard", {})
    telemetry_relative = guard_index.get("telemetry_path", "")
    telemetry_path = _safe_artifact(artifact_root, telemetry_relative)
    errors.extend(_artifact_errors(artifact_root, telemetry_path, telemetry_relative))
    telemetry: list[dict[str, Any]] = []
    if telemetry_path.is_file():
        if guard_index.get("telemetry_sha256") != sha256(telemetry_path):
            errors.append("environment telemetry hash is stale")
        try:
            telemetry = read_jsonl(telemetry_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, GuardError) as error:
            errors.append(f"environment telemetry is invalid: {error}")
        if guard_index.get("sample_count") != len(telemetry):
            errors.append("environment telemetry sample count is stale")
        telemetry_schema = _json(TELEMETRY_SCHEMA_PATH)
        Draft202012Validator.check_schema(telemetry_schema)
        validator = Draft202012Validator(
            telemetry_schema, format_checker=FormatChecker()
        )
        for sample_index, sample in enumerate(telemetry, 1):
            for schema_error in validator.iter_errors(sample):
                errors.append(
                    f"environment telemetry sample {sample_index} is invalid: "
                    f"{schema_error.message}"
                )
                break
        errors.extend(_telemetry_device_source_errors(telemetry))
    actions_relative = guard_index.get("actions_path", "")
    actions_path = _safe_artifact(artifact_root, actions_relative)
    errors.extend(_artifact_errors(artifact_root, actions_path, actions_relative))
    if actions_path.is_file() and guard_index.get("actions_sha256") != sha256(actions_path):
        errors.append("environment actions hash is stale")
    clock_guard_source = REPOSITORY_ROOT / "benchmarks/clock_guard.c"
    if index.get("clock_guard_source_sha256") != sha256(clock_guard_source):
        errors.append("raw index clock guard source hash is stale")
    if not re.fullmatch(r"[0-9a-f]{64}", str(index.get("clock_guard_sha256", ""))):
        errors.append("raw index compiled clock guard hash is invalid")
    actual_keys = [
        (
            run.get("scenario"),
            run.get("repetition"),
            run.get("order_position"),
            run.get("role"),
        )
        for run in runs
    ]
    if actual_keys != _expected_runs(config):
        errors.append("raw run order differs from AB -> BA -> AB -> BA")
    if len(runs) != 16:
        errors.append(f"raw index has {len(runs)} formal runs, expected 16")
    scenario_config = {item["id"]: item for item in config["scenarios"]}
    seen_paths: set[str] = set()
    for run in [*runs, *contaminated_runs]:
        role = run.get("role")
        scenario_id = run.get("scenario")
        if role not in {"baseline", "optimized"} or scenario_id not in scenario_config:
            errors.append("raw index contains an unknown role or scenario")
            continue
        scenario = scenario_config[scenario_id]
        is_contaminated = run in contaminated_runs
        expected_classification = "CONTAMINATED" if is_contaminated else "VALID"
        expected_run_fields = {
            "scenario_status": "formal",
            "perf_analyzer_completion_tolerance_pct": scenario[
                "perf_analyzer_completion_tolerance_pct"
            ],
            "measurement_completed": True,
            "errors": 0,
            "classification": expected_classification,
        }
        for field, expected in expected_run_fields.items():
            if run.get(field) != expected:
                errors.append(f"raw run has stale {field}: {run.get('sidecar_path')}")
        for field in ("csv", "log", "sidecar"):
            relative = run.get(f"{field}_path", "")
            if not isinstance(relative, str) or not relative or relative in seen_paths:
                errors.append(f"raw index has an invalid or duplicate {field} path")
                continue
            seen_paths.add(relative)
            path = _safe_artifact(artifact_root, relative)
            errors.extend(_artifact_errors(artifact_root, path, relative))
            if path.is_file() and run.get(f"{field}_sha256") != sha256(path):
                errors.append(f"raw {field} hash is stale: {relative}")
        sidecar_path = _safe_artifact(artifact_root, run.get("sidecar_path", ""))
        if not sidecar_path.is_file():
            continue
        sidecar = _json(sidecar_path)
        expected_fields = {
            "schema_version": 1,
            "trial_id": run.get("trial_id"),
            "slot_id": run.get("slot_id"),
            "slot_attempt": run.get("slot_attempt"),
            "classification": expected_classification,
            "guard_classification": run.get("guard_classification"),
            "guard_reasons": run.get("guard_reasons"),
            "guard_baseline_processes": run.get("guard_baseline_processes"),
            "guard_baseline_active_processes": run.get("guard_baseline_active_processes"),
            "guard_baseline_engines": run.get("guard_baseline_engines"),
            "guard_boundary": run.get("guard_boundary"),
            "scenario": scenario_id,
            "scenario_status": "formal",
            "perf_analyzer_completion_tolerance_pct": scenario[
                "perf_analyzer_completion_tolerance_pct"
            ],
            "repetition": run.get("repetition"),
            "order_position": run.get("order_position"),
            "role": role,
            "model": pair[role]["model"],
            "version": pair[role]["version"],
            "batch_size": scenario["batch_size"],
            "concurrency": scenario["concurrency"],
            "protocol": config["protocol"],
            "input_sha256": index.get("input_sha256"),
            "clock_guard_sha256": index.get("clock_guard_sha256"),
            "measurement_completed": True,
            "pa_reported_stable": run.get("pa_reported_stable"),
            "errors": 0,
            "csv_path": run.get("csv_path"),
            "log_path": run.get("log_path"),
            "pass_diagnostics": run.get("pass_diagnostics"),
            "client_request_count": run.get("client_request_count"),
        }
        for field, expected in expected_fields.items():
            if sidecar.get(field) != expected:
                errors.append(f"sidecar {run.get('sidecar_path')} has stale {field}")
        windows = run.get("measurement_windows", [])
        expected_numbers = list(range(1, len(windows) + 1))
        if (
            not isinstance(windows, list)
            or len(windows) != config["measurement"]["max_measurement_windows"]
            or run.get("measurement_windows_used") != len(windows)
            or [item.get("attempt") for item in windows] != expected_numbers
        ):
            errors.append(f"raw measurement windows are incomplete: {run.get('sidecar_path')}")
        else:
            for window in windows:
                if (
                    not isinstance(window.get("infer_per_sec"), (int, float))
                    or window["infer_per_sec"] <= 0
                    or not isinstance(window.get("p95_latency_us"), int)
                    or window["p95_latency_us"] <= 0
                ):
                    errors.append(f"raw measurement window is invalid: {run.get('sidecar_path')}")
                    break
        if sidecar.get("measurement_windows_used") != len(windows) or sidecar.get(
            "measurement_windows"
        ) != windows:
            errors.append(f"sidecar windows differ from raw index: {run.get('sidecar_path')}")
        minimum_client_requests = (
            len(windows) * int(config["measurement"]["request_count_per_window"])
        )
        if (
            not isinstance(run.get("client_request_count"), int)
            or run["client_request_count"] < minimum_client_requests
        ):
            errors.append(f"formal run served too few PA client requests: {run.get('sidecar_path')}")
        if isinstance(windows, list):
            errors.extend(
                _pass_diagnostic_errors(
                    run,
                    windows,
                    telemetry,
                    pair[role]["model"],
                    pair[role]["version"],
                    int(config["measurement"]["warmup_request_count"]),
                    int(config["measurement"]["request_count_per_window"]),
                )
            )
        log_path = _safe_artifact(artifact_root, run.get("log_path", ""))
        if log_path.is_file():
            log_content = log_path.read_text(encoding="utf-8")
            logged_windows = [
                {
                    "attempt": int(match.group("attempt")),
                    "infer_per_sec": float(match.group("throughput")),
                    "p95_latency_us": int(match.group("p95")),
                }
                for match in STABILITY_PASS.finditer(log_content)
            ]
            if logged_windows != windows:
                errors.append(f"raw index omits or changes PA windows: {run.get('log_path')}")
            logged_request_counts = [
                int(match.group("count"))
                for match in CLIENT_REQUEST_COUNT.finditer(log_content)
            ]
            logged_request_count = logged_request_counts[-1] if logged_request_counts else 0
            if run.get("client_request_count") != logged_request_count:
                errors.append(f"PA client request count is stale: {run.get('log_path')}")
            expected_pa_stable = "Failed to obtain stable measurement" not in log_content
            if run.get("pa_reported_stable") is not expected_pa_stable:
                errors.append(f"PA status is stale: {run.get('log_path')}")
            if "cb::Error:" in log_content or "command timed out" in log_content:
                errors.append(f"runtime error is present in benchmark log: {run.get('log_path')}")
        command_text = " ".join(str(item) for item in sidecar.get("command", []))
        required_fragments = (
            f"-m {pair[role]['model']}",
            f"-x {pair[role]['version']}",
            f"-b {scenario['batch_size']}",
            f"--concurrency-range {scenario['concurrency']}",
            "--measurement-mode count_windows",
            f"--measurement-request-count {config['measurement']['request_count_per_window']}",
            f"--warmup-request-count {config['measurement']['warmup_request_count']}",
            "--stability-percentage 999",
            f"--percentile {config['measurement']['reporting_percentile']}",
            f"--max-trials {config['measurement']['max_measurement_windows']}",
            "--verbose-csv",
        )
        if not all(fragment in command_text for fragment in required_fragments):
            errors.append(f"sidecar command is incomplete: {run.get('sidecar_path')}")
        if "--collect-metrics" not in command_text:
            errors.append(f"sidecar command does not collect metrics: {run.get('sidecar_path')}")
        if "<run>/" not in command_text:
            errors.append(f"sidecar command paths are not sanitized: {run.get('sidecar_path')}")
        if run.get("input_sha256") != index.get("input_sha256"):
            errors.append("raw runs do not share one deterministic input")
        if telemetry:
            try:
                recomputed_guard = recompute_guard(
                    telemetry, run.get("guard_boundary", {}), guard_config
                )
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"guard boundary is invalid: {run.get('sidecar_path')}: {error}")
            else:
                if run.get("guard_classification") != recomputed_guard["classification"]:
                    errors.append(f"runner guard classification is stale: {run.get('sidecar_path')}")
                if run.get("guard_reasons") != recomputed_guard.get("reasons", []):
                    errors.append(f"runner guard reasons are stale: {run.get('sidecar_path')}")
                for field in (
                    "baseline_processes",
                    "baseline_active_processes",
                    "baseline_engines",
                ):
                    if run.get(f"guard_{field}") != recomputed_guard.get(field, []):
                        errors.append(f"runner guard {field} is stale: {run.get('sidecar_path')}")
                if is_contaminated and recomputed_guard["classification"] != "CONTAMINATED":
                    errors.append(f"contaminated replacement lacks attribution: {run.get('sidecar_path')}")
                if not is_contaminated and recomputed_guard["classification"] != "CLEAN":
                    errors.append(f"valid formal run is not clean: {run.get('sidecar_path')}")
    errors.extend(
        _slot_numbering_errors(
            runs,
            contaminated_runs,
            int(guard_config["max_contaminated_attempts_per_slot"]),
        )
    )
    return errors


def _evidence_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
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
                    "mean_client_latency_ms": item["baseline_metrics"]["avg_latency_ms"],
                    "p50_latency_ms": item["baseline_metrics"]["p50_latency_ms"],
                    "p95_latency_ms": item["baseline_metrics"]["p95_latency_ms"],
                    "infer_per_sec": item["baseline_metrics"]["infer_per_sec"],
                },
                "optimized": {
                    "mean_client_latency_ms": item["optimized_metrics"]["avg_latency_ms"],
                    "p50_latency_ms": item["optimized_metrics"]["p50_latency_ms"],
                    "p95_latency_ms": item["optimized_metrics"]["p95_latency_ms"],
                    "infer_per_sec": item["optimized_metrics"]["infer_per_sec"],
                },
                "paired_improvement_pct": item["paired_improvement_pct"],
                "directional_improvement": item["directional_improvement"],
            }
            for item in comparison["pairs"]
        ],
    }


def _recompute(
    artifact_root: Path,
    index: dict[str, Any],
    config: dict[str, Any],
    pair: dict[str, Any],
    evidence: dict[str, Any],
    errors: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        shutil.copytree(
            artifact_root / "benchmarks/results/raw",
            temporary_root / "benchmarks/results/raw",
        )
        result = aggregate(temporary_root, index, config)
        for name in ("baseline.csv", "optimized.csv", "comparison.csv"):
            expected = temporary_root / "benchmarks/results" / name
            actual = artifact_root / "benchmarks/results" / name
            if not actual.is_file() or actual.read_bytes() != expected.read_bytes():
                errors.append(f"{name} differs from raw-data recomputation")
        expected_report = render_report(
            config,
            pair,
            evidence.get("runtime", {}),
            result,
            index.get("input_sha256", ""),
        )
        report_path = artifact_root / "benchmarks/report.md"
        if not report_path.is_file() or report_path.read_text(encoding="utf-8") != expected_report:
            errors.append("benchmark report differs from raw-data recomputation")
        measurement = evidence.get("measurement", {})
        if measurement.get("scenarios") != result["scenario_diagnostics"]:
            errors.append("evidence scenario diagnostics are stale")
        if measurement.get("valid_run_count") != 16 or measurement.get("errors") != 0:
            errors.append("benchmark evidence contains invalid or errored formal runs")
        acceptance = evidence.get("acceptance", {})
        if acceptance.get("rule") != config["acceptance"]:
            errors.append("evidence acceptance rule is stale")
        for scenario_id in ("latency", "throughput"):
            expected = _evidence_comparison(result["comparisons"][scenario_id])
            if acceptance.get(scenario_id) != expected:
                errors.append(f"evidence {scenario_id} paired comparison is stale")
        if acceptance.get("passed") is not result["acceptance_passed"]:
            errors.append("evidence final acceptance is stale")
        if not result["acceptance_passed"]:
            errors.append("recomputed paired benchmark acceptance does not pass")


def validate(artifact_root: Path = REPOSITORY_ROOT) -> list[str]:
    errors: list[str] = []
    evidence_path = artifact_root / EVIDENCE_RELATIVE
    index_path = artifact_root / INDEX_RELATIVE
    if not evidence_path.is_file():
        return ["missing step 6 benchmark evidence"]
    if not index_path.is_file():
        return ["missing benchmark raw index"]
    evidence = _json(evidence_path)
    index = _json(index_path)
    config = _json(CONFIG_PATH)
    pair = _json(PAIR_CONTRACT_PATH)
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors.extend(
        f"evidence.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(evidence),
            key=lambda item: list(item.path),
        )
    )
    freshness = {
        "benchmark_config_sha256": sha256(CONFIG_PATH),
        "source_fingerprint_sha256": source_fingerprint(),
        "pair_contract_sha256": sha256(PAIR_CONTRACT_PATH),
        "input_sha256": index.get("input_sha256"),
    }
    for field, expected in freshness.items():
        if evidence.get(field) != expected:
            errors.append(f"{field} is stale")
    expected_pair = {
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
    }
    if evidence.get("pair") != expected_pair:
        errors.append("evidence model pair is stale")
    if evidence.get("declared_build_target") != pair.get("declared_build_target"):
        errors.append("declared build target is stale")
    if evidence.get("measurement", {}).get("clock_guard_source_sha256") != sha256(
        REPOSITORY_ROOT / "benchmarks/clock_guard.c"
    ):
        errors.append("evidence clock guard source hash is stale")
    if evidence.get("initial_ready") != evidence.get("final_ready"):
        errors.append("initial and final READY states differ")
    if evidence.get("runtime_state_restored") is not True:
        errors.append("runtime state was not restored")
    for scenario in evidence.get("measurement", {}).get("scenarios", {}).values():
        if scenario.get("pa_reported_stable_run_count", 0) + scenario.get(
            "pa_reported_unstable_run_count", 0
        ) != scenario.get("run_count"):
            errors.append("PA status counts are inconsistent")
    guard_evidence = evidence.get("environment_guard", {})
    guard_index = index.get("environment_guard", {})
    expected_guard_scalars = {
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
        "forbidden_processes": config["environment_guard"]["forbidden_processes"],
        "max_contaminated_attempts_per_slot": config["environment_guard"][
            "max_contaminated_attempts_per_slot"
        ],
        "formal_slot_count": 16,
        "valid_run_count": 16,
        "contaminated_run_count": len(index.get("contaminated_runs", [])),
    }
    for field, expected in expected_guard_scalars.items():
        if guard_evidence.get(field) != expected:
            errors.append(f"environment guard evidence has stale {field}")
    expected_guard_artifacts = {
        "telemetry": (
            guard_index.get("telemetry_path"),
            guard_index.get("telemetry_sha256"),
        ),
        "actions": (
            guard_index.get("actions_path"),
            guard_index.get("actions_sha256"),
        ),
    }
    for label, (path, digest) in expected_guard_artifacts.items():
        if guard_evidence.get(label) != {"path": path, "sha256": digest}:
            errors.append(f"environment guard {label} reference is stale")
    expected_contamination_records = [
        {
            "slot_id": run.get("slot_id"),
            "slot_attempt": run.get("slot_attempt"),
            "scenario": run.get("scenario"),
            "role": run.get("role"),
            "guard_boundary": run.get("guard_boundary"),
            "reasons": run.get("guard_reasons"),
        }
        for run in index.get("contaminated_runs", [])
    ]
    if guard_evidence.get("contamination_records") != expected_contamination_records:
        errors.append("environment guard contamination records are stale")
    runtime_contract_path = index.get("runtime_contract_path", "")
    runtime_contract = _safe_artifact(artifact_root, runtime_contract_path)
    errors.extend(_artifact_errors(artifact_root, runtime_contract, runtime_contract_path))
    if runtime_contract.is_file():
        if index.get("runtime_contract_sha256") != sha256(runtime_contract):
            errors.append("runtime contract hash is stale")
        snapshot = _json(runtime_contract)
        if snapshot.get("baseline", {}).get("runtime_contract") != snapshot.get(
            "optimized", {}
        ).get("runtime_contract"):
            errors.append("live ONNX and TensorRT contracts differ")
        if snapshot.get("common_contract_sha256") != pair.get("common_contract_sha256"):
            errors.append("live contract snapshot is stale")
    errors.extend(_validate_runs(artifact_root, index, config, pair))
    expected_artifacts = {
        "raw_index": "benchmarks/results/raw/index.json",
        "baseline": "benchmarks/results/baseline.csv",
        "optimized": "benchmarks/results/optimized.csv",
        "comparison": "benchmarks/results/comparison.csv",
        "report": "benchmarks/report.md",
    }
    for label, relative in expected_artifacts.items():
        item = evidence.get("artifacts", {}).get(label, {})
        path = _safe_artifact(artifact_root, relative)
        errors.extend(_artifact_errors(artifact_root, path, relative))
        if item.get("path") != relative or (
            path.is_file() and item.get("sha256") != sha256(path)
        ):
            errors.append(f"evidence artifact reference is stale: {label}")
    if not errors:
        _recompute(artifact_root, index, config, pair, evidence, errors)
    return errors


def main() -> int:
    try:
        errors = validate()
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        AggregationError,
    ) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(
            f"[FAIL] Benchmark evidence validation found {len(errors)} issue(s).",
            file=sys.stderr,
        )
        return 1
    print("[OK] Step 6 evidence recomputes from all four paired repetitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
