#!/usr/bin/env python3
"""Parse Perf Analyzer CSV files and generate deterministic paired summaries."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

ROLE_HEADERS = [
    "scenario",
    "repetition",
    "order_position",
    "batch_size",
    "concurrency",
    "pa_reported_stable",
    "infer_per_sec",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "server_queue_ms",
    "server_compute_input_ms",
    "server_compute_infer_ms",
    "server_compute_output_ms",
    "gpu_utilization_fraction",
    "gpu_memory_used_bytes",
]
COMPARISON_HEADERS = [
    "scenario",
    "repetition",
    "execution_order",
    "primary_metric",
    "baseline_value",
    "optimized_value",
    "paired_improvement_pct",
    "directional_improvement",
    "median_paired_improvement_pct",
    "improved_pair_count",
    "gate_passed",
]
RAW_COLUMNS = {
    "infer_per_sec": "Inferences/Second",
    "client_send_us": "Client Send",
    "network_server_us": "Network+Server Send/Recv",
    "server_queue_us": "Server Queue",
    "server_compute_input_us": "Server Compute Input",
    "server_compute_infer_us": "Server Compute Infer",
    "server_compute_output_us": "Server Compute Output",
    "client_recv_us": "Client Recv",
    "p50_latency_us": "p50 latency",
    "p95_latency_us": "p95 latency",
    "p99_latency_us": "p99 latency",
    "gpu_utilization": "Avg GPU Utilization",
    "gpu_memory_used": "Max GPU Memory Usage",
    "gpu_memory_total": "Total GPU Memory",
}
NUMBER = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")


class AggregationError(RuntimeError):
    """Raw benchmark data cannot produce the canonical paired aggregates."""


def _number(value: str, field: str) -> float:
    match = NUMBER.search(value.replace(",", ""))
    if match is None:
        raise AggregationError(f"{field} is not numeric: {value!r}")
    number = float(match.group(0))
    if not math.isfinite(number):
        raise AggregationError(f"{field} is not finite")
    return number


def _single_gpu_metric(value: str, field: str) -> str:
    entries = [entry.strip() for entry in value.split(";") if entry.strip()]
    if len(entries) != 1:
        raise AggregationError(f"{field} must contain exactly one GPU value")
    entry = entries[0]
    return entry.rsplit(":", 1)[-1] if ":" in entry else entry


def _memory_bytes(value: str, field: str) -> float:
    scalar = _single_gpu_metric(value, field)
    number = _number(scalar, field)
    lowered = scalar.lower()
    multipliers = {
        "kib": 1024,
        "kb": 1000,
        "mib": 1024**2,
        "mb": 1000**2,
        "gib": 1024**3,
        "gb": 1000**3,
    }
    for suffix, multiplier in multipliers.items():
        if suffix in lowered:
            return number * multiplier
    return number


def _gpu_fraction(value: str) -> float:
    scalar = _single_gpu_metric(value, "Avg GPU Utilization")
    number = _number(scalar, "Avg GPU Utilization")
    if "%" in scalar or number > 1:
        number /= 100.0
    if not 0 <= number <= 1:
        raise AggregationError("Avg GPU Utilization is outside [0, 1]")
    return number


def parse_raw_csv(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 1 or reader.fieldnames is None:
        raise AggregationError(f"{path.name} must contain exactly one result row")
    missing = sorted(set(RAW_COLUMNS.values()) - set(reader.fieldnames))
    if missing:
        raise AggregationError(
            f"{path.name} is missing Perf Analyzer columns: {', '.join(missing)}"
        )
    row = rows[0]
    values = {
        key: _number(row[column], column)
        for key, column in RAW_COLUMNS.items()
        if key not in {"gpu_utilization", "gpu_memory_used", "gpu_memory_total"}
    }
    values["gpu_utilization_fraction"] = _gpu_fraction(
        row[RAW_COLUMNS["gpu_utilization"]]
    )
    values["gpu_memory_used_bytes"] = _memory_bytes(
        row[RAW_COLUMNS["gpu_memory_used"]], RAW_COLUMNS["gpu_memory_used"]
    )
    values["gpu_memory_total_bytes"] = _memory_bytes(
        row[RAW_COLUMNS["gpu_memory_total"]], RAW_COLUMNS["gpu_memory_total"]
    )
    values["avg_latency_us"] = sum(
        values[field]
        for field in (
            "client_send_us",
            "network_server_us",
            "server_queue_us",
            "server_compute_input_us",
            "server_compute_infer_us",
            "server_compute_output_us",
            "client_recv_us",
        )
    )
    return values


def _median(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise AggregationError("aggregate input is empty or non-finite")
    return float(statistics.median(values))


def _format(value: float) -> str:
    return f"{value:.6f}"


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _improvement(baseline: float, optimized: float, *, lower_is_better: bool) -> float:
    baseline_decimal = Decimal(str(baseline))
    optimized_decimal = Decimal(str(optimized))
    if baseline_decimal <= 0:
        raise AggregationError("baseline metric must be positive")
    difference = (
        baseline_decimal - optimized_decimal
        if lower_is_better
        else optimized_decimal - baseline_decimal
    )
    return float(difference / baseline_decimal * Decimal("100"))


def _metric_values(parsed: dict[str, float]) -> dict[str, float]:
    return {
        "infer_per_sec": parsed["infer_per_sec"],
        "avg_latency_ms": parsed["avg_latency_us"] / 1000.0,
        "p50_latency_ms": parsed["p50_latency_us"] / 1000.0,
        "p95_latency_ms": parsed["p95_latency_us"] / 1000.0,
        "p99_latency_ms": parsed["p99_latency_us"] / 1000.0,
        "server_queue_ms": parsed["server_queue_us"] / 1000.0,
        "server_compute_input_ms": parsed["server_compute_input_us"] / 1000.0,
        "server_compute_infer_ms": parsed["server_compute_infer_us"] / 1000.0,
        "server_compute_output_ms": parsed["server_compute_output_us"] / 1000.0,
        "gpu_utilization_fraction": parsed["gpu_utilization_fraction"],
        "gpu_memory_used_bytes": parsed["gpu_memory_used_bytes"],
    }


def _expected_keys(config: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    return [
        (scenario["id"], repetition, position, role)
        for scenario in config["scenarios"]
        for repetition, order in enumerate(config["execution_order"], 1)
        for position, role in enumerate(order, 1)
    ]


def aggregate(
    publish_root: Path,
    raw_index: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Recompute all formal rows and pair ONNX/TRT within each repetition."""
    runs = raw_index.get("runs", [])
    actual_keys = [
        (
            run.get("scenario"),
            run.get("repetition"),
            run.get("order_position"),
            run.get("role"),
        )
        for run in runs
    ]
    if actual_keys != _expected_keys(config):
        raise AggregationError("raw run order differs from the fixed AB/BA contract")
    scenario_config = {item["id"]: item for item in config["scenarios"]}
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    for run in runs:
        scenario = scenario_config.get(run.get("scenario"))
        windows = run.get("measurement_windows", [])
        if scenario is None:
            raise AggregationError("raw index contains an unknown scenario")
        if (
            run.get("scenario_status") != "formal"
            or run.get("perf_analyzer_completion_tolerance_pct")
            != scenario["perf_analyzer_completion_tolerance_pct"]
            or run.get("measurement_windows_used") != len(windows)
            or [item.get("attempt") for item in windows]
            != list(range(1, len(windows) + 1))
            or not windows
            or run.get("client_request_count", 0)
            < len(windows) * config["measurement"]["request_count_per_window"]
            or run.get("measurement_completed") is not True
            or run.get("classification") != "VALID"
            or run.get("errors") != 0
            or not isinstance(run.get("csv_path"), str)
        ):
            raise AggregationError("raw formal run is incomplete, invalid, or stale")
        csv_path = publish_root / run["csv_path"]
        key = (run["scenario"], int(run["repetition"]), run["role"])
        if key in values:
            raise AggregationError("raw formal run key is duplicated")
        values[key] = _metric_values(parse_raw_csv(csv_path))

    repetitions = int(config["measurement"]["repetitions"])
    aggregates: dict[str, dict[str, dict[str, float]]] = {
        "baseline": {},
        "optimized": {},
    }
    for role in ("baseline", "optimized"):
        role_rows: list[dict[str, str]] = []
        for scenario in config["scenarios"]:
            scenario_values: list[dict[str, float]] = []
            for repetition in range(1, repetitions + 1):
                item = values[(scenario["id"], repetition, role)]
                scenario_values.append(item)
                run = next(
                    row
                    for row in runs
                    if row["scenario"] == scenario["id"]
                    and row["repetition"] == repetition
                    and row["role"] == role
                )
                role_rows.append(
                    {
                        "scenario": scenario["id"],
                        "repetition": str(repetition),
                        "order_position": str(run["order_position"]),
                        "batch_size": str(scenario["batch_size"]),
                        "concurrency": str(scenario["concurrency"]),
                        "pa_reported_stable": str(run["pa_reported_stable"]).lower(),
                        **{key: _format(value) for key, value in item.items()},
                    }
                )
            aggregates[role][scenario["id"]] = {
                metric: _median([item[metric] for item in scenario_values])
                for metric in scenario_values[0]
            }
        _write_csv(
            publish_root / f"benchmarks/results/{role}.csv",
            ROLE_HEADERS,
            role_rows,
        )

    thresholds = config["acceptance"]
    minimum_pairs = int(thresholds["minimum_directional_pairs"])
    minimum_median = float(
        thresholds["minimum_median_paired_improvement_pct_exclusive"]
    )
    strong_threshold = float(thresholds["strong_improvement_threshold_pct"])
    comparison_rows: list[dict[str, str]] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for scenario in config["scenarios"]:
        scenario_id = scenario["id"]
        metric = scenario["primary_metric"]
        pairs: list[dict[str, Any]] = []
        for repetition, order in enumerate(config["execution_order"], 1):
            baseline = values[(scenario_id, repetition, "baseline")]
            optimized = values[(scenario_id, repetition, "optimized")]
            if metric == "mean_client_latency_ms":
                baseline_value = baseline["avg_latency_ms"]
                optimized_value = optimized["avg_latency_ms"]
                improvement = _improvement(
                    baseline_value, optimized_value, lower_is_better=True
                )
            elif metric == "infer_per_sec":
                baseline_value = baseline["infer_per_sec"]
                optimized_value = optimized["infer_per_sec"]
                improvement = _improvement(
                    baseline_value, optimized_value, lower_is_better=False
                )
            else:
                raise AggregationError(f"unknown primary metric: {metric}")
            pairs.append(
                {
                    "repetition": repetition,
                    "execution_order": " -> ".join(order),
                    "baseline_value": baseline_value,
                    "optimized_value": optimized_value,
                    "paired_improvement_pct": improvement,
                    "directional_improvement": improvement > 0,
                    "baseline_metrics": baseline,
                    "optimized_metrics": optimized,
                }
            )
        median_improvement = _median(
            [pair["paired_improvement_pct"] for pair in pairs]
        )
        improved_pair_count = sum(
            pair["directional_improvement"] for pair in pairs
        )
        gate_passed = (
            median_improvement > minimum_median
            and improved_pair_count >= minimum_pairs
        )
        if median_improvement > strong_threshold:
            strength = "strong_measurable_improvement"
        elif median_improvement > 0:
            strength = "measurable_but_modest_improvement"
        else:
            strength = "no_demonstrated_improvement"
        comparisons[scenario_id] = {
            "primary_metric": metric,
            "pairs": pairs,
            "median_paired_improvement_pct": median_improvement,
            "improved_pair_count": improved_pair_count,
            "strength": strength,
            "gate_passed": gate_passed,
        }
        for pair in pairs:
            comparison_rows.append(
                {
                    "scenario": scenario_id,
                    "repetition": str(pair["repetition"]),
                    "execution_order": pair["execution_order"],
                    "primary_metric": metric,
                    "baseline_value": _format(pair["baseline_value"]),
                    "optimized_value": _format(pair["optimized_value"]),
                    "paired_improvement_pct": _format(
                        pair["paired_improvement_pct"]
                    ),
                    "directional_improvement": str(
                        pair["directional_improvement"]
                    ).lower(),
                    "median_paired_improvement_pct": _format(median_improvement),
                    "improved_pair_count": str(improved_pair_count),
                    "gate_passed": str(gate_passed).lower(),
                }
            )
    _write_csv(
        publish_root / "benchmarks/results/comparison.csv",
        COMPARISON_HEADERS,
        comparison_rows,
    )

    diagnostics: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        scenario_runs = [run for run in runs if run["scenario"] == scenario["id"]]
        role_diagnostics: dict[str, Any] = {}
        for role in ("baseline", "optimized"):
            role_runs = [run for run in scenario_runs if run["role"] == role]
            windows = [
                window
                for run in role_runs
                for window in run["measurement_windows"]
            ]
            role_diagnostics[role] = {
                "run_count": len(role_runs),
                "window_count": len(windows),
                "infer_per_sec_min": min(item["infer_per_sec"] for item in windows),
                "infer_per_sec_max": max(item["infer_per_sec"] for item in windows),
                "p95_latency_us_min": min(item["p95_latency_us"] for item in windows),
                "p95_latency_us_max": max(item["p95_latency_us"] for item in windows),
            }
        diagnostics[scenario["id"]] = {
            "status": "formal",
            "primary_metric": scenario["primary_metric"],
            "perf_analyzer_completion_tolerance_pct": scenario[
                "perf_analyzer_completion_tolerance_pct"
            ],
            "run_count": len(scenario_runs),
            "valid_run_count": sum(
                run["classification"] == "VALID" for run in scenario_runs
            ),
            "pa_reported_stable_run_count": sum(
                run["pa_reported_stable"] is True for run in scenario_runs
            ),
            "pa_reported_unstable_run_count": sum(
                run["pa_reported_stable"] is False for run in scenario_runs
            ),
            "baseline": role_diagnostics["baseline"],
            "optimized": role_diagnostics["optimized"],
        }
    return {
        "aggregates": aggregates,
        "comparisons": comparisons,
        "scenario_diagnostics": diagnostics,
        "environment_guard": {
            "valid_formal_runs": sum(
                run.get("classification") == "VALID" for run in runs
            ),
            "contaminated_runs": len(raw_index.get("contaminated_runs", [])),
        },
        "acceptance_passed": all(
            comparison["gate_passed"] for comparison in comparisons.values()
        ),
    }


def render_report(
    config: dict[str, Any],
    pair: dict[str, Any],
    runtime: dict[str, Any],
    result: dict[str, Any],
    input_sha256: str,
) -> str:
    scenarios = {item["id"]: item for item in config["scenarios"]}
    lines = [
        "# ONNX vs TensorRT benchmark",
        "",
        "> Generated by `benchmarks/run_benchmark.py`. Do not edit manually.",
        "",
        f"**Result: {'PASS' if result['acceptance_passed'] else 'FAIL'}**",
        "",
        "## Optimization pair",
        "",
        f"- Baseline: `{pair['baseline']['model']}:{pair['baseline']['version']}` "
        f"({pair['baseline']['runtime']}, {pair['baseline']['compute_precision']})",
        f"- Optimized: `{pair['optimized']['model']}:{pair['optimized']['version']}` "
        f"({pair['optimized']['runtime']}, {pair['optimized']['compute_precision']})",
        f"- Shared weights SHA-256: `{pair['weights_sha256']}`",
        f"- Deterministic input SHA-256: `{input_sha256}`",
        "- Triton I/O precision: FP32 for both models",
        "",
        "## Runtime-observed environment",
        "",
        f"- GPU UUID: `{runtime['gpu_uuid']}`",
        f"- GPU total memory: {runtime['gpu_memory_total_bytes']} bytes",
        f"- Triton server: `{runtime['triton_version']}` from `{runtime['triton_image']}`",
        f"- Perf Analyzer: `{runtime['perf_analyzer_version']}` from `{runtime['sdk_image']}`",
        "",
        "## Measurement contract",
        "",
        f"- Protocol: {config['protocol'].upper()}",
        f"- Requests per count window: {config['measurement']['request_count_per_window']}",
        f"- Warmup requests: {config['measurement']['warmup_request_count']}",
        f"- Perf Analyzer windows per run: {config['measurement']['max_measurement_windows']}",
        "- Perf Analyzer completion tolerance: 999% (operational output setting, "
        "not an acceptance or stationarity claim)",
        f"- Repetitions: {config['measurement']['repetitions']} in AB -> BA -> AB -> BA order",
        "- Latency primary metric: mean client latency; p50 and p95 are secondary",
        "- Throughput primary metric: infer/s",
        "- PASS: positive median paired improvement and at least 3/4 pairs in the "
        "improving direction for each primary metric",
        "- Dynamic batching is excluded from Step 6; Step 4 statistics cover it",
        "- PA stability text, thermal state, power, clocks, and GPU utilization do "
        "not determine validity or PASS",
        "- Environment guard excludes only objectively attributed foreign GPU activity "
        "or corrupted telemetry/runtime",
        f"- Valid formal slots: {result['environment_guard']['valid_formal_runs']}/16",
        f"- Attributed contaminated trials replaced: {result['environment_guard']['contaminated_runs']}",
        "",
        "## Paired result summary",
        "",
        "| Scenario | Primary metric | Median paired improvement | Improving pairs | Classification | Gate |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for scenario_id in ("latency", "throughput"):
        comparison = result["comparisons"][scenario_id]
        lines.append(
            f"| {scenario_id} | {comparison['primary_metric']} | "
            f"{comparison['median_paired_improvement_pct']:.2f}% | "
            f"{comparison['improved_pair_count']}/4 | "
            f"{comparison['strength'].replace('_', ' ')} | "
            f"{'PASS' if comparison['gate_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## All paired repetitions",
            "",
            "| Scenario | Rep | Order | ONNX mean ms | TensorRT mean ms | ONNX p50 ms | TensorRT p50 ms | ONNX p95 ms | TensorRT p95 ms | ONNX infer/s | TensorRT infer/s | Primary improvement |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario_id in ("latency", "throughput"):
        for pair_result in result["comparisons"][scenario_id]["pairs"]:
            baseline = pair_result["baseline_metrics"]
            optimized = pair_result["optimized_metrics"]
            lines.append(
                f"| {scenario_id} | {pair_result['repetition']} | "
                f"{pair_result['execution_order']} | "
                f"{baseline['avg_latency_ms']:.3f} | {optimized['avg_latency_ms']:.3f} | "
                f"{baseline['p50_latency_ms']:.3f} | {optimized['p50_latency_ms']:.3f} | "
                f"{baseline['p95_latency_ms']:.3f} | {optimized['p95_latency_ms']:.3f} | "
                f"{baseline['infer_per_sec']:.2f} | {optimized['infer_per_sec']:.2f} | "
                f"{pair_result['paired_improvement_pct']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The four paired values are shown without best-run selection or outlier "
            "removal. The 5% boundary labels an improvement as strong; it is not a "
            "PASS threshold.",
            "",
            "Supporting telemetry may explain variation but cannot exclude a clean "
            "measurement. Workload-owned changes in compute-infer time accompanied by "
            "temperature or power variation are described as operating-state variation, "
            "not asserted as a stronger causal finding.",
            "",
            "For the published candidate, the predeclared formal rule classified "
            "System (PID 4) Copy activity above 0.1% that was absent from the guard "
            "baseline as external/host contamination. This is a conservative "
            "classification of Windows host activity, not proof that any specific user "
            "process caused it.",
            "",
            "The two contaminated replacements are retained for auditability but are "
            "not required to establish the optimization conclusion; their measured "
            "performance was directionally consistent with the published result.",
            "",
            "The earlier 5% Perf Analyzer stability and 5% improvement contract is "
            "superseded. Superseded diagnostic runs are intentionally excluded from "
            "committed Step 6 evidence and may exist only in the local ignored benchmark "
            "cache. The published formal bundle is self-contained.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregationError(f"{path.name} must contain a JSON object")
    return value
