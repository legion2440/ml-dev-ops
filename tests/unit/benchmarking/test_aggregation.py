from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.aggregate_results import RAW_COLUMNS, aggregate, parse_raw_csv
from benchmarks.run_benchmark import CONFIG_PATH


def write_raw(path: Path, infer_per_sec: float, p95_us: float) -> None:
    row = {
        "Inferences/Second": str(infer_per_sec),
        "Client Send": "10",
        "Network+Server Send/Recv": "900",
        "Server Queue": "20",
        "Server Compute Input": "30",
        "Server Compute Infer": (
            "500" if p95_us < 900 else ("900" if p95_us > 1100 else "700")
        ),
        "Server Compute Output": "40",
        "Client Recv": "90",
        "p50 latency": str(p95_us - 200),
        "p95 latency": str(p95_us),
        "p99 latency": str(p95_us + 200),
        "Avg GPU Utilization": "GPU-example:0.42;",
        "Max GPU Memory Usage": f"GPU-example:{2 * 1024**3};",
        "Total GPU Memory": f"GPU-example:{12 * 1024**3};",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RAW_COLUMNS.values()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


class AggregationTests(unittest.TestCase):
    def test_raw_units_and_median_gate_are_recomputed(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for scenario in config["scenarios"]:
                for repetition, order in enumerate(config["execution_order"], 1):
                    for position, role in enumerate(order, 1):
                        relative = Path(
                            f"benchmarks/results/raw/{scenario['id']}/{repetition}-{role}.csv"
                        )
                        optimized = role == "optimized"
                        optimized_good = optimized and repetition != 4
                        write_raw(
                            root / relative,
                            120.0 if optimized_good else (90.0 if optimized else 100.0),
                            800.0 if optimized_good else (1200.0 if optimized else 1000.0),
                        )
                        runs.append(
                            {
                                "scenario": scenario["id"],
                                "scenario_status": scenario["status"],
                                "perf_analyzer_completion_tolerance_pct": scenario[
                                    "perf_analyzer_completion_tolerance_pct"
                                ],
                                "measurement_windows_used": 3,
                                "measurement_windows": [
                                    {
                                        "attempt": attempt,
                                        "infer_per_sec": (
                                            120.0
                                            if optimized_good
                                            else (90.0 if optimized else 100.0)
                                        )
                                        + repetition
                                        + attempt,
                                        "p95_latency_us": int(
                                            (
                                                800.0
                                                if optimized_good
                                                else (1200.0 if optimized else 1000.0)
                                            )
                                            + repetition
                                            + attempt
                                        ),
                                    }
                                    for attempt in range(1, 4)
                                ],
                                "client_request_count": 1500,
                                "measurement_completed": True,
                                "pa_reported_stable": True,
                                "classification": "VALID",
                                "errors": 0,
                                "repetition": repetition,
                                "order_position": position,
                                "role": role,
                                "csv_path": relative.as_posix(),
                            }
                        )
            parsed = parse_raw_csv(root / runs[0]["csv_path"])
            self.assertEqual(parsed["p95_latency_us"], 1000.0)
            self.assertEqual(parsed["gpu_memory_used_bytes"], 2 * 1024**3)
            self.assertEqual(parsed["gpu_utilization_fraction"], 0.42)
            self.assertEqual(parsed["avg_latency_us"], 1790.0)
            result = aggregate(
                root,
                {
                    "runs": runs,
                    "contaminated_runs": [
                        {
                            "classification": "CONTAMINATED",
                            "infer_per_sec": 999999999,
                            "p95_latency_us": 1,
                        }
                    ],
                },
                config,
            )
            self.assertTrue(result["acceptance_passed"])
            self.assertAlmostEqual(
                result["aggregates"]["baseline"]["latency"]["p95_latency_ms"], 1.0
            )
            self.assertGreater(
                result["comparisons"]["latency"][
                    "median_paired_improvement_pct"
                ],
                0.0,
            )
            self.assertAlmostEqual(
                result["comparisons"]["throughput"][
                    "median_paired_improvement_pct"
                ],
                20.0,
            )
            self.assertEqual(
                result["comparisons"]["latency"]["improved_pair_count"],
                3,
            )
            self.assertEqual(
                result["comparisons"]["throughput"]["improved_pair_count"],
                3,
            )
            self.assertEqual(
                result["scenario_diagnostics"]["latency"]["baseline"][
                    "window_count"
                ],
                12,
            )
            self.assertEqual(result["environment_guard"]["valid_formal_runs"], 16)
            self.assertEqual(result["environment_guard"]["contaminated_runs"], 1)
            comparison = (
                root / "benchmarks/results/comparison.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("latency,1,baseline -> optimized", comparison)
            self.assertIn("throughput,4,optimized -> baseline", comparison)


if __name__ == "__main__":
    unittest.main()
