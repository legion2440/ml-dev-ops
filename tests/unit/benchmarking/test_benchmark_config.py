from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from benchmarks.run_benchmark import (
    CONFIG_PATH,
    PAIR_CONTRACT_PATH,
    _normalize_runtime_contract,
    build_perf_analyzer_command,
    main,
)


class BenchmarkConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.pair = json.loads(PAIR_CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_config_matches_schema_and_fixed_order(self) -> None:
        schema = json.loads(
            (CONFIG_PATH.parents[2] / "schemas/benchmark-config.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(self.config)
        self.assertEqual(
            self.config["execution_order"],
            [
                ["baseline", "optimized"],
                ["optimized", "baseline"],
                ["baseline", "optimized"],
                ["optimized", "baseline"],
            ],
        )

    def test_perf_analyzer_command_uses_count_window_and_metrics_contract(self) -> None:
        scenario = self.config["scenarios"][0]
        command = build_perf_analyzer_command(
            self.config,
            self.pair,
            "baseline",
            scenario,
            Path("input"),
            Path("raw.csv"),
        )
        rendered = " ".join(command)
        self.assertEqual(command[:2], ["perf_analyzer", "-v"])
        self.assertTrue(self.config["load_generator"]["realtime_clock_guard"])
        for fragment in (
            "--measurement-mode count_windows",
            "--measurement-request-count 500",
            "--warmup-request-count 100",
            "--stability-percentage 999",
            "--percentile 95",
            "--max-trials 3",
            "--collect-metrics",
            "--metrics-interval 100",
            "--verbose-csv",
        ):
            self.assertIn(fragment, rendered)

    def test_live_numeric_scheduler_fields_are_normalized(self) -> None:
        metadata = {
            "inputs": [{"name": "images", "datatype": "FP32", "shape": [-1, 3, 224, 224]}],
            "outputs": [{"name": "logits", "datatype": "FP32", "shape": [-1, 1000]}],
        }
        config = {
            "max_batch_size": 8,
            "dynamic_batching": {
                "preferred_batch_size": [4, 8],
                "max_queue_delay_microseconds": 5000,
            },
            "instance_group": [{"name": "resnet50_onnx", "kind": "KIND_GPU", "count": 1, "gpus": [0]}],
        }
        normalized = _normalize_runtime_contract(metadata, config)
        self.assertEqual(normalized["dynamic_batching"]["max_queue_delay_microseconds"], 5000)
        self.assertEqual(
            normalized["instance_group"],
            [{"kind": "KIND_GPU", "count": 1, "gpus": [0]}],
        )

    def test_pa_completion_setting_is_not_an_acceptance_threshold(self) -> None:
        self.assertEqual(
            {item["id"] for item in self.config["scenarios"]},
            {"latency", "throughput"},
        )
        self.assertTrue(
            all(
                item["perf_analyzer_completion_tolerance_pct"] == 999
                for item in self.config["scenarios"]
            )
        )
        self.assertEqual(
            self.config["acceptance"],
            {
                "minimum_median_paired_improvement_pct_exclusive": 0.0,
                "minimum_directional_pairs": 3,
                "strong_improvement_threshold_pct": 5.0,
            },
        )

    def test_environment_guard_is_predeclared_and_bounded(self) -> None:
        guard = self.config["environment_guard"]
        self.assertEqual(
            guard["host_observer"], "windows_gpu_engine_nvidia_smi"
        )
        self.assertEqual(guard["sample_interval_ms"], 1000)
        self.assertEqual(guard["baseline_seconds"], 5)
        self.assertEqual(guard["minimum_consecutive_baseline_samples"], 5)
        self.assertEqual(guard["gpu_engine_activity_threshold_percent"], 0.1)
        self.assertEqual(guard["max_contaminated_attempts_per_slot"], 3)
        self.assertGreaterEqual(guard["maximum_sample_gap_ms"], 1000)
        self.assertIn("vmmemWSL.exe", guard["benchmark_owned_processes"])
        self.assertIn("chrome.exe", guard["forbidden_processes"])

    def test_check_mode_never_starts_a_candidate(self) -> None:
        with (
            patch.object(sys, "argv", ["run_benchmark.py", "--check"]),
            patch(
                "scripts.validate_benchmark_evidence.validate", return_value=[]
            ),
            patch("benchmarks.run_benchmark.orchestrate") as orchestrate,
        ):
            self.assertEqual(main(), 0)
        orchestrate.assert_not_called()

    def test_historical_only_mode_never_starts_a_candidate(self) -> None:
        with (
            patch.object(sys, "argv", ["run_benchmark.py", "--historical-only"]),
            patch(
                "scripts.validate_benchmark_evidence.validate", return_value=[]
            ) as validate,
            patch("benchmarks.run_benchmark.orchestrate") as orchestrate,
        ):
            self.assertEqual(main(), 0)
        validate.assert_called_once_with(
            CONFIG_PATH.parents[2], historical_only=True
        )
        orchestrate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
