from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from benchmarks.run_benchmark import (
    CLIENT_REQUEST_COUNT,
    _model_metric_delta,
    _model_metric_snapshot,
)
from scripts.validate_benchmark_evidence import (
    _pass_diagnostic_errors,
    _telemetry_device_source_errors,
)


def snapshot(offset: int) -> dict[str, object]:
    return {
        "model": "resnet50_onnx",
        "version": "1",
        "observed_at_utc": f"2026-08-02T00:00:0{offset}Z",
        "request_count": offset * 100,
        "inference_count": offset * 100,
        "execution_count": offset * 100,
        "request_duration_ns": offset * 1_000_000,
        "queue_duration_ns": offset * 200_000,
        "compute_input_duration_ns": offset * 100_000,
        "compute_infer_duration_ns": offset * 600_000,
        "compute_output_duration_ns": offset * 50_000,
    }


class PassDiagnosticTests(unittest.TestCase):
    def test_pa_client_request_count_is_read_from_final_summary(self) -> None:
        log = "Client:\n    Request count: 1567\n    Throughput: 1005.11 infer/sec\n"
        match = CLIENT_REQUEST_COUNT.search(log)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group("count")), 1567)

    def test_boundary_sample_must_reference_unchanged_periodic_device_data(self) -> None:
        periodic = {
            "sequence": 1,
            "sample_kind": "periodic",
            "device_metrics_source_sequence": 1,
            "gpu": {"uuid": "GPU-example"},
            "nvidia_processes": [],
        }
        boundary = {
            "sequence": 2,
            "sample_kind": "boundary",
            "device_metrics_source_sequence": 1,
            "gpu": periodic["gpu"],
            "nvidia_processes": [],
        }
        self.assertEqual(_telemetry_device_source_errors([periodic, boundary]), [])
        boundary["gpu"] = {"uuid": "GPU-invented"}
        self.assertTrue(_telemetry_device_source_errors([periodic, boundary]))

    def test_model_metric_snapshot_reads_one_explicit_version(self) -> None:
        response = {
            "model_stats": [
                {
                    "name": "resnet50_onnx",
                    "version": "1",
                    "inference_count": "500",
                    "execution_count": "500",
                    "inference_stats": {
                        "success": {"count": "500", "ns": "5000000"},
                        "queue": {"count": "500", "ns": "1000000"},
                        "compute_input": {"count": "500", "ns": "500000"},
                        "compute_infer": {"count": "500", "ns": "3000000"},
                        "compute_output": {"count": "500", "ns": "250000"},
                    },
                }
            ]
        }
        with patch("benchmarks.run_benchmark._http_json", return_value=response):
            value = _model_metric_snapshot("triton:8000", "resnet50_onnx", "1")
        self.assertEqual(value["request_count"], 500)
        self.assertEqual(value["compute_infer_duration_ns"], 3_000_000)

    def test_model_metric_delta_calculates_per_request_decomposition(self) -> None:
        value = _model_metric_delta(snapshot(1), snapshot(2))
        self.assertEqual(value["request_count"], 100)
        self.assertEqual(value["execution_count"], 100)
        self.assertEqual(value["per_request_us"]["queue"], 2.0)
        self.assertEqual(value["per_request_us"]["compute_infer"], 6.0)

    def test_validator_recomputes_sequence_and_statistics(self) -> None:
        before = snapshot(1)
        after = snapshot(2)
        attempt = {"attempt": 1, "infer_per_sec": 80.5, "p95_latency_us": 14500}
        run = {
            "sidecar_path": "diagnostic.json",
            "guard_boundary": {"guard_start_seq": 5, "guard_end_seq": 8},
            "pass_diagnostics": [
                {
                    **attempt,
                    "includes_initial_warmup": True,
                    "guard_boundary": {
                        "guard_start_seq": 5,
                        "guard_end_seq": 7,
                        "guard_started_at_utc": "2026-08-02T00:00:05Z",
                        "guard_ended_at_utc": "2026-08-02T00:00:07Z",
                        "guard_started_monotonic_ns": 5_000_000_000,
                        "guard_ended_monotonic_ns": 7_000_000_000,
                    },
                    "triton_statistics": {
                        "before": before,
                        "after": after,
                        "delta": _model_metric_delta(before, after),
                    },
                }
            ],
        }
        telemetry = [
            {
                "sequence": sequence,
                "observed_at_utc": f"2026-08-02T00:00:0{sequence}Z",
                "host_monotonic_ns": sequence * 1_000_000_000,
            }
            for sequence in range(5, 9)
        ]
        self.assertEqual(
            _pass_diagnostic_errors(
                run, [attempt], telemetry, "resnet50_onnx", "1", 100
            ),
            [],
        )
        tampered = copy.deepcopy(run)
        tampered["pass_diagnostics"][0]["triton_statistics"]["delta"][
            "queue_duration_ns"
        ] += 1
        self.assertTrue(
            any(
                "delta is stale" in error
                for error in _pass_diagnostic_errors(
                    tampered,
                    [attempt],
                    telemetry,
                    "resnet50_onnx",
                    "1",
                    100,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
