"""Regression tests for the Step 7 monitoring contract."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monitoring import verify_runtime
from scripts import validate_monitoring


class MonitoringConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dashboard = validate_monitoring._load_json(
            validate_monitoring.DASHBOARD_PATH
        )
        self.datasource = validate_monitoring._load_yaml(
            validate_monitoring.DATASOURCE_PATH
        )
        self.provider = validate_monitoring._load_yaml(validate_monitoring.PROVIDER_PATH)
        self.alerts = validate_monitoring._load_yaml(validate_monitoring.ALERTS_PATH)

    def test_repository_monitoring_contract_is_valid(self) -> None:
        self.assertEqual(validate_monitoring.validate_config(), [])

    def test_throughput_cannot_use_request_success_counter(self) -> None:
        changed = copy.deepcopy(self.dashboard)
        panel = validate_monitoring._panel_map(changed)["Inference Throughput"]
        panel["targets"][0]["expr"] = (
            "sum by (model, version) "
            "(rate(nv_inference_request_success[1m]))"
        )
        errors: list[str] = []

        validate_monitoring._validate_grafana(
            changed, self.datasource, self.provider, errors
        )

        self.assertTrue(any("Inference Throughput" in error for error in errors))

    def test_latency_cannot_hide_idle_denominator_with_clamp_min(self) -> None:
        changed = copy.deepcopy(self.dashboard)
        panel = validate_monitoring._panel_map(changed)["Average Request Latency"]
        panel["targets"][0]["expr"] = (
            "sum by (model, version) "
            "(rate(nv_inference_request_duration_us[1m])) / "
            "clamp_min(sum by (model, version) "
            "(rate(nv_inference_request_success[1m])), 1) / 1000"
        )
        errors: list[str] = []

        validate_monitoring._validate_grafana(
            changed, self.datasource, self.provider, errors
        )

        self.assertTrue(any("clamp_min" in error for error in errors))

    def test_gpu_percent_scale_cannot_be_rescaled(self) -> None:
        changed = copy.deepcopy(self.dashboard)
        panel = validate_monitoring._panel_map(changed)["GPU Utilization"]
        panel["targets"][0]["expr"] += " / 100"
        errors: list[str] = []

        validate_monitoring._validate_grafana(
            changed, self.datasource, self.provider, errors
        )

        self.assertTrue(any("0..100" in error for error in errors))

    def test_high_latency_alert_requires_the_traffic_guard(self) -> None:
        changed = copy.deepcopy(self.alerts)
        rule = changed["groups"][0]["rules"][0]
        rule["expr"] = str(rule["expr"]).split("and on", 1)[0]
        errors: list[str] = []

        validate_monitoring._validate_alerts(changed, errors)

        self.assertTrue(any("HighInferenceLatency" in error for error in errors))


class MonitoringRuntimeContractTests(unittest.TestCase):
    @staticmethod
    def _sample(value: float, *, model: bool = False) -> dict[str, object]:
        metric = (
            {
                "model": validate_monitoring.TARGET_MODEL,
                "version": validate_monitoring.TARGET_VERSION,
            }
            if model
            else {"UUID": "GPU-test"}
        )
        return {"metric": metric, "timestamp": 1.0, "value": value}

    def test_zero_gpu_utilization_is_valid_runtime_data(self) -> None:
        values = {
            "inference_throughput": [self._sample(1.0, model=True)],
            "request_rate": [self._sample(1.0, model=True)],
            "average_request_latency": [self._sample(1.0, model=True)],
            "gpu_utilization": [self._sample(0.0)],
            "failed_requests": [self._sample(0.0, model=True)],
        }

        self.assertTrue(verify_runtime._required_query_data_available(values))

    def test_unrelated_compose_drift_is_not_monitoring_incompatibility(self) -> None:
        self.assertIn("docker-compose.yml", validate_monitoring.HASHED_ARTIFACTS)
        self.assertNotIn(
            "docker-compose.yml",
            validate_monitoring.CURRENT_COMPATIBILITY_HASHED_ARTIFACTS,
        )
        self.assertIn(
            "monitoring/prometheus/alerts.yml",
            validate_monitoring.CURRENT_COMPATIBILITY_HASHED_ARTIFACTS,
        )

    def test_target_model_rates_must_be_positive(self) -> None:
        values = {
            "inference_throughput": [self._sample(0.0, model=True)],
            "request_rate": [self._sample(1.0, model=True)],
            "average_request_latency": [self._sample(1.0, model=True)],
            "gpu_utilization": [self._sample(0.0)],
            "failed_requests": [self._sample(0.0, model=True)],
        }

        self.assertFalse(verify_runtime._required_query_data_available(values))

    def test_tampered_query_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prometheus-queries.json"
            value = json.loads(
                validate_monitoring.QUERY_EVIDENCE_PATH.read_text(encoding="utf-8")
            )
            value["queries"][0]["samples"][0]["value"] = 999999
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with patch.object(validate_monitoring, "QUERY_EVIDENCE_PATH", path):
                errors = validate_monitoring.validate_evidence()
        self.assertTrue(any("query evidence reference is stale" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
