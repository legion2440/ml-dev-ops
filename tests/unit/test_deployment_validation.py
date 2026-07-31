"""Regression tests for step 2 deployment validation contracts."""

from __future__ import annotations

import unittest

from scripts import validate_deployment
from scripts import validate_runtime_evidence


class ImagePinValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = validate_deployment._load_env(validate_deployment.ENV_EXAMPLE_PATH)
        compose = validate_deployment._load_yaml(validate_deployment.COMPOSE_PATH)
        self.services = compose["services"]

    def test_alternate_full_tags_do_not_require_validator_edits(self) -> None:
        changed = {
            **self.env,
            "TRITON_IMAGE": "nvcr.io/nvidia/tritonserver:99.99-py3",
            "PROMETHEUS_IMAGE": "prom/prometheus:v9.8.7",
            "GRAFANA_IMAGE": "grafana/grafana:9.8.7",
            "DCGM_EXPORTER_IMAGE": (
                "nvcr.io/nvidia/k8s/dcgm-exporter:9.8.7-6.5.4-distroless"
            ),
        }
        errors: list[str] = []

        validate_deployment._validate_env(changed, errors)
        validate_deployment._validate_images(self.services, changed, errors)

        self.assertEqual(errors, [])

    def test_latest_and_incomplete_tags_are_rejected(self) -> None:
        changed = {
            **self.env,
            "TRITON_IMAGE": "nvcr.io/nvidia/tritonserver:latest",
            "PROMETHEUS_IMAGE": "prom/prometheus:v9",
        }
        errors: list[str] = []

        validate_deployment._validate_images(self.services, changed, errors)

        self.assertTrue(any("latest" in error for error in errors))
        self.assertTrue(any("PROMETHEUS_IMAGE" in error for error in errors))

    def test_runtime_evidence_detects_changed_canonical_pin(self) -> None:
        changed = {
            **self.env,
            "TRITON_IMAGE": "nvcr.io/nvidia/tritonserver:99.99-py3",
        }
        errors: list[str] = []

        validate_runtime_evidence._validate_environment(changed, errors)

        self.assertTrue(any("stale" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
