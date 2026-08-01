from __future__ import annotations

import unittest

import yaml

from scripts.model_preparation import prepare_models


class ComposeVerifierTests(unittest.TestCase):
    def test_verifier_is_profile_only_and_has_no_ports_or_gpu(self) -> None:
        compose = yaml.safe_load((prepare_models.REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        service = compose["services"]["triton-verifier"]
        self.assertEqual(service["profiles"], ["verification"])
        self.assertNotIn("ports", service)
        self.assertNotIn("deploy", service)
        self.assertEqual(service["restart"], "no")


if __name__ == "__main__":
    unittest.main()
