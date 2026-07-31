from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_preparation import prepare_models
from scripts.validate_model_repository import (
    validate_artifact_inventory,
    validate_spec_semantics,
)


class ArtifactStalenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()
        self.manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_missing_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            errors: list[str] = []
            validate_artifact_inventory(self.manifest, errors, Path(directory))
        self.assertEqual(sum("missing" in error for error in errors), 3)

    def test_different_compute_capability_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["build"]["target"]["compute_capability"] = "8.6"
        errors = validate_spec_semantics(changed)
        self.assertTrue(any("compute capability" in error for error in errors))

    def test_contract_change_makes_generated_config_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["resnet50"]["output"]["shape"] = [-1, 999]
        generated = prepare_models.render_config(changed, "resnet50_onnx")
        tracked = (prepare_models.REPOSITORY_ROOT / "models/resnet50_onnx/config.pbtxt").read_text(
            encoding="utf-8"
        )
        self.assertNotEqual(generated, tracked)


if __name__ == "__main__":
    unittest.main()
