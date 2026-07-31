from __future__ import annotations

import copy
import json
import unittest

from scripts.model_preparation import prepare_models


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()
        self.manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_current_manifest_is_not_stale(self) -> None:
        self.assertEqual(prepare_models.manifest_staleness(self.spec, self.manifest), [])

    def test_source_hash_change_makes_manifest_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["resnet50"]["source"]["sha256"] = "a" * 64
        errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any("source SHA-256" in error for error in errors))

    def test_opset_change_makes_manifest_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["build"]["onnx_opset"] += 1
        errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any("ONNX opset" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
