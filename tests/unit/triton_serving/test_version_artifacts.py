from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.model_preparation import prepare_models


class VersionArtifactTests(unittest.TestCase):
    def test_resnet_versions_have_distinct_artifacts(self) -> None:
        manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
        versions = manifest["models"]["resnet50_onnx"]["versions"]
        self.assertNotEqual(versions["1"]["artifact"]["sha256"], versions["2"]["artifact"]["sha256"])
        self.assertEqual(versions["2"]["derived_from"], "1")

    def test_tensorrt_has_no_generic_fallback(self) -> None:
        spec = prepare_models.load_spec()
        serving = spec["models"]["resnet50"]["serving"]["tensorrt"]
        config = prepare_models.serving_model_config(spec, serving["name"])
        self.assertIn("cc_model_filenames", config)
        self.assertNotIn("default_model_filename", config)
        artifact = Path(prepare_models.serving_version_path(serving, "1"))
        self.assertFalse((prepare_models.REPOSITORY_ROOT / artifact.parent / "model.plan").exists())


if __name__ == "__main__":
    unittest.main()
