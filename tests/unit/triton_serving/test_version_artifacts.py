from __future__ import annotations

import json
import unittest

import yaml

from scripts.model_preparation import prepare_models


class VersionArtifactTests(unittest.TestCase):
    def test_resnet_versions_have_distinct_artifacts(self) -> None:
        manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
        versions = manifest["models"]["resnet50_onnx"]["versions"]
        self.assertNotEqual(versions["1"]["artifact"]["sha256"], versions["2"]["artifact"]["sha256"])
        self.assertEqual(versions["2"]["derived_from"], "1")

    def test_tensorrt_uses_one_portable_plan_filename(self) -> None:
        spec = prepare_models.load_spec()
        serving = spec["models"]["resnet50"]["serving"]["tensorrt"]
        config = prepare_models.serving_model_config(spec, serving["name"])
        self.assertEqual(config["default_model_filename"], "model.plan")
        self.assertNotIn("cc_model_filenames", config)
        self.assertEqual(
            prepare_models.serving_version_path(serving, "1"),
            "models/resnet50_tensorrt/1/model.plan",
        )

    def test_runtime_verifier_cannot_overwrite_step4(self) -> None:
        compose = yaml.safe_load(
            (prepare_models.REPOSITORY_ROOT / "docker-compose.yml").read_text(
                encoding="utf-8"
            )
        )
        service = compose["services"]["triton-verifier"]
        self.assertEqual(
            service["command"][-2:],
            ["--evidence-directory", "docs/evidence/portability"],
        )
        writable = [
            volume
            for volume in service["volumes"]
            if volume.get("read_only") is not True
        ]
        self.assertEqual(len(writable), 1)
        self.assertEqual(writable[0]["source"], "./docs/evidence/portability")
        self.assertNotIn("step-4", writable[0]["target"])


if __name__ == "__main__":
    unittest.main()
