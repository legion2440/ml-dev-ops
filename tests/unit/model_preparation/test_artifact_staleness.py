from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

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
        self.assertEqual(
            sum("missing" in error for error in errors),
            len(prepare_models.serving_artifact_paths(self.spec)),
        )

    def test_consistent_compute_capability_change_stales_generated_data(self) -> None:
        changed = copy.deepcopy(self.spec)
        current = changed["build"]["target"]["compute_capability"]
        major, minor = current.split(".", 1)
        capability = f"{major}.{int(minor) + 1}"
        changed["build"]["target"]["compute_capability"] = capability
        serving = changed["models"]["resnet50"]["serving"]["tensorrt"]
        for version in serving["versions"].values():
            artifact = PurePosixPath(version["artifact_path"])
            version["artifact_path"] = (
                artifact.parent / f"model_cc{capability.replace('.', '')}.plan"
            ).as_posix()

        self.assertEqual(validate_spec_semantics(changed), [])
        generated = prepare_models.render_config(changed, serving["name"])
        tracked = (prepare_models.REPOSITORY_ROOT / serving["config_path"]).read_text(
            encoding="utf-8"
        )
        self.assertNotEqual(generated, tracked)
        manifest_errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any("compute capability" in error for error in manifest_errors))
        self.assertTrue(any("version 1 path" in error for error in manifest_errors))

    def test_contract_change_makes_generated_config_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["resnet50"]["output"]["shape"][-1] += 1
        changed["models"]["resnet50"]["labels"]["count"] += 1
        self.assertEqual(validate_spec_semantics(changed), [])
        generated = prepare_models.render_config(changed, "resnet50_onnx")
        tracked = (prepare_models.REPOSITORY_ROOT / "models/resnet50_onnx/config.pbtxt").read_text(
            encoding="utf-8"
        )
        self.assertNotEqual(generated, tracked)
        manifest_errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any(".output" in error for error in manifest_errors))

    def test_profile_change_makes_generated_data_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        resnet = changed["models"]["resnet50"]
        serving = resnet["serving"]
        new_max_batch = serving["tensorrt"]["profile"]["max"][0] + 1
        serving["tensorrt"]["profile"]["max"][0] = new_max_batch
        serving["tensorrt"]["max_batch_size"] = new_max_batch
        serving["onnx"]["max_batch_size"] = new_max_batch

        self.assertEqual(validate_spec_semantics(changed), [])
        generated = prepare_models.render_config(changed, serving["tensorrt"]["name"])
        tracked = (
            prepare_models.REPOSITORY_ROOT / serving["tensorrt"]["config_path"]
        ).read_text(encoding="utf-8")
        self.assertNotEqual(generated, tracked)
        manifest_errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any("profile" in error for error in manifest_errors))


if __name__ == "__main__":
    unittest.main()
