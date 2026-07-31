from __future__ import annotations

import copy
import unittest

from scripts.model_preparation import prepare_models
from scripts.validate_model_repository import validate_spec_semantics


class ModelSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()

    def test_accepted_spec_has_no_semantic_errors(self) -> None:
        self.assertEqual(validate_spec_semantics(self.spec), [])

    def test_latest_preparation_image_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["build"]["exporter_image"] = (
            "docker.io/ultralytics/ultralytics:latest@sha256:" + "a" * 64
        )
        self.assertTrue(any("latest" in error for error in validate_spec_semantics(changed)))

    def test_unresolved_source_hash_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["resnet50"]["source"]["hash_status"] = "unresolved"
        changed["models"]["resnet50"]["source"]["sha256"] = None
        errors = validate_spec_semantics(changed)
        self.assertTrue(any("must be resolved" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
