from __future__ import annotations

import copy
import unittest
from pathlib import PurePosixPath

from scripts.model_preparation import prepare_models
from scripts.validate_model_repository import validate_spec_semantics


class TensorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()

    def test_resnet_onnx_and_tensorrt_share_one_contract(self) -> None:
        resnet = self.spec["models"]["resnet50"]
        onnx = resnet["serving"]["onnx"]
        tensorrt = resnet["serving"]["tensorrt"]
        self.assertEqual(onnx["max_batch_size"], tensorrt["max_batch_size"])
        for profile_shape in tensorrt["profile"].values():
            self.assertEqual(profile_shape[1:], resnet["input"]["shape"][1:])

    def test_spec_models_expose_only_dynamic_batch(self) -> None:
        for model in self.spec["models"].values():
            expected_axes = {
                model["input"]["name"]: ["batch"],
                model["output"]["name"]: ["batch"],
            }
            self.assertEqual(model["export"]["dynamic_axes"], expected_axes)
            for tensor_kind in ("input", "output"):
                shape = model[tensor_kind]["shape"]
                self.assertEqual(shape[0], -1)
                self.assertTrue(all(value > 0 for value in shape[1:]))

    def test_tensorrt_config_has_no_cross_capability_fallback(self) -> None:
        serving = self.spec["models"]["resnet50"]["serving"]["tensorrt"]
        config_path = prepare_models.REPOSITORY_ROOT / serving["config_path"]
        config = config_path.read_text(encoding="utf-8")
        self.assertIn("cc_model_filenames {", config)
        self.assertNotIn("default_model_filename", config)

        artifact = PurePosixPath(serving["versions"]["1"]["artifact_path"])
        fallback = prepare_models.REPOSITORY_ROOT / artifact.parent.as_posix() / "model.plan"
        self.assertFalse(fallback.exists())

    def test_embedded_yolo_nms_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["yolo11n"]["export"]["nms"] = True
        self.assertTrue(any("NMS" in error for error in validate_spec_semantics(changed)))


if __name__ == "__main__":
    unittest.main()
