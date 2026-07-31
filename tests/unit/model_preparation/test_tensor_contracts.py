from __future__ import annotations

import copy
import unittest

from scripts.model_preparation import prepare_models
from scripts.validate_model_repository import validate_spec_semantics


class TensorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()

    def test_resnet_onnx_and_tensorrt_share_one_contract(self) -> None:
        resnet = self.spec["models"]["resnet50"]
        self.assertEqual(resnet["input"]["shape"], [-1, 3, 224, 224])
        self.assertEqual(resnet["output"]["shape"], [-1, 1000])
        self.assertEqual(resnet["serving"]["onnx"]["max_batch_size"], 8)
        self.assertEqual(resnet["serving"]["tensorrt"]["max_batch_size"], 8)

    def test_yolo_only_batch_is_dynamic(self) -> None:
        yolo = self.spec["models"]["yolo11n"]
        self.assertEqual(
            yolo["export"]["dynamic_axes"],
            {"images": ["batch"], "output0": ["batch"]},
        )
        self.assertEqual(yolo["input"]["shape"], [-1, 3, 640, 640])
        self.assertEqual(yolo["output"]["shape"], [-1, 84, 8400])

    def test_embedded_yolo_nms_is_rejected(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["yolo11n"]["export"]["nms"] = True
        self.assertTrue(any("NMS" in error for error in validate_spec_semantics(changed)))


if __name__ == "__main__":
    unittest.main()
