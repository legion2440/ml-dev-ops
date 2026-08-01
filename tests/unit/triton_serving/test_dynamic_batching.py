from __future__ import annotations

import copy
import unittest

from scripts.model_preparation import prepare_models
from shared.triton_model_config import validate_contract_relationships


class DynamicBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()

    def test_spec_scheduling_is_the_model_config_scheduling(self) -> None:
        for name in prepare_models.SERVING_MODELS:
            config = prepare_models.serving_model_config(self.spec, name)
            self.assertEqual(validate_contract_relationships(config), [])

    def test_preferred_batch_above_capacity_is_rejected(self) -> None:
        config = copy.deepcopy(prepare_models.serving_model_config(self.spec, "yolo11n_onnx"))
        config["dynamic_batching"]["preferred_batch_size"] = [config["max_batch_size"] + 1]
        self.assertTrue(any("exceeds" in error for error in validate_contract_relationships(config)))

    def test_negative_queue_delay_is_rejected(self) -> None:
        config = copy.deepcopy(prepare_models.serving_model_config(self.spec, "resnet50_onnx"))
        config["dynamic_batching"]["max_queue_delay_microseconds"] = -1
        self.assertTrue(any("nonnegative" in error for error in validate_contract_relationships(config)))


if __name__ == "__main__":
    unittest.main()
