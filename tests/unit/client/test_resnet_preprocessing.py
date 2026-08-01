from __future__ import annotations

import unittest

import numpy as np

from client.preprocessing import preprocess_classification
from tests.unit.client.helpers import contract, loaded_image


class ResNetPreprocessingTests(unittest.TestCase):
    def test_shape_and_normalization_come_from_contract(self) -> None:
        entry = contract("resnet50_onnx")
        tensor = preprocess_classification([loaded_image()], entry)
        expected = [1, *entry["input"]["shape"][1:]]
        self.assertEqual(list(tensor.shape), expected)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
