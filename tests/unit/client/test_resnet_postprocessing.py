from __future__ import annotations

import unittest

import numpy as np

from client.postprocessing import classification_predictions, stable_softmax
from tests.unit.client.helpers import contract


class ResNetPostprocessingTests(unittest.TestCase):
    def test_softmax_is_stable_for_large_logits(self) -> None:
        probabilities = stable_softmax(np.array([[10000.0, 9999.0]], dtype=np.float64))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_top_k_uses_contract_labels(self) -> None:
        entry = contract("resnet50_onnx")
        logits = np.arange(len(entry["labels"]), dtype=np.float32)[None, :]
        predictions = classification_predictions(
            logits, entry["labels"], entry["output_semantics"], 3
        )[0]
        self.assertEqual([item["rank"] for item in predictions], [1, 2, 3])
        self.assertEqual(predictions[0]["label"], entry["labels"][-1])

    def test_runtime_rejects_non_logit_output_semantics(self) -> None:
        entry = contract("resnet50_onnx")
        logits = np.zeros((1, len(entry["labels"])), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "output semantics"):
            classification_predictions(logits, entry["labels"], {"kind": "scores"}, 1)


if __name__ == "__main__":
    unittest.main()
