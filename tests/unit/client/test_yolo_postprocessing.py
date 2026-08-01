from __future__ import annotations

import unittest

import numpy as np

from client.postprocessing import detection_predictions
from client.preprocessing import preprocess_detection
from tests.unit.client.helpers import contract, loaded_image


class YoloPostprocessingTests(unittest.TestCase):
    def test_inverse_letterbox_round_trip_and_clipping(self) -> None:
        entry = contract("yolo11n_onnx")
        source = loaded_image(width=320, height=160)
        _, metadata = preprocess_detection([source], entry)
        geometry = metadata[0]
        output_shape = [1, *entry["output"]["shape"][1:]]
        output = np.zeros(output_shape, dtype=np.float32)
        original = np.array([50.0, 20.0, 250.0, 120.0])
        model_box = original.copy()
        model_box[[0, 2]] = model_box[[0, 2]] * geometry.scale + geometry.pad_x
        model_box[[1, 3]] = model_box[[1, 3]] * geometry.scale + geometry.pad_y
        output[0, 0, 0] = (model_box[0] + model_box[2]) / 2
        output[0, 1, 0] = (model_box[1] + model_box[3]) / 2
        output[0, 2, 0] = model_box[2] - model_box[0]
        output[0, 3, 0] = model_box[3] - model_box[1]
        output[0, 4, 0] = 0.9
        prediction = detection_predictions(
            output,
            metadata,
            entry["labels"],
            entry["output_semantics"],
            0.25,
            0.7,
            10,
        )[0][0]
        np.testing.assert_allclose(prediction["box_xyxy"], original, atol=1e-4)
        x1, y1, x2, y2 = prediction["box_xyxy"]
        self.assertTrue(0 <= x1 < x2 <= source.width)
        self.assertTrue(0 <= y1 < y2 <= source.height)

    def test_runtime_rejects_detection_semantics_drift(self) -> None:
        entry = contract("yolo11n_onnx")
        source = loaded_image()
        _, metadata = preprocess_detection([source], entry)
        output = np.zeros([1, *entry["output"]["shape"][1:]], dtype=np.float32)
        changes = {
            "kind": "unknown",
            "box_format": "xyxy",
            "class_scores_start": 5,
            "has_objectness": True,
            "class_aware_nms": False,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                semantics = {**entry["output_semantics"], field: value}
                with self.assertRaisesRegex(ValueError, "semantics|offset"):
                    detection_predictions(
                        output,
                        metadata,
                        entry["labels"],
                        semantics,
                        0.25,
                        0.7,
                        10,
                    )


if __name__ == "__main__":
    unittest.main()
