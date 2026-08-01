from __future__ import annotations

import unittest

from client.preprocessing import preprocess_detection
from tests.unit.client.helpers import contract, loaded_image


class YoloLetterboxTests(unittest.TestCase):
    def test_centered_letterbox_geometry(self) -> None:
        entry = contract("yolo11n_onnx")
        source = loaded_image(width=320, height=160)
        tensor, metadata = preprocess_detection([source], entry)
        self.assertEqual(list(tensor.shape), [1, *entry["input"]["shape"][1:]])
        geometry = metadata[0]
        target_height, target_width = entry["input"]["shape"][-2:]
        self.assertAlmostEqual(geometry.scale, min(target_width / 320, target_height / 160))
        self.assertEqual(geometry.pad_x, 0)
        self.assertGreater(geometry.pad_y, 0)


if __name__ == "__main__":
    unittest.main()
