from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from client.input_loader import LoadedImage
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

    def test_odd_resize_geometry_matches_pinned_torchvision_reference(self) -> None:
        height, width = 683, 1024
        y_coordinates, x_coordinates = np.indices((height, width))
        pixels = np.stack(
            (
                x_coordinates % 256,
                y_coordinates % 256,
                (x_coordinates + 3 * y_coordinates) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        source = LoadedImage(
            path=Path("torchvision-reference.png"),
            name="torchvision-reference.png",
            sha256="0" * 64,
            width=width,
            height=height,
            image=Image.fromarray(pixels, "RGB"),
        )

        tensor = preprocess_classification([source], contract("resnet50_onnx"))[0]

        self.assertEqual(
            hashlib.sha256(tensor.tobytes()).hexdigest(),
            "8e80b04d6d9e98dc466cd4f2cd396a632863d9ce9e5a6b21118e7b5abaa7cafa",
        )


if __name__ == "__main__":
    unittest.main()
