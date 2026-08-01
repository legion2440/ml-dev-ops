from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from client.input_loader import InputError, discover_images, load_image


class InputLoaderTests(unittest.TestCase):
    def test_directory_order_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.jpg", "A.png", "a.jpg"):
                Image.new("RGB", (2, 3)).save(root / name)
            self.assertEqual(
                [path.name for path in discover_images(root)],
                ["a.jpg", "A.png", "b.jpg"],
            )

    def test_malformed_image_has_sanitized_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(InputError, r"broken\.jpg") as context:
                load_image(path)
            self.assertNotIn(directory, str(context.exception))


if __name__ == "__main__":
    unittest.main()
