from __future__ import annotations

import unittest

from scripts.validate_client import _validate_samples


class SampleTests(unittest.TestCase):
    def test_tracked_samples_are_valid(self) -> None:
        errors: list[str] = []
        _validate_samples(errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
