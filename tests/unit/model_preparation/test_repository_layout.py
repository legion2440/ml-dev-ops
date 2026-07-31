from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.validate_model_repository import (
    EXPECTED_MODEL_DIRECTORIES,
    validate_version_directories,
)


class RepositoryLayoutTests(unittest.TestCase):
    def _root_with_versions(self, versions: tuple[str, ...]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for model_name in EXPECTED_MODEL_DIRECTORIES:
            for version in versions:
                (root / model_name / version).mkdir(parents=True)
        return root

    def test_version_one_is_accepted(self) -> None:
        self.assertEqual(validate_version_directories(self._root_with_versions(("1",))), [])

    def test_version_zero_is_rejected(self) -> None:
        errors = validate_version_directories(self._root_with_versions(("0",)))
        self.assertTrue(any("invalid version directory" in error for error in errors))

    def test_zero_padded_version_is_rejected(self) -> None:
        errors = validate_version_directories(self._root_with_versions(("01",)))
        self.assertTrue(any("invalid version directory" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
