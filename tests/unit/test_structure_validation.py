"""Regression tests for step 1 repository structure contracts."""

from __future__ import annotations

import unittest

from scripts.validate_structure import REQUIRED_DIRECTORIES


class RequiredDirectoryContractTests(unittest.TestCase):
    def test_benchmark_raw_artifacts_use_canonical_results_path(self) -> None:
        self.assertIn("benchmarks/results/raw", REQUIRED_DIRECTORIES)
        self.assertNotIn("benchmarks/raw", REQUIRED_DIRECTORIES)


if __name__ == "__main__":
    unittest.main()
