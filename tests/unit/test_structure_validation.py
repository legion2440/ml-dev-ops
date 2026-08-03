"""Regression tests for step 1 repository structure contracts."""

from __future__ import annotations

import unittest

from scripts.validate_structure import REQUIRED_DIRECTORIES, REQUIRED_FILES


class RequiredDirectoryContractTests(unittest.TestCase):
    def test_benchmark_raw_artifacts_use_canonical_results_path(self) -> None:
        self.assertIn("benchmarks/results/raw", REQUIRED_DIRECTORIES)
        self.assertNotIn("benchmarks/raw", REQUIRED_DIRECTORIES)

    def test_portability_evidence_and_historical_snapshots_are_required(self) -> None:
        self.assertIn("docs/evidence/portability", REQUIRED_DIRECTORIES)
        for path in (
            "docs/evidence/step-4/runtime-integrity.json",
            "docs/evidence/step-4/runtime-model-manifest.json",
            "docs/evidence/step-4/runtime-model-spec.yaml",
            "docs/evidence/portability/build-record.json",
            "docs/evidence/portability/serving-runtime.json",
            "docs/evidence/portability/repository-versions.txt",
        ):
            self.assertIn(path, REQUIRED_FILES)


if __name__ == "__main__":
    unittest.main()
