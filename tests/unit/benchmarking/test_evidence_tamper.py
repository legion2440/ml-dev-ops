from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from benchmarks.run_benchmark import REPOSITORY_ROOT
from scripts.validate_benchmark_evidence import validate


class BenchmarkEvidenceTamperTests(unittest.TestCase):
    def test_modified_aggregate_is_rejected(self) -> None:
        evidence = REPOSITORY_ROOT / "docs/evidence/step-6/benchmark-runtime.json"
        if not evidence.is_file():
            self.skipTest("runtime benchmark evidence is not generated yet")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY_ROOT / "benchmarks/results", root / "benchmarks/results")
            (root / "benchmarks").mkdir(exist_ok=True)
            shutil.copy2(REPOSITORY_ROOT / "benchmarks/report.md", root / "benchmarks/report.md")
            target_evidence = root / "docs/evidence/step-6"
            target_evidence.mkdir(parents=True)
            shutil.copy2(evidence, target_evidence / evidence.name)
            comparison = root / "benchmarks/results/comparison.csv"
            comparison.write_text(
                comparison.read_text(encoding="utf-8").replace("true", "false", 1),
                encoding="utf-8",
                newline="\n",
            )
            errors = validate(root)
            self.assertTrue(any("comparison" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
