from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmarks.run_benchmark import (
    REPOSITORY_ROOT,
    benchmark_compatibility_projection,
    canonical_sha256,
)
from scripts.validate_benchmark_evidence import (
    validate,
    validate_current_compatibility,
    validate_historical,
)


class BenchmarkEvidenceTamperTests(unittest.TestCase):
    evidence = REPOSITORY_ROOT / "docs/evidence/step-6/benchmark-runtime.json"

    @classmethod
    def _copy_bundle(cls, root: Path) -> None:
        shutil.copytree(REPOSITORY_ROOT / "benchmarks/results", root / "benchmarks/results")
        (root / "benchmarks").mkdir(exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / "benchmarks/report.md", root / "benchmarks/report.md")
        target_evidence = root / "docs/evidence/step-6"
        target_evidence.mkdir(parents=True)
        shutil.copy2(cls.evidence, target_evidence / cls.evidence.name)

    def test_modified_aggregate_is_rejected(self) -> None:
        if not self.evidence.is_file():
            self.skipTest("runtime benchmark evidence is not generated yet")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_bundle(root)
            comparison = root / "benchmarks/results/comparison.csv"
            comparison.write_text(
                comparison.read_text(encoding="utf-8").replace("true", "false", 1),
                encoding="utf-8",
                newline="\n",
            )
            errors = validate(root)
            self.assertTrue(any("comparison" in error for error in errors))

    def test_historical_source_manifest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_bundle(root)
            path = root / "docs/evidence/step-6/benchmark-runtime.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["runtime_source_hashes"]["docker-compose.yml"] = "0" * 64
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            errors = validate_historical(root)
        self.assertTrue(any("source manifest hash" in error for error in errors))

    def test_runtime_fingerprint_and_semantic_projection_are_restored(self) -> None:
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(
            evidence["runtime_source_fingerprint_sha256"],
            "82e10584916355dfd2332055dc785a093b95d5265d37b62c9b7388fc274f4f62",
        )
        current = benchmark_compatibility_projection()
        self.assertEqual(current, evidence["runtime_compatibility_projection"])
        self.assertEqual(
            canonical_sha256(current),
            evidence["runtime_compatibility_projection_sha256"],
        )

    def test_critical_contract_change_does_not_invalidate_history(self) -> None:
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(validate_historical(), [])
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            for relative in (
                ".env.example",
                "docker-compose.yml",
                "benchmarks/configs/benchmark.json",
                "shared/benchmark-model-pair.json",
            ):
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPOSITORY_ROOT / relative, target)
            config_path = source_root / "benchmarks/configs/benchmark.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["acceptance"]["minimum_directional_pairs"] = 4
            config_path.write_text(
                json.dumps(config, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            errors = validate_current_compatibility(evidence, source_root)
        self.assertTrue(any("incompatible" in error for error in errors))

    def test_check_modes_are_read_only(self) -> None:
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        paths = [self.evidence]
        paths.extend(
            REPOSITORY_ROOT / item["path"]
            for item in evidence["artifacts"].values()
        )
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(validate(), [])
        self.assertEqual(validate(historical_only=True), [])
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
