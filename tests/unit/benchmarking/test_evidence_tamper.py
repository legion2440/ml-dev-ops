from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import run_benchmark
from benchmarks.run_benchmark import (
    REPOSITORY_ROOT,
    benchmark_compatibility_projection,
    canonical_sha256,
    normalized_benchmark_compatibility_projection,
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
        stored = evidence["runtime_compatibility_projection"]
        self.assertEqual(
            normalized_benchmark_compatibility_projection(current),
            normalized_benchmark_compatibility_projection(stored),
        )
        self.assertEqual(
            canonical_sha256(stored),
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

    def _assert_behavioral_probe_change_is_incompatible(
        self, target: str, replacement: object
    ) -> None:
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        with patch(target, replacement):
            errors = validate_current_compatibility(evidence)
        self.assertTrue(any("incompatible" in error for error in errors))

    def test_pa_command_behavior_change_is_incompatible(self) -> None:
        original = run_benchmark.build_perf_analyzer_command

        def changed(*args: object, **kwargs: object) -> list[str]:
            return [*original(*args, **kwargs), "--changed-pa-semantics"]

        self._assert_behavioral_probe_change_is_incompatible(
            "benchmarks.run_benchmark.build_perf_analyzer_command", changed
        )

    def test_aggregation_behavior_change_is_incompatible(self) -> None:
        original = run_benchmark.summarize_paired_measurements

        def changed(*args: object, **kwargs: object) -> dict[str, object]:
            result = copy.deepcopy(original(*args, **kwargs))
            result["median_paired_improvement_pct"] = -999.0
            return result

        self._assert_behavioral_probe_change_is_incompatible(
            "benchmarks.run_benchmark.summarize_paired_measurements", changed
        )

    def test_guard_classification_behavior_change_is_incompatible(self) -> None:
        original = run_benchmark.recompute_guard

        def changed(*args: object, **kwargs: object) -> dict[str, object]:
            result = copy.deepcopy(original(*args, **kwargs))
            result["classification"] = "ERROR"
            return result

        self._assert_behavioral_probe_change_is_incompatible(
            "benchmarks.run_benchmark.recompute_guard", changed
        )

    def test_replacement_behavior_change_is_incompatible(self) -> None:
        original = run_benchmark.replacement_decision

        def changed(*args: object, **kwargs: object) -> dict[str, object]:
            result = copy.deepcopy(original(*args, **kwargs))
            result["action"] = "changed-replacement-semantics"
            return result

        self._assert_behavioral_probe_change_is_incompatible(
            "benchmarks.run_benchmark.replacement_decision", changed
        )


if __name__ == "__main__":
    unittest.main()
