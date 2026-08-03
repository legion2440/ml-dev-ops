from __future__ import annotations

import json
import unittest

from benchmarks.run_benchmark import PAIR_CONTRACT_PATH
from scripts.model_preparation.prepare_models import (
    MANIFEST_PATH,
    render_benchmark_pair_contract,
)


class GeneratedBenchmarkPairTests(unittest.TestCase):
    def test_pair_is_rendered_from_current_manifest(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        actual = json.loads(PAIR_CONTRACT_PATH.read_text(encoding="utf-8"))
        first = render_benchmark_pair_contract(manifest)
        second = render_benchmark_pair_contract(manifest)
        self.assertEqual(first, second)
        self.assertEqual(actual, first)

    def test_pair_exposes_lineage_without_repository_artifact_paths(self) -> None:
        pair = json.loads(PAIR_CONTRACT_PATH.read_text(encoding="utf-8"))
        rendered = json.dumps(pair)
        self.assertNotIn("artifact", rendered)
        self.assertNotIn("models/", rendered)
        self.assertTrue(pair["parity_requirement"]["required"])
        self.assertNotIn("status", pair["parity_requirement"])
        self.assertNotIn("max_abs_error", pair["parity_requirement"])
        self.assertEqual(pair["baseline"]["io_precision"], pair["optimized"]["io_precision"])


if __name__ == "__main__":
    unittest.main()
