from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/client-runtime-evidence.schema.json"
EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-5/client-runtime.json"


class ClientEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(schema)
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_current_runtime_evidence_matches_schema(self) -> None:
        self.assertEqual(list(self.validator.iter_errors(self.evidence)), [])

    def test_legacy_mixed_case_tensorrt_key_is_rejected(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["classification"]["tensorRT"] = changed["classification"].pop(
            "tensorrt"
        )

        self.assertTrue(list(self.validator.iter_errors(changed)))


if __name__ == "__main__":
    unittest.main()
