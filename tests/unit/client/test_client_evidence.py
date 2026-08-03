from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.validate_client_evidence import (
    HISTORICAL_CLIENT_SEMANTICS_SHA256,
    _semantic_contract_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/client-runtime-evidence.schema.json"
EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-5/client-runtime.json"
CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"


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

    def test_schema_migration_preserves_historical_client_semantics(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            _semantic_contract_sha256(contract),
            HISTORICAL_CLIENT_SEMANTICS_SHA256,
        )
        changed = copy.deepcopy(contract)
        changed["models"]["resnet50_onnx"]["input"]["shape"][-1] += 1
        self.assertNotEqual(
            _semantic_contract_sha256(changed),
            HISTORICAL_CLIENT_SEMANTICS_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
