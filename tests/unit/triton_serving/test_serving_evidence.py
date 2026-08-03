from __future__ import annotations

import copy
import hashlib
import json
import unittest

from scripts.validate_serving_evidence import (
    EVIDENCE_PATH,
    HISTORICAL_MANIFEST_PATH,
    HISTORICAL_SPEC_PATH,
    INTEGRITY_PATH,
    REPOSITORY_PATH,
    serving_semantic_projection,
    validate_evidence,
    validate_historical,
)


class ServingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(HISTORICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_historical_evidence_is_semantically_valid(self) -> None:
        self.assertEqual(validate_evidence(self.evidence, self.manifest), [])

    def test_historical_bundle_is_self_contained_and_read_only(self) -> None:
        paths = (
            EVIDENCE_PATH,
            REPOSITORY_PATH,
            HISTORICAL_MANIFEST_PATH,
            HISTORICAL_SPEC_PATH,
            INTEGRITY_PATH,
        )
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(validate_historical(), [])
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(after, before)

    def test_host_provenance_is_not_serving_semantics(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["generated_at_utc"] = "2099-01-01T00:00:00Z"
        changed["build"]["gpu"]["compute_capability"] = "8.6"
        changed["models"]["resnet50_tensorrt"]["compute_capability"] = "8.6"
        changed["models"]["resnet50_tensorrt"]["versions"]["1"]["artifact"][
            "sha256"
        ] = "0" * 64
        self.assertEqual(
            serving_semantic_projection(changed),
            serving_semantic_projection(self.manifest),
        )

    def test_io_change_is_serving_incompatible(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["models"]["resnet50_tensorrt"]["input"]["shape"][-1] += 1
        self.assertNotEqual(
            serving_semantic_projection(changed),
            serving_semantic_projection(self.manifest),
        )

    def test_missing_protocol_is_rejected(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["protocols"].pop("grpc")
        self.assertTrue(validate_evidence(changed, self.manifest))

    def test_equal_execution_and_request_counts_are_rejected(self) -> None:
        changed = copy.deepcopy(self.evidence)
        item = next(iter(changed["dynamic_batching"].values()))
        item["execution_count_delta"] = item["requests"]
        self.assertTrue(any("does not prove" in error for error in validate_evidence(changed, self.manifest)))

    def test_wrong_selected_version_and_cleanup_are_rejected(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["version_switching"]["default_with_tracked_policy"] = "1"
        changed["final_repository_ready_models"] = ["resnet50_onnx"]
        errors = validate_evidence(changed, self.manifest)
        self.assertTrue(any("version 2" in error for error in errors))
        self.assertTrue(any("cleanup" in error for error in errors))

    def test_ready_version_after_cleanup_is_rejected(self) -> None:
        changed = copy.deepcopy(self.evidence)
        readiness = changed["model_readiness_after_cleanup"]["resnet50_onnx"]
        readiness["versions"]["2"] = True
        self.assertTrue(
            any(
                "remained ready" in error
                for error in validate_evidence(changed, self.manifest)
            )
        )

    def test_attempt_count_and_final_attempt_are_enforced(self) -> None:
        changed = copy.deepcopy(self.evidence)
        item = next(iter(changed["dynamic_batching"].values()))
        item["attempts_used"] += 1
        self.assertTrue(
            any(
                "attempts_used" in error
                for error in validate_evidence(changed, self.manifest)
            )
        )
        changed = copy.deepcopy(self.evidence)
        item = next(iter(changed["dynamic_batching"].values()))
        item["attempts"][-1]["passed"] = False
        self.assertTrue(
            any(
                "final batching attempt" in error
                for error in validate_evidence(changed, self.manifest)
            )
        )


if __name__ == "__main__":
    unittest.main()
