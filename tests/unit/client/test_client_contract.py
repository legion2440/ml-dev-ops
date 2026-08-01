from __future__ import annotations

import copy
import json
import unittest

from scripts.model_preparation import prepare_models


class ClientContractTests(unittest.TestCase):
    def test_generated_contract_is_current(self) -> None:
        manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
        tracked = json.loads(prepare_models.CLIENT_CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(tracked, prepare_models.render_client_contract(manifest))

    def test_manifest_contract_change_stales_generated_contract(self) -> None:
        manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
        tracked = json.loads(prepare_models.CLIENT_CONTRACT_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(manifest)
        changed["models"]["yolo11n_onnx"]["max_batch_size"] += 1
        self.assertNotEqual(tracked, prepare_models.render_client_contract(changed))

    def test_contract_excludes_model_repository_details(self) -> None:
        tracked = json.loads(prepare_models.CLIENT_CONTRACT_PATH.read_text(encoding="utf-8"))
        serialized = json.dumps(tracked, sort_keys=True)
        for forbidden in ("artifact", "model.onnx", "model.plan", "source_url", "weights"):
            self.assertNotIn(forbidden, serialized)

    def test_detection_contract_explicitly_has_no_objectness(self) -> None:
        tracked = json.loads(prepare_models.CLIENT_CONTRACT_PATH.read_text(encoding="utf-8"))
        semantics = tracked["models"]["yolo11n_onnx"]["output_semantics"]
        self.assertIs(semantics["has_objectness"], False)


if __name__ == "__main__":
    unittest.main()
