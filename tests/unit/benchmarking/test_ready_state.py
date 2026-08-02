from __future__ import annotations

import json
import unittest

from benchmarks.run_benchmark import (
    CLIENT_CONTRACT_PATH,
    PAIR_CONTRACT_PATH,
    BenchmarkError,
    _restore,
    _unload_all,
    _validate_restorable_initial_state,
)


class FakeController:
    def __init__(self, ready: set[tuple[str, str]], versions: dict[str, list[str]]) -> None:
        self.ready = set(ready)
        self.versions = versions

    def ready_set(self) -> set[tuple[str, str]]:
        return set(self.ready)

    def unload(self, model: str) -> None:
        self.ready = {item for item in self.ready if item[0] != model}

    def load(self, model: str) -> None:
        self.ready.update((model, version) for version in self.versions[model])


class ReadyIsolationTests(unittest.TestCase):
    def test_full_ready_set_is_unloaded_and_restored(self) -> None:
        pair = json.loads(PAIR_CONTRACT_PATH.read_text(encoding="utf-8"))
        versions = {
            role["model"]: role["available_versions"]
            for role in (pair["baseline"], pair["optimized"])
        }
        initial = {(pair["baseline"]["model"], version) for version in pair["baseline"]["available_versions"]}
        controller = FakeController(initial, versions)
        _unload_all(controller, initial, 0.1)
        self.assertEqual(controller.ready_set(), set())
        _restore(controller, initial, 0.1)
        self.assertEqual(controller.ready_set(), initial)

    def test_partial_initial_model_state_is_rejected(self) -> None:
        contract = json.loads(CLIENT_CONTRACT_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(BenchmarkError):
            _validate_restorable_initial_state({("resnet50_onnx", "1")}, contract)


if __name__ == "__main__":
    unittest.main()
