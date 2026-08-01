from __future__ import annotations

import copy
import json
import unittest

from scripts.model_preparation import prepare_models
from shared.triton_model_config import render_load_config_json, render_pbtxt


class ConfigRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()

    def test_tracked_and_runtime_renderers_share_complete_config(self) -> None:
        for name in prepare_models.SERVING_MODELS:
            config = prepare_models.serving_model_config(self.spec, name)
            tracked = (prepare_models.REPOSITORY_ROOT / "models" / name / "config.pbtxt").read_text(encoding="utf-8")
            self.assertEqual(tracked, render_pbtxt(config))
            versions = config["version_policy"]["specific"]["versions"]
            runtime = json.loads(render_load_config_json(config, versions)["parameters"]["config"])
            self.assertEqual(runtime, config)

    def test_scheduling_change_makes_generated_pbtxt_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        serving = changed["models"]["resnet50"]["serving"]["onnx"]
        serving["scheduling"]["dynamic_batching"]["max_queue_delay_microseconds"] += 1
        self.assertNotEqual(
            prepare_models.render_config(changed, serving["name"]),
            (prepare_models.REPOSITORY_ROOT / serving["config_path"]).read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
