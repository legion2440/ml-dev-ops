from __future__ import annotations

import unittest

from scripts.model_preparation import prepare_models


class VersionPolicyTests(unittest.TestCase):
    def test_policies_match_declared_versions(self) -> None:
        spec = prepare_models.load_spec()
        for model in spec["models"].values():
            for serving in model["serving"].values():
                expected = sorted(int(version) for version in serving["versions"])
                self.assertEqual(serving["version_policy"]["specific"], expected)
        self.assertEqual(
            len(spec["models"]["resnet50"]["serving"]["onnx"]["versions"]), 2
        )


if __name__ == "__main__":
    unittest.main()
