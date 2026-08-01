from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.model_preparation import prepare_models


class HistoricalSmokeTests(unittest.TestCase):
    def _hashes(self) -> dict[str, str]:
        directory = prepare_models.REPOSITORY_ROOT / "docs/evidence/step-3"
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir()
            if path.is_file()
        }

    def test_check_is_read_only_and_runtime_mode_is_disabled(self) -> None:
        script = Path("deployment/triton/smoke_models.py")
        before = self._hashes()
        checked = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=prepare_models.REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(self._hashes(), before)
        runtime = subprocess.run(
            [sys.executable, str(script)],
            cwd=prepare_models.REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(runtime.returncode, 0)
        self.assertEqual(self._hashes(), before)


if __name__ == "__main__":
    unittest.main()
