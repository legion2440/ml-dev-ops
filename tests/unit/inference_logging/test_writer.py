from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from client.logging.writer import append_event
from tests.unit.inference_logging.helpers import event


class WriterTests(unittest.TestCase):
    def test_append_does_not_truncate_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_event(path, event())
            append_event(path, event())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertNotIn(b"\r\n", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
