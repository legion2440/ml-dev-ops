from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from client.logging.csv_export import CSV_HEADER, export_csv
from client.logging.writer import append_event
from tests.unit.inference_logging.helpers import event, error_event


class CsvExportTests(unittest.TestCase):
    def test_csv_rows_equal_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            destination = Path(directory) / "events.csv"
            append_event(source, event())
            append_event(source, error_event())
            self.assertEqual(export_csv(source, destination), 2)
            with destination.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], CSV_HEADER)
            self.assertEqual(len(rows) - 1, 2)


if __name__ == "__main__":
    unittest.main()
