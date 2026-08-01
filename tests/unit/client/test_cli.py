from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from client.inference_client import main


class CliTests(unittest.TestCase):
    def test_no_arguments_prints_help_and_succeeds(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("classify", output.getvalue())

    def test_oversized_batch_is_rejected_before_network(self) -> None:
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main(
                [
                    "detect",
                    "missing-directory",
                    "--model",
                    "yolo11n_onnx",
                    "--batch-size",
                    "999",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("maximum", error.getvalue())
        self.assertNotIn("does not exist", error.getvalue())

    @patch("client.inference_client.export_csv", return_value=0)
    def test_export_logs_accepts_canonical_output_csv_flag(self, export_csv) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "export-logs",
                    "--input-log",
                    "logs/source.jsonl",
                    "--output-csv",
                    "logs/result.csv",
                ]
            )
        self.assertEqual(code, 0)
        export_csv.assert_called_once()
        self.assertIn("logs/result.csv", output.getvalue())


if __name__ == "__main__":
    unittest.main()
