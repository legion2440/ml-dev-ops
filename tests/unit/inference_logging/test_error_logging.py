from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from PIL import Image

from client.inference_client import CONFIG_PATH, CONTRACT_PATH, _run_inference
from client.logging.csv_export import read_events
from client.logging.writer import sanitize_error, validate_event
from tests.unit.inference_logging.helpers import error_event


class ErrorLoggingTests(unittest.TestCase):
    def test_failed_request_event_is_valid(self) -> None:
        validate_event(error_event())

    def test_host_paths_are_removed_from_errors(self) -> None:
        source = r"cannot open C:\Users\person\secret\input.jpg"
        sanitized = sanitize_error(source)
        self.assertNotIn("C:\\Users", sanitized)
        self.assertIn("<path>", sanitized)

    def test_failed_inference_appends_an_error_event(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            log_path = Path(directory) / "events.jsonl"
            Image.new("RGB", (32, 32)).save(image_path)
            transport = mock.Mock()
            transport.infer.side_effect = RuntimeError(f"failed at {image_path}")
            arguments = SimpleNamespace(
                model="resnet50_onnx",
                version="1",
                batch_size=1,
                input=str(image_path),
                protocol="http",
                timeout=1.0,
                no_auto_load=False,
                log_file=str(log_path),
                http_url=None,
                grpc_url=None,
                top_k=5,
            )
            with (
                mock.patch("client.inference_client._transport", return_value=transport),
                mock.patch("client.inference_client.RepositoryController") as controller,
            ):
                controller.return_value.ensure_ready.return_value = False
                with self.assertRaises(RuntimeError):
                    _run_inference(arguments, config, contract, "classification")
            events = read_events(log_path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["status"], "error")
            self.assertNotIn(directory, events[0]["error"])


if __name__ == "__main__":
    unittest.main()
