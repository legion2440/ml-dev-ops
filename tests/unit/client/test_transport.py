from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from client.transport import HttpTransport


class _Result:
    def as_numpy(self, name: str) -> np.ndarray:
        return np.ones((1, 2), dtype=np.float32)

    def get_response(self) -> dict[str, str]:
        return {"model_name": "model", "model_version": "1"}


class TransportTests(unittest.TestCase):
    def test_http_inference_uses_binary_tensors_and_microsecond_timeout(self) -> None:
        transport = HttpTransport.__new__(HttpTransport)
        transport._timeout = 2.5
        transport._client = mock.Mock()
        transport._client.infer.return_value = _Result()
        item = mock.Mock()
        output = mock.Mock()
        module = mock.Mock()
        module.InferInput.return_value = item
        module.InferRequestedOutput.return_value = output
        transport._module = module
        result = transport.infer(
            model="model",
            version="1",
            input_name="input",
            output_name="output",
            tensor=np.ones((1, 3), dtype=np.float32),
            request_id="request",
        )
        item.set_data_from_numpy.assert_called_once()
        self.assertTrue(item.set_data_from_numpy.call_args.kwargs["binary_data"])
        self.assertEqual(transport._client.infer.call_args.kwargs["timeout"], 2_500_000)
        self.assertEqual(result.model_version, "1")


if __name__ == "__main__":
    unittest.main()
