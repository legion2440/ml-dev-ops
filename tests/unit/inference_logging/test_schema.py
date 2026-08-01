from __future__ import annotations

import copy
import unittest

from client.logging.writer import LogError, validate_event
from tests.unit.inference_logging.helpers import event


class SchemaTests(unittest.TestCase):
    def test_success_event_is_valid(self) -> None:
        validate_event(event())

    def test_absolute_input_name_is_rejected(self) -> None:
        value = copy.deepcopy(event())
        value["inputs"][0]["name"] = r"C:\Users\person\image.jpg"
        with self.assertRaises(LogError):
            validate_event(value)

    def test_invalid_request_uuid_is_rejected(self) -> None:
        value = copy.deepcopy(event())
        value["request_id"] = "not-a-uuid"
        with self.assertRaises(LogError):
            validate_event(value)


if __name__ == "__main__":
    unittest.main()
