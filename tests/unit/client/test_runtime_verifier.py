from __future__ import annotations

import unittest
from unittest import mock

from client.verify_runtime import _wait_for_server_health


class RuntimeVerifierTests(unittest.TestCase):
    @mock.patch("client.verify_runtime.time.sleep")
    def test_health_wait_retries_transitional_cleanup_state(self, sleep) -> None:
        transport = mock.Mock()
        transport.health.side_effect = [
            {"live": True, "ready": False},
            {"live": True, "ready": True},
        ]

        _wait_for_server_health(transport, 1.0)

        sleep.assert_called_once_with(0.25)


if __name__ == "__main__":
    unittest.main()
