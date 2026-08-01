from __future__ import annotations

import unittest

import numpy as np

from client.postprocessing import class_aware_nms


class NmsTests(unittest.TestCase):
    def test_nms_suppresses_only_within_same_class(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1])
        self.assertEqual(class_aware_nms(boxes, scores, classes, 0.5, 10), [0, 2])

    def test_global_limit_keeps_highest_score_across_classes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.5, 0.9], dtype=np.float32)
        classes = np.array([0, 1])
        self.assertEqual(class_aware_nms(boxes, scores, classes, 0.5, 1), [1])


if __name__ == "__main__":
    unittest.main()
