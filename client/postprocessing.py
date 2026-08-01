"""Stable classification and class-aware YOLO postprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

from client.preprocessing import LetterboxMetadata


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def classification_predictions(
    logits: np.ndarray, labels: list[str], top_k: int
) -> list[list[dict[str, Any]]]:
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError("Classification output does not match client contract")
    if not 1 <= top_k <= len(labels):
        raise ValueError(f"top_k must be between 1 and {len(labels)}")
    probabilities = stable_softmax(logits.astype(np.float64, copy=False))
    results: list[list[dict[str, Any]]] = []
    for row in probabilities:
        indices = np.argsort(-row, kind="stable")[:top_k]
        results.append(
            [
                {
                    "rank": rank,
                    "class_id": int(class_id),
                    "label": labels[int(class_id)],
                    "probability": float(row[class_id]),
                }
                for rank, class_id in enumerate(indices, 1)
            ]
        )
    return results


def box_iou(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    x1 = np.maximum(reference[0], candidates[:, 0])
    y1 = np.maximum(reference[1], candidates[:, 1])
    x2 = np.minimum(reference[2], candidates[:, 2])
    y2 = np.minimum(reference[3], candidates[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    reference_area = max(0.0, reference[2] - reference[0]) * max(
        0.0, reference[3] - reference[1]
    )
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0, candidates[:, 3] - candidates[:, 1]
    )
    union = reference_area + candidate_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        class_count = 0
        while order.size and class_count < max_detections:
            current = int(order[0])
            kept.append(current)
            class_count += 1
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[box_iou(boxes[current], boxes[remaining]) <= iou_threshold]
    kept.sort(key=lambda index: (-float(scores[index]), index))
    return kept[:max_detections]


def detection_predictions(
    output: np.ndarray,
    metadata: list[LetterboxMetadata],
    labels: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[list[dict[str, Any]]]:
    if output.ndim != 3 or output.shape[0] != len(metadata) or output.shape[1] != len(labels) + 4:
        raise ValueError("Detection output does not match client contract")
    results: list[list[dict[str, Any]]] = []
    for raw, geometry in zip(output, metadata, strict=True):
        candidates = raw.T
        class_scores = candidates[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        selected = np.flatnonzero(scores >= confidence_threshold)
        xywh = candidates[selected, :4]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        if boxes.size:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry.pad_x) / geometry.scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry.pad_y) / geometry.scale
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, geometry.original_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, geometry.original_height)
        valid = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        boxes = boxes[valid]
        selected = selected[valid]
        chosen = class_aware_nms(
            boxes,
            scores[selected],
            class_ids[selected],
            iou_threshold,
            max_detections,
        )
        results.append(
            [
                {
                    "class_id": int(class_ids[selected[index]]),
                    "label": labels[int(class_ids[selected[index]])],
                    "confidence": float(scores[selected[index]]),
                    "box_xyxy": [float(value) for value in boxes[index]],
                }
                for index in chosen
            ]
        )
    return results
