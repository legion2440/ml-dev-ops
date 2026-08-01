"""Contract-driven ResNet and YOLO image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from client.input_loader import LoadedImage


@dataclass(frozen=True)
class LetterboxMetadata:
    original_width: int
    original_height: int
    scale: float
    pad_x: float
    pad_y: float


def _normalized_chw(image: Image.Image, scale: list[float]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = (float(value) for value in scale)
    array = low + (array / 255.0) * (high - low)
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


def preprocess_classification(
    images: list[LoadedImage], contract: dict[str, Any]
) -> np.ndarray:
    preprocessing = contract["preprocessing"]
    if preprocessing["channel_order"] != "RGB" or preprocessing["tensor_layout"] != "CHW":
        raise ValueError("Unsupported classification color or tensor layout")
    resize = int(preprocessing["resize"])
    crop = int(preprocessing["center_crop"])
    mean = np.asarray(preprocessing["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(preprocessing["std"], dtype=np.float32).reshape(3, 1, 1)
    tensors: list[np.ndarray] = []
    for item in images:
        width, height = item.image.size
        if width <= height:
            resized_width = resize
            resized_height = int(resize * height / width)
        else:
            resized_height = resize
            resized_width = int(resize * width / height)
        resized = item.image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        left = int(round((resized.width - crop) / 2.0))
        top = int(round((resized.height - crop) / 2.0))
        cropped = resized.crop((left, top, left + crop, top + crop))
        tensor = _normalized_chw(cropped, preprocessing["scale"])
        tensors.append((tensor - mean) / std)
    return np.ascontiguousarray(np.stack(tensors), dtype=np.float32)


def preprocess_detection(
    images: list[LoadedImage], contract: dict[str, Any]
) -> tuple[np.ndarray, list[LetterboxMetadata]]:
    preprocessing = contract["preprocessing"]
    if (
        preprocessing["channel_order"] != "RGB"
        or preprocessing["tensor_layout"] != "CHW"
        or preprocessing["resize_mode"] != "letterbox"
        or preprocessing["letterbox_center"] is not True
    ):
        raise ValueError("Unsupported detection color, layout, or resize mode")
    target_height, target_width = (int(value) for value in preprocessing["resize"])
    if [target_height, target_width] != contract["input"]["shape"][-2:]:
        raise ValueError("Detection resize and input tensor dimensions differ")
    padding_value = int(preprocessing["padding_value"])
    tensors: list[np.ndarray] = []
    metadata: list[LetterboxMetadata] = []
    for item in images:
        scale = min(target_width / item.width, target_height / item.height)
        resized_width = max(1, round(item.width * scale))
        resized_height = max(1, round(item.height * scale))
        resized = item.image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        canvas = Image.new("RGB", (target_width, target_height), (padding_value,) * 3)
        canvas.paste(resized, (left, top))
        tensors.append(_normalized_chw(canvas, preprocessing["scale"]))
        metadata.append(
            LetterboxMetadata(
                original_width=item.width,
                original_height=item.height,
                scale=scale,
                pad_x=float(left),
                pad_y=float(top),
            )
        )
    return np.ascontiguousarray(np.stack(tensors), dtype=np.float32), metadata
