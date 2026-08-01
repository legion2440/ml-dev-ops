"""Deterministic, sanitized image input discovery and decoding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    name: str
    sha256: str
    width: int
    height: int
    image: Image.Image

    def log_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


def discover_images(path: Path) -> list[Path]:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            (
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            ),
            key=lambda item: (item.name.casefold(), item.name),
        )
    else:
        raise InputError(f"Input does not exist: {path.name}")
    if not candidates:
        raise InputError("Input contains no supported JPG, JPEG, or PNG images")
    unsupported = [item.name for item in candidates if item.suffix.lower() not in SUPPORTED_SUFFIXES]
    if unsupported:
        raise InputError(f"Unsupported image extension: {unsupported[0]}")
    return candidates


def load_image(path: Path) -> LoadedImage:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise InputError(f"Cannot decode image: {path.name}") from error
    width, height = image.size
    if width < 1 or height < 1:
        raise InputError(f"Image has invalid dimensions: {path.name}")
    return LoadedImage(path, path.name, digest, width, height, image)


def batches(items: list[LoadedImage], batch_size: int) -> Iterable[list[LoadedImage]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
