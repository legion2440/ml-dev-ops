from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from client.input_loader import LoadedImage

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def contract(model: str) -> dict:
    value = json.loads(
        (REPOSITORY_ROOT / "shared/client-model-contracts.json").read_text(encoding="utf-8")
    )
    return value["models"][model]


def loaded_image(width: int = 320, height: int = 160) -> LoadedImage:
    return LoadedImage(
        path=Path("image.jpg"),
        name="image.jpg",
        sha256="0" * 64,
        width=width,
        height=height,
        image=Image.new("RGB", (width, height), (128, 64, 32)),
    )
