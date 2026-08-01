"""Append-only, schema-validated inference event logging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/inference-event.schema.json"
WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s\"']+")
POSIX_PATH = re.compile(r"(?<![\w:/])/(?:home|mnt|users|tmp|var)/[^\s\"']+")


class LogError(ValueError):
    pass


def sanitize_error(value: object) -> str:
    message = " ".join(str(value).split())
    message = WINDOWS_PATH.sub("<path>", message)
    message = POSIX_PATH.sub("<path>", message)
    return message[:1000] or "Inference failed"


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_event(event: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(event), key=lambda item: list(item.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise LogError(f"Invalid inference event at {path}: {errors[0].message}")


def append_event(path: Path, event: dict[str, Any]) -> None:
    validate_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
