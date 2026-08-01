"""Deterministic CSV projection of primary JSONL inference events."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from client.logging.writer import LogError, validate_event

CSV_HEADER = [
    "timestamp_utc",
    "request_id",
    "model",
    "requested_version",
    "resolved_version",
    "protocol",
    "input_count",
    "input_names",
    "batch_size",
    "preprocessing_ms",
    "request_ms",
    "postprocessing_ms",
    "total_ms",
    "status",
    "prediction_summary_json",
    "error",
]


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise LogError(f"Inference log does not exist: {path.name}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise LogError(f"Invalid JSONL at line {line_number}") from error
        if not isinstance(event, dict):
            raise LogError(f"JSONL line {line_number} must be an object")
        validate_event(event)
        events.append(event)
    return events


def _row(event: dict[str, Any]) -> dict[str, Any]:
    timing = event["timing_ms"]
    return {
        "timestamp_utc": event["timestamp_utc"],
        "request_id": event["request_id"],
        "model": event["model"],
        "requested_version": event["requested_version"] or "",
        "resolved_version": event["resolved_version"] or "",
        "protocol": event["protocol"],
        "input_count": len(event["inputs"]),
        "input_names": "|".join(item["name"] for item in event["inputs"]),
        "batch_size": event["batch_size"],
        "preprocessing_ms": timing["preprocessing"],
        "request_ms": timing["request"],
        "postprocessing_ms": timing["postprocessing"],
        "total_ms": timing["total"],
        "status": event["status"],
        "prediction_summary_json": json.dumps(
            event["predictions"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if event["predictions"] is not None
        else "",
        "error": event["error"] or "",
    }


def export_csv(source: Path, destination: Path) -> int:
    events = read_events(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_row(event) for event in events)
    temporary.replace(destination)
    return len(events)
