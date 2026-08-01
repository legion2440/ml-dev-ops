from __future__ import annotations

import copy


def event() -> dict:
    return {
        "schema_version": 1,
        "timestamp_utc": "2026-08-01T00:00:00Z",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "model": "resnet50_onnx",
        "requested_version": "1",
        "resolved_version": "1",
        "protocol": "http",
        "inputs": [
            {
                "name": "image.jpg",
                "sha256": "0" * 64,
                "width": 10,
                "height": 20,
            }
        ],
        "batch_size": 1,
        "timing_ms": {
            "preprocessing": 1.0,
            "request": 2.0,
            "postprocessing": 3.0,
            "total": 6.0,
        },
        "status": "success",
        "predictions": [{"input_name": "image.jpg", "items": []}],
        "error": None,
    }


def error_event(message: str = "failed") -> dict:
    value = copy.deepcopy(event())
    value["status"] = "error"
    value["resolved_version"] = None
    value["predictions"] = None
    value["error"] = message
    return value
