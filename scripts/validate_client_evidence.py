"""Daemon-free semantic and freshness validation of step 5 runtime evidence."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from client.logging.csv_export import CSV_HEADER, read_events
from client.verify_runtime import (
    CONTRACT_PATH,
    CSV_PATH,
    EVIDENCE_PATH,
    JSONL_PATH,
    PREDICTIONS_PATH,
    REQUIREMENTS_PATH,
    SAMPLES_MANIFEST_PATH,
    sha256,
)

SCHEMA_PATH = REPOSITORY_ROOT / "schemas/client-runtime-evidence.schema.json"
WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)")
POSIX_HOST_PATH = re.compile(r"(?<![\w:/])/(?:home|mnt|users|tmp|var)/")
HISTORICAL_CLIENT_CONTRACT_SHA256 = (
    "8bd13e17773b9c90a756d0d2f4d9971fb207c55b89601a234892fbdc639ef5a3"
)
HISTORICAL_SOURCE_FINGERPRINT_SHA256 = (
    "6a2d1a30c8114ac6eb708fd6465d167c5fe8ac04166d2498f3b66713e4cc0bb4"
)
HISTORICAL_CLIENT_SEMANTICS_SHA256 = (
    "2beb7aed5a3b1b34e3a33fa10240305e6e9f41bd98afd0cf9555fd4f87b8e0c9"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _semantic_contract_sha256(contract: dict[str, Any]) -> str:
    """Hash host- and schema-independent client model behavior."""
    import hashlib

    payload = json.dumps(
        contract.get("models"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _case_events(
    events: list[dict[str, Any]], model: str, version: str, protocol: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["model"] == model
        and event["requested_version"] == version
        and event["resolved_version"] == version
        and event["protocol"] == protocol
    ]


def _validate_predictions(
    events: list[dict[str, Any]], contract: dict[str, Any], errors: list[str]
) -> int:
    detections = 0
    for event in events:
        entry = contract["models"].get(event["model"])
        if entry is None:
            errors.append("runtime evidence contains a model outside the client contract")
            continue
        labels = entry["labels"]
        if event["status"] != "success":
            errors.append("runtime evidence contains a failed inference event")
            continue
        dimensions = {item["name"]: (item["width"], item["height"]) for item in event["inputs"]}
        if event["batch_size"] != len(event["inputs"]):
            errors.append("event batch size does not match its input count")
        if len(event["predictions"]) != len(event["inputs"]):
            errors.append("prediction count does not match the request input count")
        for prediction in event["predictions"]:
            if prediction["input_name"] not in dimensions:
                errors.append("prediction input name is outside its request")
                continue
            width, height = dimensions[prediction["input_name"]]
            classification_ranks: list[int] = []
            for item in prediction["items"]:
                class_id = item["class_id"]
                if not 0 <= class_id < len(labels) or item["label"] != labels[class_id]:
                    errors.append("prediction class ID or label differs from the client contract")
                    continue
                if "probability" in item:
                    classification_ranks.append(item["rank"])
                    if not math.isfinite(item["probability"]) or not 0 <= item["probability"] <= 1:
                        errors.append("classification probability is invalid")
                else:
                    detections += 1
                    confidence = item["confidence"]
                    box = item["box_xyxy"]
                    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        errors.append("detection confidence is invalid")
                    if not all(math.isfinite(value) for value in box):
                        errors.append("detection box is not finite")
                    elif not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                        errors.append("detection box is outside the source image")
            if classification_ranks and classification_ranks != list(
                range(1, len(classification_ranks) + 1)
            ):
                errors.append("classification ranks are not consecutive")
    return detections


def validate() -> list[str]:
    required = (EVIDENCE_PATH, JSONL_PATH, CSV_PATH, PREDICTIONS_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return ["missing step 5 evidence: " + ", ".join(missing)]
    evidence = _json(EVIDENCE_PATH)
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = [
        f"evidence.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(evidence),
            key=lambda item: list(item.path),
        )
    ]
    expected_hashes = {
        "client_contract_sha256": HISTORICAL_CLIENT_CONTRACT_SHA256,
        "samples_manifest_sha256": sha256(SAMPLES_MANIFEST_PATH),
        "requirements_sha256": sha256(REQUIREMENTS_PATH),
        "source_fingerprint_sha256": HISTORICAL_SOURCE_FINGERPRINT_SHA256,
    }
    for field, expected in expected_hashes.items():
        if evidence.get(field) != expected:
            errors.append(f"{field} is stale")
    logging = evidence.get("logging", {})
    for field, path in (
        ("jsonl_sha256", JSONL_PATH),
        ("csv_sha256", CSV_PATH),
        ("predictions_sha256", PREDICTIONS_PATH),
    ):
        if logging.get(field) != sha256(path):
            errors.append(f"logging.{field} is stale")
    events = read_events(JSONL_PATH)
    contract = _json(CONTRACT_PATH)
    if _semantic_contract_sha256(contract) != HISTORICAL_CLIENT_SEMANTICS_SHA256:
        errors.append("current client contract is incompatible with Step 5 semantics")
    if logging.get("jsonl_events") != len(events):
        errors.append("JSONL event count is stale")
    with CSV_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != CSV_HEADER:
            errors.append("CSV header differs from the fixed export contract")
    if logging.get("csv_rows") != len(rows) or len(rows) != len(events):
        errors.append("CSV and JSONL row counts differ")
    for row, event in zip(rows, events):
        if (
            row["request_id"] != event["request_id"]
            or row["model"] != event["model"]
            or row["status"] != event["status"]
            or int(row["input_count"]) != len(event["inputs"])
            or int(row["batch_size"]) != event["batch_size"]
        ):
            errors.append("CSV row content differs from its JSONL source event")
    samples = _json(SAMPLES_MANIFEST_PATH)["samples"]
    sample_names = {entry["filename"] for entry in samples}
    if evidence.get("samples_count") != len(samples):
        errors.append("sample count is stale")
    http_classification = _case_events(events, "resnet50_onnx", "1", "http")
    if [event["batch_size"] for event in http_classification] != [4, 4, 2]:
        errors.append("full ResNet HTTP sample run must use batches 4, 4, 2")
    if {item["name"] for event in http_classification for item in event["inputs"]} != sample_names:
        errors.append("full ResNet HTTP run does not cover every sample")
    if len(_case_events(events, "resnet50_onnx", "2", "grpc")) != 1:
        errors.append("ResNet ONNX v2 gRPC case is missing")
    if len(_case_events(events, "resnet50_tensorrt", "1", "http")) != 1:
        errors.append("ResNet TensorRT HTTP case is missing")
    http_detection = _case_events(events, "yolo11n_onnx", "1", "http")
    if [event["batch_size"] for event in http_detection] != [2, 2, 2, 2, 2]:
        errors.append("full YOLO HTTP sample run must use five batches of 2")
    if {item["name"] for event in http_detection for item in event["inputs"]} != sample_names:
        errors.append("full YOLO HTTP run does not cover every sample")
    if len(_case_events(events, "yolo11n_onnx", "1", "grpc")) != 1:
        errors.append("YOLO gRPC case is missing")
    detections = _validate_predictions(events, contract, errors)
    if detections < 1 or evidence.get("detection", {}).get("detections") != detections:
        errors.append("detection count is missing or stale")
    classification_counts = Counter(
        len(prediction["items"])
        for event in events
        if event["model"].startswith("resnet50")
        for prediction in event["predictions"]
    )
    if set(classification_counts) != {5}:
        errors.append("classification runtime cases must contain exactly top-5 predictions")
    if evidence.get("initial_ready") != evidence.get("final_ready"):
        errors.append("initial and final READY states differ")
    for path in required:
        content = path.read_text(encoding="utf-8", errors="replace")
        if WINDOWS_PATH.search(content) or POSIX_HOST_PATH.search(content):
            errors.append(f"host-specific path leaked into {path.name}")
        if b"\r\n" in path.read_bytes():
            errors.append(f"evidence is not LF-only: {path.name}")
    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Client evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    print("[OK] Step 5 client runtime evidence is current and semantically valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
