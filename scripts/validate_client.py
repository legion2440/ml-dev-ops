"""Daemon-free validation for the production client and persistent logging."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from client.logging.csv_export import CSV_HEADER
from scripts.model_preparation import prepare_models

CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"
CONTRACT_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/client-model-contracts.schema.json"
EVENT_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/inference-event.schema.json"
CONFIG_PATH = REPOSITORY_ROOT / "client/client-config.yaml"
SAMPLE_DIRECTORY = REPOSITORY_ROOT / "client/samples"
SAMPLE_MANIFEST_PATH = SAMPLE_DIRECTORY / "manifest.json"
REQUIREMENTS_PATH = REPOSITORY_ROOT / "requirements.txt"
REQUIRED_FILES = {
    "client/inference_client.py",
    "client/transport.py",
    "client/preprocessing.py",
    "client/postprocessing.py",
    "client/input_loader.py",
    "client/client-config.yaml",
    "client/logging/writer.py",
    "client/logging/csv_export.py",
    "client/verify_runtime.py",
    "schemas/client-model-contracts.schema.json",
    "schemas/inference-event.schema.json",
    "schemas/client-runtime-evidence.schema.json",
    "shared/client-model-contracts.json",
}
EXPECTED_REQUIREMENTS = {
    "jsonschema",
    "pyyaml",
    "numpy",
    "pillow",
    "tritonclient",
}
REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[(?P<extras>[^]]+)\])?==(?P<version>[^\s]+)$"
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _schema_errors(instance: dict[str, Any], schema_path: Path, label: str) -> list[str]:
    schema = _json(schema_path)
    Draft202012Validator.check_schema(schema)
    return [
        f"{label}.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path)
        )
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_requirements(errors: list[str]) -> None:
    parsed: dict[str, tuple[str | None, str]] = {}
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT.fullmatch(line)
        if not match:
            errors.append(f"host requirement is not exactly pinned: {line}")
            continue
        name = match.group("name").lower().replace("_", "-")
        if name in parsed:
            errors.append(f"duplicate host requirement: {name}")
        parsed[name] = (match.group("extras"), match.group("version"))
    if set(parsed) != EXPECTED_REQUIREMENTS:
        errors.append("requirements.txt must contain exactly the validator/client direct dependencies")
    extras = parsed.get("tritonclient", (None, ""))[0]
    if extras is None or {item.strip() for item in extras.split(",")} != {"http", "grpc"}:
        errors.append("tritonclient must enable exactly the http and grpc extras")


def _validate_config(contract: dict[str, Any], errors: list[str]) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        errors.append("client-config.yaml must use schema version 1")
        return
    endpoints = config.get("endpoints", {})
    if set(endpoints) != {"http", "grpc"} or any(
        not isinstance(value, str) or ":" not in value for value in endpoints.values()
    ):
        errors.append("client endpoints must define HTTP and gRPC host:port values")
    models = config.get("models", {})
    for task in ("classification", "detection"):
        name = models.get(task)
        if name not in contract.get("models", {}) or contract["models"][name]["task"] != task:
            errors.append(f"default {task} model is inconsistent with the client contract")
    if config.get("defaults", {}).get("auto_load") is not True:
        errors.append("client auto_load must default to true")
    detection = config.get("detection", {})
    for field in ("confidence_threshold", "iou_threshold"):
        value = detection.get(field)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"detection {field} must be within [0, 1]")


def _validate_samples(errors: list[str]) -> None:
    manifest = _json(SAMPLE_MANIFEST_PATH)
    entries = manifest.get("samples", [])
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        errors.append("sample manifest must use schema version 1 and contain samples")
        return
    if len(entries) < 10:
        errors.append("at least 10 sample images are required")
    filenames: set[str] = set()
    digests: set[str] = set()
    total_size = 0
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("sample manifest entries must be objects")
            continue
        filename = str(entry.get("filename", ""))
        if filename in filenames or Path(filename).name != filename:
            errors.append(f"duplicate or unsafe sample filename: {filename}")
            continue
        filenames.add(filename)
        path = SAMPLE_DIRECTORY / filename
        if not path.is_file():
            errors.append(f"sample is missing: {filename}")
            continue
        size = path.stat().st_size
        total_size += size
        if size > 1024 * 1024:
            errors.append(f"sample exceeds 1 MiB: {filename}")
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            errors.append(f"sample SHA-256 is stale: {filename}")
        if digest in digests:
            errors.append(f"sample content is duplicated: {filename}")
        digests.add(digest)
        if entry.get("license") not in {"CC0-1.0", "PDM-1.0"}:
            errors.append(f"sample license is not public-domain compatible: {filename}")
        if not str(entry.get("source_url", "")).startswith("https://"):
            errors.append(f"sample source URL must use HTTPS: {filename}")
        if not str(entry.get("attribution", "")).strip():
            errors.append(f"sample attribution is missing: {filename}")
        try:
            with Image.open(path) as image:
                image.load()
                dimensions = (image.width, image.height)
            if dimensions != (entry.get("width"), entry.get("height")):
                errors.append(f"sample dimensions are stale: {filename}")
        except OSError:
            errors.append(f"sample cannot be decoded: {filename}")
    disk_images = {
        path.name for path in SAMPLE_DIRECTORY.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if disk_images != filenames:
        errors.append("sample image directory and manifest inventory differ")
    if total_size > 8 * 1024 * 1024:
        errors.append("sample image inventory exceeds 8 MiB")


def _validate_boundaries(errors: list[str]) -> None:
    for path in (REPOSITORY_ROOT / "client").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        if "models/model-spec" in lowered or "models/model-manifest" in lowered:
            errors.append(f"client reads the model repository directly: {path.relative_to(REPOSITORY_ROOT)}")
        if "scripts.model_preparation" in lowered:
            errors.append(f"client imports model-preparation: {path.relative_to(REPOSITORY_ROOT)}")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content):
            errors.append(f"client contains a host-specific absolute path: {path.relative_to(REPOSITORY_ROOT)}")
        if b"\r\n" in path.read_bytes():
            errors.append(f"client source is not LF-only: {path.relative_to(REPOSITORY_ROOT)}")


def validate() -> list[str]:
    errors = [path + " is missing" for path in sorted(REQUIRED_FILES) if not (REPOSITORY_ROOT / path).exists()]
    if errors:
        return errors
    contract = _json(CONTRACT_PATH)
    errors.extend(_schema_errors(contract, CONTRACT_SCHEMA_PATH, "client-contract"))
    event_schema = _json(EVENT_SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(event_schema)
    except Exception as error:
        errors.append(f"inference event schema is invalid: {error}")
    manifest = _json(prepare_models.MANIFEST_PATH)
    if contract != prepare_models.render_client_contract(manifest):
        errors.append("shared/client-model-contracts.json is stale")
    _validate_requirements(errors)
    _validate_config(contract, errors)
    _validate_samples(errors)
    _validate_boundaries(errors)
    expected_header = [
        "timestamp_utc", "request_id", "model", "requested_version", "resolved_version",
        "protocol", "input_count", "input_names", "batch_size", "preprocessing_ms",
        "request_ms", "postprocessing_ms", "total_ms", "status",
        "prediction_summary_json", "error",
    ]
    if CSV_HEADER != expected_header:
        errors.append("CSV export header differs from the audit contract")
    process = subprocess.run(
        [sys.executable, "client/inference_client.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or "classify" not in process.stdout or "detect" not in process.stdout:
        errors.append("client CLI without arguments must print help and exit 0")
    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Client validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    print("[OK] Production client, logging, contract, and samples validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
