"""Validate committed step 3 preparation and Triton evidence without Docker."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-3"
PREPARATION_PATH = EVIDENCE_DIRECTORY / "preparation.json"
SMOKE_PATH = EVIDENCE_DIRECTORY / "triton-model-smoke.json"
REPOSITORY_PATH = EVIDENCE_DIRECTORY / "model-repository.txt"
MANIFEST_PATH = EVIDENCE_DIRECTORY / "model-manifest-v1.json"
ENV_EXAMPLE_PATH = REPOSITORY_ROOT / ".env.example"
MODEL_NAMES = {"resnet50_onnx", "resnet50_tensorrt", "yolo11n_onnx"}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _validate_preparation(preparation: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "manifest_path": "docs/evidence/step-3/model-manifest-v1.json",
        "manifest_sha256": _sha256(MANIFEST_PATH),
    }
    if preparation != expected:
        errors.append("preparation.json must contain only the current manifest path and SHA-256")


def _validate_smoke(
    smoke: dict[str, Any],
    manifest: dict[str, Any],
    env: dict[str, str],
    errors: list[str],
) -> None:
    if smoke.get("schema_version") != 1 or smoke.get("ok") is not True:
        errors.append("triton-model-smoke.json must record schema 1 and ok=true")
    try:
        timestamp_value = str(smoke.get("captured_at_utc", "")).replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(timestamp_value)
        if timestamp.tzinfo is None:
            errors.append("model smoke timestamp must include a timezone")
    except ValueError:
        errors.append("model smoke timestamp is invalid")
    if smoke.get("triton_url") != f"http://127.0.0.1:{env.get('TRITON_HTTP_PORT')}":
        errors.append("model smoke URL is stale relative to .env.example")
    if smoke.get("triton_image") != env.get("TRITON_IMAGE"):
        errors.append("model smoke Triton image is stale relative to .env.example")
    if smoke.get("spec_sha256") != manifest.get("spec_sha256"):
        errors.append("model smoke spec SHA-256 is stale relative to the manifest")
    if smoke.get("manifest_sha256") != _sha256(MANIFEST_PATH):
        errors.append("model smoke manifest SHA-256 is stale")
    if smoke.get("repository_index") != "passed" or smoke.get("unload") != "passed":
        errors.append("model smoke must record repository index and unload success")

    models = smoke.get("models", {})
    if not isinstance(models, dict) or set(models) != MODEL_NAMES:
        errors.append("model smoke must contain exactly the three serving models")
        return
    expected_batches = {
        name: manifest["models"][name]["smoke_batches"] for name in MODEL_NAMES
    }
    for name, expected in expected_batches.items():
        entry = models[name]
        for field in ("explicit_load", "readiness", "metadata", "config"):
            if entry.get(field) != "passed":
                errors.append(f"model smoke {name}.{field} is not passed")
        batches = entry.get("batches", [])
        if [item.get("batch") for item in batches] != expected:
            errors.append(f"model smoke batches are stale for {name}")
        expected_output = manifest["models"][name]["output"]["shape"][1:]
        for item in batches:
            if item.get("output_shape") != [item.get("batch"), *expected_output]:
                errors.append(f"model smoke output shape is stale for {name}")
            if item.get("finite") is not True:
                errors.append(f"model smoke output is not finite for {name}")

    parity = smoke.get("resnet_parity", [])
    expected_resnet_batches = manifest["models"]["resnet50_onnx"]["smoke_batches"]
    if [item.get("batch") for item in parity] != expected_resnet_batches:
        errors.append("ResNet runtime parity batches are stale")
    tolerances = manifest["models"]["resnet50_onnx"]["parity_tolerances"]
    for item in parity:
        values = (
            item.get("max_abs_error"),
            item.get("mean_abs_error"),
            item.get("cosine_similarity"),
            item.get("top1_agreement"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            errors.append("ResNet runtime parity contains invalid metrics")
            continue
        if item.get("status") != "passed":
            errors.append("ResNet runtime parity is not passed")
        if item["max_abs_error"] > tolerances["max_abs_error"]:
            errors.append("ResNet runtime max error exceeds the predefined tolerance")
        if item["mean_abs_error"] > tolerances["mean_abs_error"]:
            errors.append("ResNet runtime mean error exceeds the predefined tolerance")
        if item["cosine_similarity"] < tolerances["minimum_cosine_similarity"]:
            errors.append("ResNet runtime cosine similarity is below tolerance")
        if item["top1_agreement"] < tolerances["minimum_top1_agreement"]:
            errors.append("ResNet runtime top-1 agreement is below tolerance")


def _validate_repository_listing(errors: list[str]) -> None:
    with REPOSITORY_PATH.open(encoding="utf-8", newline="") as evidence_file:
        reader = csv.DictReader(evidence_file, delimiter="\t")
        rows = list(reader)
    if reader.fieldnames != ["model", "version", "state", "reason"]:
        errors.append("model-repository.txt has an invalid header")
        return
    if {row["model"] for row in rows} != MODEL_NAMES or len(rows) != 3:
        errors.append("model-repository.txt must contain exactly three models")
    for row in rows:
        if row["version"] != "1" or row["state"] != "READY" or row["reason"] != "-":
            errors.append(f"model-repository.txt does not record {row['model']} version 1 READY")


def _validate_sanitization(errors: list[str]) -> None:
    for path in (PREPARATION_PATH, SMOKE_PATH, REPOSITORY_PATH, MANIFEST_PATH):
        content = path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content) or "/mnt/" in content:
            errors.append(f"{path.name} contains a host-specific path")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{path.name} must use LF")
        if any(marker in content.upper() for marker in ("PASSWORD=", "TOKEN=", "SECRET=")):
            errors.append(f"{path.name} contains a sensitive field")


def main() -> int:
    errors: list[str] = []
    try:
        preparation = _load_json(PREPARATION_PATH)
        smoke = _load_json(SMOKE_PATH)
        manifest = _load_json(MANIFEST_PATH)
        env = _load_env(ENV_EXAMPLE_PATH)
        _validate_preparation(preparation, errors)
        _validate_smoke(smoke, manifest, env, errors)
        _validate_repository_listing(errors)
        _validate_sanitization(errors)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"[FAIL] Cannot load model evidence: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Model evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    print("[OK] Step 3 evidence records artifact-complete and runtime-verified states.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
