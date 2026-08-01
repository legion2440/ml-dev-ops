"""Validate committed step 4 serving evidence without Docker."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/serving-runtime.json"
REPOSITORY_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/repository-versions.txt"
MANIFEST_PATH = REPOSITORY_ROOT / "models/model-manifest.json"
SPEC_PATH = REPOSITORY_ROOT / "models/model-spec.yaml"
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/serving-evidence.schema.json"
ENV_PATH = REPOSITORY_ROOT / ".env.example"
REQUIRED_EXTENSIONS = {"model_repository", "statistics", "model_configuration"}
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+[\\/]")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def validate_evidence(evidence: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    schema = _load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = [
        f"evidence.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in Draft202012Validator(schema).iter_errors(evidence)
    ]
    expected_matrix = {
        f"{name}:{version}"
        for name, entry in manifest.get("models", {}).items()
        for version in entry.get("versions", {})
    }
    if set(evidence.get("models", {})) != expected_matrix:
        errors.append("protocol matrix does not match manifest model versions")
    if set(evidence.get("dynamic_batching", {})) != set(manifest.get("models", {})):
        errors.append("dynamic batching model set does not match manifest")
    if not REQUIRED_EXTENSIONS.issubset(set(evidence.get("extensions", []))):
        errors.append("required Triton extensions are missing")
    for name, result in evidence.get("dynamic_batching", {}).items():
        expected_version = str(max(manifest["models"][name]["version_policy"]["specific"]))
        if result.get("model_version") != expected_version:
            errors.append(f"{name} batching version is stale")
        requests = result.get("requests")
        if result.get("success_count_delta") != requests or result.get("inference_count_delta") != requests:
            errors.append(f"{name} batching counts do not equal requests")
        execution = result.get("execution_count_delta")
        if not isinstance(execution, int) or not isinstance(requests, int) or not 0 < execution < requests:
            errors.append(f"{name} execution count does not prove batching")
        if not any(batch > 1 for batch in result.get("observed_batch_sizes", [])):
            errors.append(f"{name} has no observed batch larger than one")
        if result.get("passed") is not True or result.get("finite_outputs") is not True:
            errors.append(f"{name} batching result is not passed")
        attempts = result.get("attempts", [])
        attempts_used = result.get("attempts_used")
        if (
            not isinstance(attempts, list)
            or not isinstance(attempts_used, int)
            or attempts_used != len(attempts)
            or not 1 <= attempts_used <= 3
        ):
            errors.append(f"{name} attempts_used is inconsistent")
        elif (
            attempts[-1].get("attempt") != attempts_used
            or attempts[-1].get("passed") is not True
        ):
            errors.append(f"{name} final batching attempt is inconsistent")
    switching = evidence.get("version_switching", {})
    if switching.get("sequence") != ["1", "2", "1+2"]:
        errors.append("version switching sequence is stale")
    if switching.get("default_with_tracked_policy") != "2":
        errors.append("tracked policy did not select ResNet version 2")
    if evidence.get("final_repository_ready_models") != []:
        errors.append("cleanup ready set must be empty")
    readiness = evidence.get("model_readiness_after_cleanup", {})
    if set(readiness) != set(manifest.get("models", {})):
        errors.append("cleanup readiness model set does not match manifest")
    else:
        for name, entry in manifest["models"].items():
            recorded = readiness.get(name, {})
            versions = recorded.get("versions", {})
            if set(versions) != set(entry["versions"]):
                errors.append(f"{name} cleanup readiness version set is stale")
            if recorded.get("model_ready") is not False or any(
                ready is not False for ready in versions.values()
            ):
                errors.append(f"{name} remained ready after cleanup")
    return errors


def _validate_listing(manifest: dict[str, Any], errors: list[str]) -> None:
    with REPOSITORY_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        rows = list(reader)
    if reader.fieldnames != ["model", "version", "state", "reason"]:
        errors.append("repository-versions.txt header is invalid")
        return
    resnet_versions = {
        row["version"] for row in rows if row["model"] == "resnet50_onnx" and row["state"] == "READY"
    }
    expected = set(manifest["models"]["resnet50_onnx"]["versions"])
    if resnet_versions != expected:
        errors.append("repository snapshot does not show both ResNet versions READY")


def main() -> int:
    errors: list[str] = []
    try:
        evidence = _load(EVIDENCE_PATH)
        manifest = _load(MANIFEST_PATH)
        env = _env()
        errors.extend(validate_evidence(evidence, manifest))
        if evidence.get("manifest_sha256") != _sha256(MANIFEST_PATH):
            errors.append("evidence manifest SHA-256 is stale")
        if evidence.get("spec_sha256") != _sha256(SPEC_PATH):
            errors.append("evidence spec SHA-256 is stale")
        if evidence.get("images") != {
            "server": env.get("TRITON_IMAGE"), "sdk": env.get("TRITON_SDK_IMAGE")
        }:
            errors.append("evidence image pins are stale")
        _validate_listing(manifest, errors)
        for path in (EVIDENCE_PATH, REPOSITORY_PATH):
            content = path.read_text(encoding="utf-8")
            if WINDOWS_PATH.search(content) or "/mnt/" in content:
                errors.append(f"{path.name} contains a host path")
            if b"\r\n" in path.read_bytes():
                errors.append(f"{path.name} must use LF")
            if any(marker in content.upper() for marker in ("PASSWORD=", "TOKEN=", "SECRET=")):
                errors.append(f"{path.name} contains a secret marker")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"[FAIL] Cannot load serving evidence: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Serving evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    print("[OK] Step 4 serving evidence is current and machine-verifiable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
