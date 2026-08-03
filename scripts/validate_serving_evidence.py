"""Validate committed step 4 serving evidence without Docker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.triton.verify_serving import verification_semantic_contract  # noqa: E402

EVIDENCE_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/serving-runtime.json"
REPOSITORY_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/repository-versions.txt"
HISTORICAL_MANIFEST_PATH = (
    REPOSITORY_ROOT / "docs/evidence/step-4/runtime-model-manifest.json"
)
HISTORICAL_SPEC_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/runtime-model-spec.yaml"
INTEGRITY_PATH = REPOSITORY_ROOT / "docs/evidence/step-4/runtime-integrity.json"
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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serving_semantic_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project serving behavior while excluding plan and host provenance."""
    models: dict[str, Any] = {}
    for name, entry in sorted(manifest.get("models", {}).items()):
        config = {
            key: value
            for key, value in entry.get("model_config", {}).items()
            if key not in {"default_model_filename", "cc_model_filenames"}
        }
        models[name] = {
            "logical_model_id": entry.get("logical_model_id"),
            "versions": sorted(entry.get("versions", {}), key=int),
            "input": entry.get("input"),
            "output": entry.get("output"),
            "precision": entry.get("precision"),
            "compute_precision": entry.get("compute_precision"),
            "io_precision": entry.get("io_precision"),
            "max_batch_size": entry.get("max_batch_size"),
            "version_policy": entry.get("version_policy"),
            "scheduling": entry.get("scheduling"),
            "model_config": config,
        }
    return {
        "schema_version": 1,
        "models": models,
        "verification": verification_semantic_contract(),
    }


def _env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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


def _validate_listing(
    manifest: dict[str, Any], errors: list[str], path: Path = REPOSITORY_PATH
) -> None:
    with path.open(encoding="utf-8", newline="") as source:
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


def _sanitization_errors(paths: tuple[Path, ...]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if WINDOWS_PATH.search(content) or "/mnt/" in content:
            errors.append(f"{path.name} contains a host path")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{path.name} must use LF")
        if any(
            marker in content.upper() for marker in ("PASSWORD=", "TOKEN=", "SECRET=")
        ):
            errors.append(f"{path.name} contains a secret marker")
    return errors


def validate_historical(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Validate Step 4 solely against its immutable runtime snapshots."""
    step = root / "docs/evidence/step-4"
    evidence_path = step / "serving-runtime.json"
    repository_path = step / "repository-versions.txt"
    manifest_path = step / "runtime-model-manifest.json"
    spec_path = step / "runtime-model-spec.yaml"
    integrity_path = step / "runtime-integrity.json"
    errors: list[str] = []
    try:
        evidence = _load(evidence_path)
        manifest = _load(manifest_path)
        integrity = _load(integrity_path)
        artifacts = integrity.get("historical_artifacts", {})
        expected_paths = {
            "serving_runtime": evidence_path,
            "repository_versions": repository_path,
            "model_manifest": manifest_path,
            "model_spec": spec_path,
        }
        for label, path in expected_paths.items():
            relative = path.relative_to(root).as_posix()
            if artifacts.get(label) != {"path": relative, "sha256": _sha256(path)}:
                errors.append(f"historical artifact hash is stale: {label}")
        projection = integrity.get("runtime_compatibility_projection")
        if not isinstance(projection, dict) or integrity.get(
            "runtime_compatibility_projection_sha256"
        ) != _canonical_sha256(projection):
            errors.append("historical compatibility projection hash is stale")
        errors.extend(validate_evidence(evidence, manifest))
        if evidence.get("manifest_sha256") != _sha256(manifest_path):
            errors.append("historical evidence manifest SHA-256 is stale")
        if evidence.get("spec_sha256") != _sha256(spec_path):
            errors.append("historical evidence spec SHA-256 is stale")
        _validate_listing(manifest, errors, repository_path)
        errors.extend(
            _sanitization_errors(
                (
                    evidence_path,
                    repository_path,
                    manifest_path,
                    spec_path,
                    integrity_path,
                )
            )
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        errors.append(f"cannot load historical serving evidence: {error}")
    return errors


def validate_current_compatibility(
    integrity: dict[str, Any] | None = None,
    source_root: Path = REPOSITORY_ROOT,
) -> list[str]:
    """Compare current serving semantics with the historical runtime contract."""
    if integrity is None:
        integrity = _load(INTEGRITY_PATH)
    stored = integrity.get("runtime_compatibility_projection")
    current = serving_semantic_projection(
        _load(source_root / "models/model-manifest.json")
    )
    if not isinstance(stored, dict) or current != stored:
        return ["current serving contract is incompatible with historical Step 4"]
    return []


def validate_portability(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Validate the selected-GPU build record and new serving runtime proof."""
    errors: list[str] = []
    try:
        directory = root / "docs/evidence/portability"
        evidence_path = directory / "serving-runtime.json"
        repository_path = directory / "repository-versions.txt"
        build_record_path = directory / "build-record.json"
        manifest_path = root / "models/model-manifest.json"
        spec_path = root / "models/model-spec.yaml"
        evidence = _load(evidence_path)
        manifest = _load(manifest_path)
        build_record = _load(build_record_path)
        build_schema = _load(root / "schemas/portability-build-record.schema.json")
        Draft202012Validator.check_schema(build_schema)
        errors.extend(
            f"build-record.{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
            for error in Draft202012Validator(build_schema).iter_errors(build_record)
        )
        env = _env(root / ".env.example")
        errors.extend(validate_evidence(evidence, manifest))
        if evidence.get("manifest_sha256") != _sha256(manifest_path):
            errors.append("portability evidence manifest SHA-256 is stale")
        if evidence.get("spec_sha256") != _sha256(spec_path):
            errors.append("portability evidence spec SHA-256 is stale")
        if evidence.get("images") != {
            "server": env.get("TRITON_IMAGE"), "sdk": env.get("TRITON_SDK_IMAGE")
        }:
            errors.append("portability evidence image pins are stale")
        _validate_listing(manifest, errors, repository_path)
        engine = manifest["models"]["resnet50_tensorrt"]["versions"]["1"][
            "artifact"
        ]
        if build_record.get("gpu") != manifest.get("build", {}).get("gpu"):
            errors.append("portability build GPU differs from manifest provenance")
        if build_record.get("toolchain") != manifest.get("build", {}).get(
            "tensorrt_environment"
        ):
            errors.append("portability toolchain differs from manifest provenance")
        if build_record.get("engine") != engine:
            errors.append("portability engine differs from manifest artifact")
        if manifest.get("build", {}).get(
            "tensorrt_build_record_sha256"
        ) != _sha256(build_record_path):
            errors.append("manifest TensorRT build-record SHA-256 is stale")
        errors.extend(
            _sanitization_errors((evidence_path, repository_path, build_record_path))
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        errors.append(f"cannot load portability evidence: {error}")
    return errors


def validate(*, historical_only: bool = False) -> list[str]:
    errors = validate_historical()
    if not historical_only:
        errors.extend(validate_current_compatibility())
        errors.extend(validate_portability())
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-only",
        action="store_true",
        help="validate immutable Step 4 evidence without current compatibility",
    )
    arguments = parser.parse_args()
    errors = validate(historical_only=arguments.historical_only)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Serving evidence validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    if arguments.historical_only:
        print("[OK] Step 4 historical serving evidence integrity passes.")
    else:
        print(
            "[OK] Step 4 historical integrity, current compatibility, and "
            "GPU-portability runtime evidence pass."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
