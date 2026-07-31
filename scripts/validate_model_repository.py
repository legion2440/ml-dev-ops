"""Validate the step 3 model specification and local artifact repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.model_preparation import prepare_models as preparation  # noqa: E402

SPEC_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/model-spec.schema.json"
MANIFEST_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/model-manifest.schema.json"
EXPECTED_MODEL_DIRECTORIES = {
    "resnet50_onnx",
    "resnet50_tensorrt",
    "yolo11n_onnx",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]")
LOCK_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+) \\$")
LOCK_HASH = re.compile(r"^    --hash=sha256:[0-9a-f]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _schema_errors(
    instance: dict[str, Any], schema_path: Path, label: str
) -> list[str]:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "$"
        errors.append(f"{label}.{location}: {error.message}")
    return errors


def validate_spec_semantics(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    build = spec.get("build", {})
    for key in ("exporter_image", "tensorrt_builder_image"):
        reference = str(build.get(key, ""))
        match = preparation.IMAGE_REFERENCE.fullmatch(reference)
        if not match:
            errors.append(f"{key} must use a complete tag@sha256 reference")
        elif match.group("tag").lower() == "latest":
            errors.append(f"{key} must not use latest")
    if build.get("simplify") is not False:
        errors.append("ONNX simplify must remain false in step 3")
    if build.get("target", {}).get("compute_capability") != "8.9":
        errors.append("TensorRT target compute capability must be 8.9")

    models = spec.get("models", {})
    if set(models) != {"resnet50", "yolo11n"}:
        errors.append("model spec must contain exactly resnet50 and yolo11n")
        return errors
    expected_contracts = {
        "resnet50": (["batch", 3, 224, 224], ["batch", 1000]),
        "yolo11n": (["batch", 3, 640, 640], ["batch", 84, 8400]),
    }
    for model_key, (expected_input, expected_output) in expected_contracts.items():
        model = models[model_key]
        source = model.get("source", {})
        if source.get("hash_status") != "resolved":
            errors.append(f"{model_key} source hash must be resolved")
        if not preparation.FULL_SHA256.fullmatch(str(source.get("sha256", ""))):
            errors.append(f"{model_key} source SHA-256 must be complete")
        if model.get("export", {}).get("nms") is not False:
            errors.append(f"{model_key} export must keep NMS outside the model")
        dynamic_axes = model.get("export", {}).get("dynamic_axes", {})
        tensor_names = {
            model.get("input", {}).get("name"),
            model.get("output", {}).get("name"),
        }
        invalid_axes = any(value != ["batch"] for value in dynamic_axes.values())
        if set(dynamic_axes) != tensor_names or invalid_axes:
            errors.append(f"{model_key} must expose only a dynamic batch axis")
        input_shape = model.get("input", {}).get("shape", [])
        output_shape = model.get("output", {}).get("shape", [])
        rendered_input = ["batch" if value == -1 else value for value in input_shape]
        rendered_output = ["batch" if value == -1 else value for value in output_shape]
        if rendered_input != expected_input or rendered_output != expected_output:
            errors.append(f"{model_key} tensor contract is not the accepted step 3 contract")
        labels = model.get("labels", {})
        for field in ("source_sha256", "generated_sha256"):
            if not preparation.FULL_SHA256.fullmatch(str(labels.get(field, ""))):
                errors.append(f"{model_key} labels {field} must be complete")

    resnet = models["resnet50"]
    expected_preprocessing = {
        "resize": 232,
        "center_crop": 224,
        "scale": [0.0, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    if resnet.get("preprocessing") != expected_preprocessing:
        errors.append("ResNet preprocessing does not match IMAGENET1K_V2")
    if resnet.get("source", {}).get("weights_license") is not None:
        errors.append("ResNet pretrained weights must not be labeled BSD-3-Clause")
    if "ImageNet-derived" not in str(resnet.get("source", {}).get("weights_terms", "")):
        errors.append("ResNet weights terms must retain the ImageNet review notice")
    profile = resnet.get("serving", {}).get("tensorrt", {}).get("profile", {})
    if profile != {
        "min": [1, 3, 224, 224],
        "opt": [4, 3, 224, 224],
        "max": [8, 3, 224, 224],
    }:
        errors.append("TensorRT profile must remain min=1, opt=4, max=8")
    if not str(resnet.get("serving", {}).get("tensorrt", {}).get("artifact_path", "")).endswith(
        "/model_cc89.plan"
    ):
        errors.append("TensorRT artifact must use the strict model_cc89.plan filename")
    yolo_source = models["yolo11n"].get("source", {})
    if (
        yolo_source.get("code_license") != "AGPL-3.0"
        or yolo_source.get("weights_license") != "AGPL-3.0"
    ):
        errors.append("YOLO11 code and weights must record AGPL-3.0")
    return errors


def validate_lock_file(path: Path = preparation.LOCK_PATH) -> list[str]:
    errors: list[str] = []
    requirements: list[str] = []
    expecting_hash = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if (
            not line
            or line.startswith("#")
            or line.startswith("--index")
            or line.startswith("--extra-index")
        ):
            continue
        match = LOCK_REQUIREMENT.fullmatch(line)
        if match:
            if expecting_hash:
                errors.append(f"requirements.lock:{line_number} previous package has no hash")
            requirements.append(match.group(1).lower())
            expecting_hash = True
            continue
        if LOCK_HASH.fullmatch(line) and expecting_hash:
            expecting_hash = False
            continue
        errors.append(f"requirements.lock:{line_number} is not an exact hashed requirement")
    if expecting_hash:
        errors.append("requirements.lock final package has no hash")
    if len(requirements) != len(set(requirements)):
        errors.append("requirements.lock contains duplicate packages")
    expected = {
        "torch",
        "torchvision",
        "ultralytics",
        "onnx",
        "onnxconverter-common",
        "onnxruntime",
        "numpy",
        "pyyaml",
    }
    if set(requirements) != expected:
        errors.append("requirements.lock does not contain the exact exporter package set")
    return errors


def validate_version_directories(models_root: Path) -> list[str]:
    errors: list[str] = []
    for model_name in EXPECTED_MODEL_DIRECTORIES:
        model_path = models_root / model_name
        if not model_path.is_dir():
            errors.append(f"missing model directory: models/{model_name}")
            continue
        for child in model_path.iterdir():
            if not child.is_dir():
                continue
            invalid_name = (
                not child.name.isdigit()
                or str(int(child.name)) != child.name
                or int(child.name) < 1
            )
            if invalid_name:
                errors.append(f"invalid version directory: models/{model_name}/{child.name}")
            elif child.name != "1":
                errors.append(
                    f"step 3 permits only model version 1: "
                    f"models/{model_name}/{child.name}"
                )
    return errors


def _validate_layout(spec: dict[str, Any], errors: list[str]) -> None:
    models_root = REPOSITORY_ROOT / "models"
    actual_directories = {path.name for path in models_root.iterdir() if path.is_dir()}
    unexpected = sorted(actual_directories - EXPECTED_MODEL_DIRECTORIES)
    if unexpected:
        errors.append("unexpected top-level model directories: " + ", ".join(unexpected))
    errors.extend(validate_version_directories(models_root))
    for model_name in EXPECTED_MODEL_DIRECTORIES:
        config_path = models_root / model_name / "config.pbtxt"
        if not config_path.is_file():
            errors.append(f"missing Triton config: models/{model_name}/config.pbtxt")
            continue
        content = config_path.read_text(encoding="utf-8")
        if content != preparation.render_config(spec, model_name):
            errors.append(f"stale Triton config: models/{model_name}/config.pbtxt")
        for forbidden in ("dynamic_batching", "version_policy", "instance_group"):
            if forbidden in content:
                errors.append(f"models/{model_name}/config.pbtxt contains out-of-scope {forbidden}")
    tensorrt_directory = models_root / "resnet50_tensorrt" / "1"
    if (tensorrt_directory / "model.plan").exists():
        errors.append("TensorRT fallback model.plan must not exist")


def _validate_git_tracking(errors: list[str]) -> None:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        errors.append("cannot inspect Git tracking state")
        return
    tracked = process.stdout.decode("utf-8").split("\0")
    forbidden = [
        path
        for path in tracked
        if path.endswith(("model.onnx", "model_cc89.plan", ".pt"))
        or path.startswith(".cache/model-preparation/")
    ]
    if forbidden:
        errors.append("model binaries or cache are tracked by Git: " + ", ".join(forbidden))


def _validate_metadata_sanitization(paths: Iterable[Path], errors: list[str]) -> None:
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(content) or UNC_PATH.search(content) or "/mnt/" in content:
            errors.append(f"{path.relative_to(REPOSITORY_ROOT).as_posix()} contains a host path")
        if b"\r\n" in path.read_bytes():
            errors.append(f"{path.relative_to(REPOSITORY_ROOT).as_posix()} must use LF")


def _validate_manifest(spec: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    if not preparation.MANIFEST_PATH.is_file():
        errors.append("missing generated models/model-manifest.json")
        return None
    try:
        manifest = _load_json(preparation.MANIFEST_PATH)
        errors.extend(_schema_errors(manifest, MANIFEST_SCHEMA_PATH, "model-manifest.json"))
        errors.extend(preparation.manifest_staleness(spec, manifest))
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"cannot load model manifest: {error}")
        return None


def validate_artifact_inventory(
    manifest: dict[str, Any], errors: list[str], root: Path = REPOSITORY_ROOT
) -> None:
    for model_name, entry in manifest.get("models", {}).items():
        artifact = entry.get("artifact", {})
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str):
            errors.append(f"manifest {model_name} has no artifact path")
            continue
        path = root / relative_path
        if not path.is_file():
            errors.append(f"artifact-complete validation missing {relative_path}")
            continue
        if preparation.sha256_file(path) != artifact.get("sha256"):
            errors.append(f"artifact SHA-256 is stale for {relative_path}")
        if path.stat().st_size != artifact.get("size_bytes"):
            errors.append(f"artifact size is stale for {relative_path}")


def validate_repository(structure_only: bool) -> list[str]:
    errors: list[str] = []
    try:
        spec = preparation.load_spec()
        errors.extend(_schema_errors(spec, SPEC_SCHEMA_PATH, "model-spec.yaml"))
        errors.extend(validate_spec_semantics(spec))
        errors.extend(validate_lock_file())
        _validate_layout(spec, errors)
        errors.extend(preparation.check_generated(spec))
        _validate_git_tracking(errors)
        manifest = _validate_manifest(spec, errors)
        metadata_paths = [
            preparation.SPEC_PATH,
            preparation.LOCK_PATH,
            preparation.MANIFEST_PATH,
            preparation.PREPARATION_EVIDENCE_PATH,
            *(REPOSITORY_ROOT.glob("models/*/config.pbtxt")),
            *(REPOSITORY_ROOT.glob("models/*.txt")),
        ]
        _validate_metadata_sanitization(metadata_paths, errors)
        if not structure_only and manifest is not None:
            validate_artifact_inventory(manifest, errors)
            if not errors:
                preparation.validate_artifacts(spec)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        yaml.YAMLError,
        preparation.PreparationError,
    ) as error:
        errors.append(str(error))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="validate tracked repository state without requiring local model binaries",
    )
    arguments = parser.parse_args()
    errors = validate_repository(arguments.structure_only)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        state = "structure-only" if arguments.structure_only else "artifact-complete"
        print(
            f"[FAIL] Model repository {state} validation found {len(errors)} issue(s).",
            file=sys.stderr,
        )
        return 1
    state = "structure-only" if arguments.structure_only else "artifact-complete"
    print(f"[OK] Model repository {state} validation passed for three serving models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
