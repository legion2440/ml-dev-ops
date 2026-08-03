"""Validate the step 3 model specification and local artifact repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
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

    models = spec.get("models", {})
    if set(models) != {"resnet50", "yolo11n"}:
        errors.append("model spec must contain exactly resnet50 and yolo11n")
        return errors

    expected_serving = {
        "resnet50": {"onnx", "tensorrt"},
        "yolo11n": {"onnx"},
    }
    for model_key, serving_kinds in expected_serving.items():
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
        for tensor_kind in ("input", "output"):
            shape = model.get(tensor_kind, {}).get("shape", [])
            valid_shape = (
                isinstance(shape, list)
                and len(shape) >= 2
                and shape[0] == -1
                and all(isinstance(value, int) and value > 0 for value in shape[1:])
            )
            if not valid_shape:
                errors.append(
                    f"{model_key} {tensor_kind} must have one dynamic batch "
                    "dimension followed by positive fixed dimensions"
                )
        labels = model.get("labels", {})
        for field in ("source_sha256", "generated_sha256"):
            if not preparation.FULL_SHA256.fullmatch(str(labels.get(field, ""))):
                errors.append(f"{model_key} labels {field} must be complete")

        servings = model.get("serving", {})
        if set(servings) != serving_kinds:
            errors.append(f"{model_key} serving variants are inconsistent")
            continue
        smoke_batches = model.get("smoke_batches", [])
        for serving_kind, serving in servings.items():
            serving_name = str(serving.get("name", ""))
            config_path = str(serving.get("config_path", ""))
            expected_parent = PurePosixPath("models") / serving_name
            if config_path != (expected_parent / "config.pbtxt").as_posix():
                errors.append(f"{model_key} {serving_kind} config path is inconsistent")
            versions = serving.get("versions", {})
            version_numbers = sorted(int(version) for version in versions)
            policy = serving.get("version_policy", {}).get("specific", [])
            if policy != version_numbers:
                errors.append(f"{model_key} {serving_kind} version policy is inconsistent")
            artifact_names: set[str] = set()
            for version, details in versions.items():
                artifact_path = PurePosixPath(str(details.get("artifact_path", "")))
                if artifact_path.parent != expected_parent / version:
                    errors.append(
                        f"{model_key} {serving_kind} version {version} path is inconsistent"
                    )
                artifact_names.add(artifact_path.name)
                if serving_kind == "onnx" and artifact_path.name != "model.onnx":
                    errors.append(
                        f"{model_key} ONNX version {version} must use model.onnx"
                    )
            if len(artifact_names) != 1:
                errors.append(f"{model_key} {serving_kind} artifact filenames differ by version")
            max_batch_size = serving.get("max_batch_size")
            if (
                not isinstance(smoke_batches, list)
                or not smoke_batches
                or not isinstance(max_batch_size, int)
                or any(
                    not isinstance(batch, int) or batch < 1 or batch > max_batch_size
                    for batch in smoke_batches
                )
            ):
                errors.append(f"{model_key} smoke batches exceed serving capacity")
            dynamic = serving.get("scheduling", {}).get("dynamic_batching", {})
            preferred = dynamic.get("preferred_batch_sizes", [])
            delay = dynamic.get("max_queue_delay_microseconds")
            if (
                not isinstance(preferred, list)
                or not preferred
                or not isinstance(max_batch_size, int)
                or any(
                    not isinstance(batch, int) or batch < 1 or batch > max_batch_size
                    for batch in preferred
                )
            ):
                errors.append(f"{model_key} {serving_kind} dynamic batching is inconsistent")
            if not isinstance(delay, int) or delay < 0:
                errors.append(f"{model_key} {serving_kind} queue delay is inconsistent")

    resnet = models["resnet50"]
    resnet_input = resnet.get("input", {}).get("shape", [])
    resnet_output = resnet.get("output", {}).get("shape", [])
    resnet_preprocessing = resnet.get("preprocessing", {})
    resnet_scale = resnet_preprocessing.get("scale", [])
    if (
        not isinstance(resnet_scale, list)
        or len(resnet_scale) != 2
        or not all(isinstance(value, (int, float)) for value in resnet_scale)
        or resnet_scale[0] >= resnet_scale[1]
    ):
        errors.append("ResNet preprocessing scale must be an increasing range")
    if len(resnet_input) == 4:
        crop = resnet_preprocessing.get("center_crop")
        resize = resnet_preprocessing.get("resize")
        if (
            not isinstance(crop, int)
            or not isinstance(resize, int)
            or resnet_input[-2:] != [crop, crop]
            or resize < crop
        ):
            errors.append("ResNet preprocessing spatial dimensions are inconsistent")
        channels = resnet_input[1]
        if any(
            not isinstance(resnet_preprocessing.get(field), list)
            or len(resnet_preprocessing[field]) != channels
            for field in ("mean", "std")
        ):
            errors.append("ResNet normalization channels are inconsistent with its input")
    if (
        len(resnet_output) != 2
        or resnet_output[-1] != resnet.get("labels", {}).get("count")
    ):
        errors.append("ResNet output width is inconsistent with its label count")
    if resnet.get("source", {}).get("weights_license") is not None:
        errors.append("ResNet pretrained weights must not claim an unverified license")
    if not str(resnet.get("source", {}).get("weights_terms", "")).strip():
        errors.append("ResNet weights terms must retain an upstream review notice")

    onnx_serving = resnet.get("serving", {}).get("onnx", {})
    tensorrt_serving = resnet.get("serving", {}).get("tensorrt", {})
    if onnx_serving.get("max_batch_size") != tensorrt_serving.get("max_batch_size"):
        errors.append("ResNet serving variants must share max_batch_size")
    profile = tensorrt_serving.get("profile", {})
    profile_shapes = [profile.get(name, []) for name in ("min", "opt", "max")]
    if any(
        not isinstance(shape, list)
        or not shape
        or len(shape) != len(resnet_input)
        or shape[1:] != resnet_input[1:]
        or not isinstance(shape[0], int)
        or shape[0] < 1
        for shape in profile_shapes
    ):
        errors.append("TensorRT profile dimensions are inconsistent with the input contract")
    else:
        profile_batches = [shape[0] for shape in profile_shapes]
        if profile_batches != sorted(profile_batches):
            errors.append("TensorRT profile batches must be ordered min <= opt <= max")
        if profile_batches[-1] != tensorrt_serving.get("max_batch_size"):
            errors.append("TensorRT profile max batch is inconsistent with max_batch_size")
        if any(
            batch < profile_batches[0] or batch > profile_batches[-1]
            for batch in resnet.get("smoke_batches", [])
        ):
            errors.append("ResNet smoke batches fall outside the TensorRT profile")

    for version in tensorrt_serving.get("versions", {}).values():
        actual_plan = PurePosixPath(str(version.get("artifact_path", ""))).name
        if actual_plan != "model.plan":
            errors.append("TensorRT artifact filename must be portable model.plan")

    yolo = models["yolo11n"]
    yolo_input = yolo.get("input", {}).get("shape", [])
    yolo_preprocessing = yolo.get("preprocessing", {})
    yolo_scale = yolo_preprocessing.get("scale", [])
    if (
        not isinstance(yolo_scale, list)
        or len(yolo_scale) != 2
        or not all(isinstance(value, (int, float)) for value in yolo_scale)
        or yolo_scale[0] >= yolo_scale[1]
    ):
        errors.append("YOLO preprocessing scale must be an increasing range")
    if len(yolo_input) == 4:
        if yolo_preprocessing.get("resize") != yolo_input[-2:]:
            errors.append("YOLO preprocessing resize is inconsistent with its input")
        if len(str(yolo_preprocessing.get("channel_order", ""))) != yolo_input[1]:
            errors.append("YOLO channel order is inconsistent with its input")
    yolo_source = models["yolo11n"].get("source", {})
    repository_license = str(spec.get("repository_license", "")).removesuffix("-only")
    if yolo_source.get("code_license") != yolo_source.get("weights_license"):
        errors.append("YOLO code and weights must record the same license")
    if yolo_source.get("code_license") != repository_license:
        errors.append("YOLO license must be consistent with the repository license")
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


def validate_version_directories(models_root: Path, spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for model_name in EXPECTED_MODEL_DIRECTORIES:
        model_path = models_root / model_name
        if not model_path.is_dir():
            errors.append(f"missing model directory: models/{model_name}")
            continue
        expected_versions = {
            version
            for model in spec["models"].values()
            for serving in model["serving"].values()
            if serving["name"] == model_name
            for version in serving["versions"]
        }
        actual_versions: set[str] = set()
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
            else:
                actual_versions.add(child.name)
        unexpected_versions = actual_versions - expected_versions
        if unexpected_versions:
            errors.append(
                f"unexpected version directories for {model_name}: "
                f"{sorted(unexpected_versions)}"
            )
    return errors


def _validate_layout(spec: dict[str, Any], errors: list[str]) -> None:
    models_root = REPOSITORY_ROOT / "models"
    actual_directories = {path.name for path in models_root.iterdir() if path.is_dir()}
    unexpected = sorted(actual_directories - EXPECTED_MODEL_DIRECTORIES)
    if unexpected:
        errors.append("unexpected top-level model directories: " + ", ".join(unexpected))
    errors.extend(validate_version_directories(models_root, spec))
    for model_name in EXPECTED_MODEL_DIRECTORIES:
        config_path = models_root / model_name / "config.pbtxt"
        if not config_path.is_file():
            errors.append(f"missing Triton config: models/{model_name}/config.pbtxt")
            continue
        content = config_path.read_text(encoding="utf-8")
        if content != preparation.render_config(spec, model_name):
            errors.append(f"stale Triton config: models/{model_name}/config.pbtxt")


def _validate_git_tracking(spec: dict[str, Any], errors: list[str]) -> None:
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
    artifact_paths = {
        version["artifact_path"]
        for model in spec["models"].values()
        for serving in model["serving"].values()
        for version in serving["versions"].values()
    }
    forbidden = [
        path
        for path in tracked
        if path in artifact_paths
        or path.endswith(".pt")
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
        for version, version_entry in entry.get("versions", {}).items():
            artifact = version_entry.get("artifact", {})
            relative_path = artifact.get("path")
            if not isinstance(relative_path, str):
                errors.append(f"manifest {model_name}:{version} has no artifact path")
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
        _validate_git_tracking(spec, errors)
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
