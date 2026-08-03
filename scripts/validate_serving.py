"""Validate step 4 serving structure and optional local artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.model_preparation import prepare_models
from shared.triton_model_config import (
    render_load_config_json,
    render_pbtxt,
    validate_contract_relationships,
)

COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"
ENV_PATH = REPOSITORY_ROOT / ".env.example"
VERIFIER_PATH = REPOSITORY_ROOT / "deployment/triton/verify_serving.py"


def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def validate_structure() -> list[str]:
    errors: list[str] = []
    spec = prepare_models.load_spec()
    manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    env = _env()
    if not VERIFIER_PATH.is_file():
        errors.append("deployment/triton/verify_serving.py is missing")
    if manifest.get("schema_version") != 2 or spec.get("schema_version") != 2:
        errors.append("model spec and manifest must use schema version 2")
    for name in prepare_models.SERVING_MODELS:
        config = prepare_models.serving_model_config(spec, name)
        errors.extend(f"{name}: {error}" for error in validate_contract_relationships(config))
        config_path = REPOSITORY_ROOT / "models" / name / "config.pbtxt"
        if config_path.read_text(encoding="utf-8") != render_pbtxt(config):
            errors.append(f"models/{name}/config.pbtxt is stale")
        entry = manifest.get("models", {}).get(name, {})
        if entry.get("model_config") != config:
            errors.append(f"manifest {name} ModelConfig is stale")
        versions = sorted(int(version) for version in entry.get("versions", {}))
        try:
            wrapper = render_load_config_json(config, versions)
            rendered = json.loads(wrapper["parameters"]["config"])
            if rendered != config:
                errors.append(f"{name} runtime load wrapper is not the complete ModelConfig")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{name} runtime load wrapper failed: {error}")
    trt_config = manifest.get("models", {}).get("resnet50_tensorrt", {}).get(
        "model_config", {}
    )
    if trt_config.get("default_model_filename") != "model.plan":
        errors.append("TensorRT serving config must explicitly select model.plan")
    if "cc_model_filenames" in trt_config:
        errors.append("single-plan TensorRT serving config must not map compute capability")
    service = compose.get("services", {}).get("triton-verifier", {})
    if service.get("profiles") != ["verification"]:
        errors.append("triton-verifier must belong only to the verification profile")
    if service.get("ports"):
        errors.append("triton-verifier must not publish ports")
    if service.get("restart") != "no":
        errors.append("triton-verifier restart policy must be no")
    if service.get("deploy", {}).get("resources"):
        errors.append("triton-verifier must not reserve a GPU")
    if service.get("image") != "${TRITON_SDK_IMAGE:?TRITON_SDK_IMAGE is required}":
        errors.append("triton-verifier must consume the SDK image pin")
    command = service.get("command", [])
    expected_suffix = ["--evidence-directory", "docs/evidence/portability"]
    if not isinstance(command, list) or command[-2:] != expected_suffix:
        errors.append("triton-verifier must write only portability evidence")
    writable_mounts = [
        volume
        for volume in service.get("volumes", [])
        if isinstance(volume, dict) and volume.get("read_only") is not True
    ]
    if writable_mounts != [
        {
            "type": "bind",
            "source": "./docs/evidence/portability",
            "target": "/workspace/docs/evidence/portability",
        }
    ]:
        errors.append("triton-verifier writable mount must be limited to portability evidence")
    sdk = env.get("TRITON_SDK_IMAGE", "")
    server = env.get("TRITON_IMAGE", "")
    if sdk.replace("-sdk", "") != server:
        errors.append("Triton SDK and server images must use one release line")
    process = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), "--check"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        errors.append("local verifier check failed: " + (process.stderr.strip() or process.stdout.strip()))
    return errors


def validate_artifacts(errors: list[str]) -> None:
    manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))
    versions = manifest["models"]["resnet50_onnx"]["versions"]
    paths = [REPOSITORY_ROOT / versions[version]["artifact"]["path"] for version in ("1", "2")]
    if any(not path.is_file() for path in paths):
        errors.append("ResNet ONNX v1/v2 artifacts are required")
    elif prepare_models.sha256_file(paths[0]) == prepare_models.sha256_file(paths[1]):
        errors.append("ResNet ONNX v1/v2 artifacts must have different SHA-256 values")
    process = subprocess.run(
        [sys.executable, "scripts/validate_model_repository.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        errors.append("artifact-level model validation failed: " + (process.stderr.strip() or process.stdout.strip()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-only", action="store_true")
    arguments = parser.parse_args()
    try:
        errors = validate_structure()
        if not arguments.structure_only:
            validate_artifacts(errors)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Serving validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1
    state = "structure-only" if arguments.structure_only else "artifact-complete"
    print(f"[OK] Triton serving {state} validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
