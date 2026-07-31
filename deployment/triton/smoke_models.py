"""Load, inspect, infer, compare, and unload the three step 3 Triton models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "models/model-manifest.json"
ENV_EXAMPLE_PATH = REPOSITORY_ROOT / ".env.example"
EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-3"
SMOKE_EVIDENCE_PATH = EVIDENCE_DIRECTORY / "triton-model-smoke.json"
REPOSITORY_EVIDENCE_PATH = EVIDENCE_DIRECTORY / "model-repository.txt"
MODEL_NAMES = ("resnet50_onnx", "resnet50_tensorrt", "yolo11n_onnx")


class SmokeError(RuntimeError):
    """A Triton runtime contract or inference check failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    expected_status: int = 200,
    timeout: float = 180.0,
) -> Any:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            content = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SmokeError(f"{method} {path} returned {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SmokeError(f"Cannot reach Triton at {base_url}: {error.reason}") from error
    if status != expected_status:
        raise SmokeError(f"{method} {path} returned {status}, expected {expected_status}")
    return json.loads(content) if content else None


def _status(base_url: str, path: str) -> int:
    request = urllib.request.Request(base_url + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _serving_contracts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = manifest.get("models", {})
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise SmokeError("Model manifest does not contain the exact runtime model set")
    return models


def _assert_metadata(name: str, metadata: dict[str, Any], model: dict[str, Any]) -> None:
    if metadata.get("name") != name:
        raise SmokeError(f"{name} metadata has an unexpected model name")
    inputs = metadata.get("inputs", [])
    outputs = metadata.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise SmokeError(f"{name} metadata must expose one input and one output")
    expected_input = model["input"]
    expected_output = model["output"]
    if (
        inputs[0].get("name") != expected_input["name"]
        or inputs[0].get("datatype") != expected_input["dtype"]
    ):
        raise SmokeError(f"{name} input metadata is stale")
    if (
        outputs[0].get("name") != expected_output["name"]
        or outputs[0].get("datatype") != expected_output["dtype"]
    ):
        raise SmokeError(f"{name} output metadata is stale")
    if inputs[0].get("shape") != expected_input["shape"]:
        raise SmokeError(f"{name} input shape metadata is stale")
    if outputs[0].get("shape") != expected_output["shape"]:
        raise SmokeError(f"{name} output shape metadata is stale")


def _assert_config(name: str, config: dict[str, Any], model: dict[str, Any]) -> None:
    if config.get("name") != name:
        raise SmokeError(f"{name} runtime config has an unexpected model name")
    if int(config.get("max_batch_size", -1)) != model["max_batch_size"]:
        raise SmokeError(f"{name} runtime max_batch_size is stale")
    # Triton materializes its default latest-version policy in the runtime
    # response even when no version_policy is present in the tracked config.
    if "dynamic_batching" in config:
        raise SmokeError(f"{name} runtime config enables out-of-scope dynamic batching")
    inputs = config.get("input", [])
    outputs = config.get("output", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise SmokeError(f"{name} runtime config must expose one input and one output")
    for actual, expected, label in (
        (inputs[0], model["input"], "input"),
        (outputs[0], model["output"], "output"),
    ):
        dimensions = [int(value) for value in actual.get("dims", [])]
        if actual.get("name") != expected["name"]:
            raise SmokeError(f"{name} runtime {label} name is stale")
        if actual.get("data_type") != f"TYPE_{expected['dtype']}":
            raise SmokeError(f"{name} runtime {label} dtype is stale")
        if dimensions != expected["shape"][1:]:
            raise SmokeError(f"{name} runtime {label} dimensions are stale")


def _synthetic_input(model: dict[str, Any], batch: int) -> list[float]:
    item_size = math.prod(int(value) for value in model["input"]["shape"][1:])
    # Shared zero values keep JSON generation compact while remaining deterministic.
    return [0.0] * (batch * item_size)


def _infer(base_url: str, name: str, model: dict[str, Any], batch: int) -> list[float]:
    input_tensor = model["input"]
    output_tensor = model["output"]
    payload = {
        "inputs": [
            {
                "name": input_tensor["name"],
                "shape": [batch, *input_tensor["shape"][1:]],
                "datatype": input_tensor["dtype"],
                "data": _synthetic_input(model, batch),
            }
        ],
        "outputs": [{"name": output_tensor["name"]}],
    }
    response = _request(
        base_url,
        "POST",
        f"/v2/models/{urllib.parse.quote(name)}/infer",
        payload,
    )
    outputs = response.get("outputs", [])
    if len(outputs) != 1:
        raise SmokeError(f"{name} batch {batch} returned an unexpected output set")
    output = outputs[0]
    expected_shape = [batch, *output_tensor["shape"][1:]]
    if output.get("name") != output_tensor["name"] or output.get("shape") != expected_shape:
        raise SmokeError(f"{name} batch {batch} returned shape {output.get('shape')}")
    data = output.get("data")
    if not isinstance(data, list) or len(data) != math.prod(expected_shape):
        raise SmokeError(f"{name} batch {batch} returned incomplete output data")
    values = [float(value) for value in data]
    if not all(math.isfinite(value) for value in values):
        raise SmokeError(f"{name} batch {batch} returned non-finite values")
    return values


def _parity(
    reference: list[float],
    candidate: list[float],
    batch: int,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise SmokeError("ResNet runtime parity output sizes differ")
    differences = [abs(left - right) for left, right in zip(reference, candidate, strict=True)]
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    dot = sum(left * right for left, right in zip(reference, candidate, strict=True))
    norm_left = math.sqrt(sum(value * value for value in reference))
    norm_right = math.sqrt(sum(value * value for value in candidate))
    cosine = dot / (norm_left * norm_right)
    width = 1000
    matches = 0
    for index in range(batch):
        start = index * width
        end = start + width
        left_top = max(range(start, end), key=reference.__getitem__) - start
        right_top = max(range(start, end), key=candidate.__getitem__) - start
        matches += left_top == right_top
    top1 = matches / batch
    checks = {
        "max_abs_error": maximum <= tolerances["max_abs_error"],
        "mean_abs_error": mean <= tolerances["mean_abs_error"],
        "cosine_similarity": cosine >= tolerances["minimum_cosine_similarity"],
        "top1_agreement": top1 >= tolerances["minimum_top1_agreement"],
    }
    if not all(checks.values()):
        raise SmokeError(f"ResNet runtime parity failed for batch {batch}: {checks}")
    return {
        "batch": batch,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "cosine_similarity": cosine,
        "top1_agreement": top1,
        "status": "passed",
    }


def _repository_evidence(index: list[dict[str, Any]]) -> str:
    rows = ["model\tversion\tstate\treason"]
    selected = [row for row in index if row.get("name") in MODEL_NAMES]
    for row in sorted(selected, key=lambda item: str(item.get("name"))):
        reason = str(row.get("reason", "")).replace("\t", " ").replace("\n", " ")
        reason = reason or "-"
        rows.append(
            f"{row.get('name', '')}\t{row.get('version', '')}\t{row.get('state', '')}\t{reason}"
        )
    return "\n".join(rows) + "\n"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def run_smoke(base_url: str, env_file: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    env = _load_env(env_file)
    contracts = _serving_contracts(manifest)
    loaded: list[str] = []
    try:
        initial_index = _request(base_url, "POST", "/v2/repository/index", {})
        visible = {row.get("name") for row in initial_index}
        missing = sorted(set(MODEL_NAMES) - visible)
        if missing:
            raise SmokeError("Triton repository index is missing: " + ", ".join(missing))

        runtime_models: dict[str, Any] = {}
        for name in MODEL_NAMES:
            encoded = urllib.parse.quote(name)
            _request(base_url, "POST", f"/v2/repository/models/{encoded}/load", {})
            loaded.append(name)
            if _status(base_url, f"/v2/models/{encoded}/ready") != 200:
                raise SmokeError(f"{name} did not become ready after explicit load")
            metadata = _request(base_url, "GET", f"/v2/models/{encoded}")
            config = _request(base_url, "GET", f"/v2/models/{encoded}/config")
            _assert_metadata(name, metadata, contracts[name])
            _assert_config(name, config, contracts[name])
            runtime_models[name] = {
                "explicit_load": "passed",
                "readiness": "passed",
                "metadata": "passed",
                "config": "passed",
                "batches": [],
            }

        ready_index = _request(base_url, "POST", "/v2/repository/index", {"ready": True})
        ready_names = {row.get("name") for row in ready_index if row.get("state") == "READY"}
        if ready_names != set(MODEL_NAMES):
            raise SmokeError(f"Ready repository set is {sorted(ready_names)}")

        resnet_outputs: dict[str, dict[int, list[float]]] = {
            "resnet50_onnx": {},
            "resnet50_tensorrt": {},
        }
        for name in ("resnet50_onnx", "resnet50_tensorrt"):
            model = contracts[name]
            for batch in model["smoke_batches"]:
                values = _infer(base_url, name, model, int(batch))
                resnet_outputs[name][int(batch)] = values
                runtime_models[name]["batches"].append(
                    {"batch": int(batch), "output_shape": [int(batch), 1000], "finite": True}
                )

        yolo = contracts["yolo11n_onnx"]
        for batch in yolo["smoke_batches"]:
            _infer(base_url, "yolo11n_onnx", yolo, int(batch))
            runtime_models["yolo11n_onnx"]["batches"].append(
                {"batch": int(batch), "output_shape": [int(batch), 84, 8400], "finite": True}
            )

        parity = [
            _parity(
                resnet_outputs["resnet50_onnx"][int(batch)],
                resnet_outputs["resnet50_tensorrt"][int(batch)],
                int(batch),
                contracts["resnet50_onnx"]["parity_tolerances"],
            )
            for batch in contracts["resnet50_onnx"]["smoke_batches"]
        ]

        repository_text = _repository_evidence(ready_index)
        smoke = {
            "schema_version": 1,
            "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "ok": True,
            "triton_url": base_url,
            "triton_image": env["TRITON_IMAGE"],
            "spec_sha256": manifest["spec_sha256"],
            "manifest_sha256": _sha256(MANIFEST_PATH),
            "repository_index": "passed",
            "models": runtime_models,
            "resnet_parity": parity,
            "unload": "passed",
        }
    finally:
        unload_errors: list[str] = []
        for name in reversed(loaded):
            encoded = urllib.parse.quote(name)
            try:
                _request(base_url, "POST", f"/v2/repository/models/{encoded}/unload", {})
                if _status(base_url, f"/v2/models/{encoded}/ready") == 200:
                    unload_errors.append(f"{name} remained ready")
            except SmokeError as error:
                unload_errors.append(str(error))
        if unload_errors and sys.exc_info()[0] is None:
            raise SmokeError("Explicit unload failed: " + "; ".join(unload_errors))

    _write_atomic(SMOKE_EVIDENCE_PATH, json.dumps(smoke, indent=2, sort_keys=True) + "\n")
    _write_atomic(REPOSITORY_EVIDENCE_PATH, repository_text)
    print("[OK] Triton loaded, inferred, compared, and unloaded all three models.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.example"))
    parser.add_argument("--url")
    parser.add_argument("--check", action="store_true", help="validate committed evidence only")
    arguments = parser.parse_args()
    try:
        if arguments.check:
            process = subprocess.run(
                [sys.executable, "scripts/validate_model_evidence.py"],
                cwd=REPOSITORY_ROOT,
                check=False,
            )
            return process.returncode
        env_file = arguments.env_file
        if not env_file.is_absolute():
            env_file = REPOSITORY_ROOT / env_file
        env = _load_env(env_file)
        base_url = arguments.url or f"http://127.0.0.1:{env['TRITON_HTTP_PORT']}"
        run_smoke(base_url.rstrip("/"), env_file)
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        SmokeError,
    ) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
