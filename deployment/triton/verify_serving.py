"""Verify Triton serving protocols, batching, versions, reload, and cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared.triton_model_config import render_load_config_json, validate_contract_relationships

MANIFEST_PATH = REPOSITORY_ROOT / "models/model-manifest.json"
SPEC_PATH = REPOSITORY_ROOT / "models/model-spec.yaml"
ENV_PATH = REPOSITORY_ROOT / ".env.example"
EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-4"
EVIDENCE_PATH = EVIDENCE_DIRECTORY / "serving-runtime.json"
REPOSITORY_EVIDENCE_PATH = EVIDENCE_DIRECTORY / "repository-versions.txt"
REQUIRED_EXTENSIONS = {"model_repository", "statistics", "model_configuration"}
MODEL_NAMES = ("resnet50_onnx", "resnet50_tensorrt", "yolo11n_onnx")
BURSTS = {
    "resnet50_onnx": (32, 16),
    "resnet50_tensorrt": (32, 16),
    "yolo11n_onnx": (8, 4),
}
MAX_BATCHING_ATTEMPTS = 3
PROTOCOL_ATOL = 1e-5
PROTOCOL_RTOL = 1e-5


class VerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"{path.name} must contain a JSON object")
    return value


def _load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _http_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise VerificationError(f"HTTP {error.code} for {path}: {detail}") from error
    return json.loads(body) if body else None


def _control(http_url: str, model: str, action: str, payload: dict[str, Any] | None = None) -> None:
    _http_json(
        http_url,
        f"/v2/repository/models/{model}/{action}",
        method="POST",
        payload={} if payload is None else payload,
    )


def _ready_rows(http_url: str) -> list[dict[str, Any]]:
    rows = _http_json(http_url, "/v2/repository/index", method="POST", payload={})
    return [row for row in rows if row.get("state") == "READY"]


def _synthetic(entry: dict[str, Any], batch: int, np: Any) -> Any:
    shape = [batch, *entry["input"]["shape"][1:]]
    return np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)


def _http_infer(client: Any, module: Any, name: str, version: str, entry: dict[str, Any], data: Any) -> tuple[Any, str]:
    item = module.InferInput(entry["input"]["name"], list(data.shape), "FP32")
    item.set_data_from_numpy(data, binary_data=True)
    output = module.InferRequestedOutput(entry["output"]["name"], binary_data=True)
    result = client.infer(name, [item], model_version=version, outputs=[output])
    return result.as_numpy(entry["output"]["name"]), str(result.get_response()["model_version"])


def _grpc_infer(client: Any, module: Any, name: str, version: str, entry: dict[str, Any], data: Any) -> tuple[Any, str]:
    item = module.InferInput(entry["input"]["name"], list(data.shape), "FP32")
    item.set_data_from_numpy(data)
    output = module.InferRequestedOutput(entry["output"]["name"])
    result = client.infer(name, [item], model_version=version, outputs=[output])
    return result.as_numpy(entry["output"]["name"]), str(result.get_response().model_version)


def _stats(http_url: str, name: str, version: str) -> dict[str, Any]:
    value = _http_json(http_url, f"/v2/models/{name}/versions/{version}/stats")
    rows = value.get("model_stats", [])
    if len(rows) != 1:
        raise VerificationError(f"Expected one statistics row for {name}:{version}")
    return rows[0]


def _counter(row: dict[str, Any], name: str) -> int:
    return int(row.get(name, 0))


def _batch_counts(row: dict[str, Any]) -> dict[int, int]:
    return {
        int(item["batch_size"]): int(item.get("compute_infer", {}).get("count", 0))
        for item in row.get("batch_stats", [])
    }


def _batching(
    http_url: str,
    http_module: Any,
    manifest: dict[str, Any],
    name: str,
    np: Any,
) -> dict[str, Any]:
    entry = manifest["models"][name]
    version = str(max(entry["version_policy"]["specific"]))
    requests, concurrency = BURSTS[name]
    payload = render_load_config_json(entry["model_config"], [int(version)])
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, MAX_BATCHING_ATTEMPTS + 1):
        _control(http_url, name, "unload")
        _control(http_url, name, "load", payload)
        baseline = _stats(http_url, name, version)
        client = http_module.InferenceServerClient(url=http_url, concurrency=concurrency)
        data = _synthetic(entry, 1, np)
        handles = []
        for request_id in range(requests):
            item = http_module.InferInput(entry["input"]["name"], list(data.shape), "FP32")
            item.set_data_from_numpy(data, binary_data=True)
            output = http_module.InferRequestedOutput(entry["output"]["name"], binary_data=True)
            handles.append(
                client.async_infer(
                    name,
                    [item],
                    model_version=version,
                    outputs=[output],
                    request_id=str(request_id),
                )
            )
        finite = True
        for handle in handles:
            result = handle.get_result()
            finite = finite and bool(np.isfinite(result.as_numpy(entry["output"]["name"])).all())
        after = _stats(http_url, name, version)
        before_batches = _batch_counts(baseline)
        after_batches = _batch_counts(after)
        observed = sorted(
            batch
            for batch, count in after_batches.items()
            if batch > 1 and count > before_batches.get(batch, 0)
        )
        record = {
            "attempt": attempt,
            "requests": requests,
            "concurrency": concurrency,
            "inference_count_delta": _counter(after, "inference_count") - _counter(baseline, "inference_count"),
            "execution_count_delta": _counter(after, "execution_count") - _counter(baseline, "execution_count"),
            "success_count_delta": int(after.get("inference_stats", {}).get("success", {}).get("count", 0))
            - int(baseline.get("inference_stats", {}).get("success", {}).get("count", 0)),
            "observed_batch_sizes": observed,
            "finite_outputs": finite,
        }
        record["passed"] = bool(
            record["success_count_delta"] == requests
            and record["inference_count_delta"] == requests
            and 0 < record["execution_count_delta"] < requests
            and observed
            and finite
        )
        attempts.append(record)
        if record["passed"]:
            return {"model_version": version, "attempts": attempts, **record}
    raise VerificationError(f"Dynamic batching was not observed for {name} in three attempts")


def _snapshot(rows: list[dict[str, Any]]) -> str:
    lines = ["model\tversion\tstate\treason"]
    for row in sorted(rows, key=lambda item: (item.get("name", ""), item.get("version", ""))):
        lines.append(
            f"{row.get('name', '')}\t{row.get('version', '')}\t{row.get('state', '')}\t{row.get('reason') or '-'}"
        )
    return "\n".join(lines) + "\n"


def _run(http_url: str, grpc_url: str) -> None:
    import numpy as np
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient

    manifest = _load_json(MANIFEST_PATH)
    env = _load_env()
    http = httpclient.InferenceServerClient(url=http_url, concurrency=16)
    grpc = grpcclient.InferenceServerClient(url=grpc_url)
    metadata_http = http.get_server_metadata()
    metadata_grpc = grpc.get_server_metadata()
    extensions = set(metadata_http.get("extensions", []))
    grpc_extensions = set(metadata_grpc.extensions)
    missing = REQUIRED_EXTENSIONS - extensions
    if missing or not REQUIRED_EXTENSIONS.issubset(grpc_extensions):
        raise VerificationError(f"Missing required server extensions: {sorted(missing)}")

    for name in MODEL_NAMES:
        _control(http_url, name, "unload")
        _control(http_url, name, "load")

    matrix: dict[str, Any] = {}
    for name in MODEL_NAMES:
        entry = manifest["models"][name]
        for version in entry["versions"]:
            data = _synthetic(entry, 1, np)
            http_metadata = http.get_model_metadata(name, version)
            grpc_metadata = grpc.get_model_metadata(name, version)
            if http_metadata["name"] != name or grpc_metadata.name != name:
                raise VerificationError(f"Metadata name mismatch for {name}:{version}")
            http_output, http_version = _http_infer(http, httpclient, name, version, entry, data)
            grpc_output, grpc_version = _grpc_infer(grpc, grpcclient, name, version, entry, data)
            expected_shape = [1, *entry["output"]["shape"][1:]]
            if list(http_output.shape) != expected_shape or list(grpc_output.shape) != expected_shape:
                raise VerificationError(f"Output shape mismatch for {name}:{version}")
            if http_version != version or grpc_version != version:
                raise VerificationError(f"Response version mismatch for {name}:{version}")
            difference = np.abs(http_output - grpc_output)
            maximum_protocol_error = float(difference.max())
            if not np.isfinite(http_output).all() or not np.isfinite(grpc_output).all():
                raise VerificationError(f"Non-finite protocol output for {name}:{version}")
            if not np.allclose(
                http_output,
                grpc_output,
                rtol=PROTOCOL_RTOL,
                atol=PROTOCOL_ATOL,
            ):
                raise VerificationError(
                    f"HTTP/gRPC numerical parity failed for {name}:{version}; "
                    f"max_abs_error={maximum_protocol_error}"
                )
            matrix[f"{name}:{version}"] = {
                "http": "passed",
                "grpc": "passed",
                "response_version": version,
                "output_shape": expected_shape,
                "output_dtype": str(http_output.dtype).upper(),
                "max_abs_protocol_error": maximum_protocol_error,
            }

    dynamic = {
        name: _batching(http_url, httpclient, manifest, name, np) for name in MODEL_NAMES
    }

    name = "resnet50_onnx"
    entry = manifest["models"][name]
    selected: list[str] = []
    for version in ("1", "2"):
        _control(http_url, name, "unload")
        _control(http_url, name, "load", render_load_config_json(entry["model_config"], [int(version)]))
        _, selected_version = _http_infer(http, httpclient, name, "", entry, _synthetic(entry, 1, np))
        if selected_version != version:
            raise VerificationError(f"Default selection expected {version}, got {selected_version}")
        selected.append(selected_version)
    _control(http_url, name, "unload")
    _control(http_url, name, "load")
    repository_rows = _ready_rows(http_url)
    ready_resnet = {row.get("version") for row in repository_rows if row.get("name") == name}
    if ready_resnet != {"1", "2"}:
        raise VerificationError(f"Tracked policy did not load both ResNet versions: {ready_resnet}")
    _write(REPOSITORY_EVIDENCE_PATH, _snapshot(repository_rows))
    data = _synthetic(entry, 1, np)
    output_one, _ = _http_infer(http, httpclient, name, "1", entry, data)
    output_two, _ = _http_infer(http, httpclient, name, "2", entry, data)
    _, default_version = _http_infer(http, httpclient, name, "", entry, data)
    if default_version != "2" or not np.array_equal(output_one, output_two):
        raise VerificationError("Tracked ResNet version policy or v1/v2 parity failed")
    _control(http_url, name, "load")
    _, reload_version = _http_infer(http, httpclient, name, "", entry, data)
    if reload_version != "2":
        raise VerificationError("In-place reload without unload failed")

    for model in MODEL_NAMES:
        _control(http_url, model, "unload")
    cleanup_deadline = time.monotonic() + 30
    while time.monotonic() < cleanup_deadline:
        final_ready = _ready_rows(http_url)
        if not final_ready and http.is_server_live() and http.is_server_ready():
            break
        time.sleep(0.25)
    if final_ready or not http.is_server_live() or not http.is_server_ready():
        raise VerificationError("Cleanup did not leave an empty ready repository and healthy server")

    evidence = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_path": "models/model-manifest.json",
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "spec_path": "models/model-spec.yaml",
        "spec_sha256": _sha256(SPEC_PATH),
        "images": {"server": env["TRITON_IMAGE"], "sdk": env["TRITON_SDK_IMAGE"]},
        "extensions": sorted(REQUIRED_EXTENSIONS),
        "protocols": {"http": "passed", "grpc": "passed"},
        "models": matrix,
        "dynamic_batching": dynamic,
        "version_switching": {
            "sequence": selected + ["1+2"],
            "default_with_tracked_policy": default_version,
            "version_parity": "passed",
            "reload_without_unload": "passed",
            "passed": True,
        },
        "final_repository_ready_models": [],
        "server_after_cleanup": {"live": True, "ready": True},
    }
    _write(EVIDENCE_PATH, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("[OK] HTTP/gRPC, dynamic batching, version switching, reload, and cleanup passed.")


def _check() -> None:
    manifest = _load_json(MANIFEST_PATH)
    if set(manifest.get("models", {})) != set(MODEL_NAMES):
        raise VerificationError("Manifest serving model set is invalid")
    for name, entry in manifest["models"].items():
        errors = validate_contract_relationships(entry["model_config"])
        if errors:
            raise VerificationError(f"{name} ModelConfig: {'; '.join(errors)}")
    print("[OK] Serving verifier contracts are locally importable and current.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-url", default="triton:8000")
    parser.add_argument("--grpc-url", default="triton:8001")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.check:
            _check()
        else:
            _run(arguments.http_url, arguments.grpc_url)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, VerificationError) as error:
        if not arguments.check:
            for model in MODEL_NAMES:
                try:
                    _control(arguments.http_url, model, "unload")
                except (OSError, ValueError, TypeError, VerificationError):
                    pass
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
