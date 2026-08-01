#!/usr/bin/env python3
"""Run the step 5 real-image client matrix and capture sanitized evidence."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != SCRIPT_DIRECTORY]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from client.inference_client import main as client_main
from client.logging.csv_export import read_events
from client.transport import HttpTransport, RepositoryController

EVIDENCE_DIRECTORY = REPOSITORY_ROOT / "docs/evidence/step-5"
EVIDENCE_PATH = EVIDENCE_DIRECTORY / "client-runtime.json"
JSONL_PATH = EVIDENCE_DIRECTORY / "inference-log.jsonl"
CSV_PATH = EVIDENCE_DIRECTORY / "inference-log.csv"
PREDICTIONS_PATH = EVIDENCE_DIRECTORY / "predictions.txt"
CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"
SAMPLES_MANIFEST_PATH = REPOSITORY_ROOT / "client/samples/manifest.json"
REQUIREMENTS_PATH = REPOSITORY_ROOT / "requirements.txt"
FINGERPRINT_PATHS = (
    "client/inference_client.py",
    "client/input_loader.py",
    "client/preprocessing.py",
    "client/postprocessing.py",
    "client/transport.py",
    "client/logging/writer.py",
    "client/logging/csv_export.py",
    "client/verify_runtime.py",
    "client/client-config.yaml",
    "schemas/inference-event.schema.json",
    "schemas/client-model-contracts.schema.json",
    "schemas/client-runtime-evidence.schema.json",
    "shared/client-model-contracts.json",
    "client/samples/manifest.json",
    "requirements.txt",
)


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in FINGERPRINT_PATHS:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((REPOSITORY_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _ready_rows(values: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"model": model, "version": version}
        for model, version in sorted(values, key=lambda item: (item[0], int(item[1])))
    ]


def _run(label: str, arguments: list[str], transcript: list[str]) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = client_main(arguments)
    transcript.append(f"=== {label} ===")
    transcript.extend(stdout.getvalue().rstrip().splitlines())
    if code != 0:
        detail = stderr.getvalue().strip() or stdout.getvalue().strip()
        raise VerificationError(f"{label} failed: {detail}")


def _validate_initial_state(
    initial: set[tuple[str, str]], contract: dict[str, Any]
) -> None:
    for model, entry in contract["models"].items():
        expected = set(entry["versions"])
        ready = {version for name, version in initial if name == model}
        if ready and ready != expected:
            raise VerificationError(
                f"Initial READY state for {model} is partial; exact restoration is not possible"
            )


def _wait_for_server_health(transport: HttpTransport, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_health: dict[str, bool] = {}
    while True:
        last_health = transport.health()
        if last_health == {"live": True, "ready": True}:
            return
        if time.monotonic() >= deadline:
            raise VerificationError(
                f"Triton did not become live and ready: {last_health}"
            )
        time.sleep(0.25)


def _case_arguments(
    command: str,
    input_path: str,
    model: str,
    version: str,
    protocol: str,
    batch_size: int,
) -> list[str]:
    return [
        command,
        input_path,
        "--model",
        model,
        "--version",
        version,
        "--protocol",
        protocol,
        "--batch-size",
        str(batch_size),
        "--log-file",
        "docs/evidence/step-5/inference-log.jsonl",
    ]


def _runtime_matrix(transcript: list[str]) -> None:
    sample_directory = "client/samples"
    sample = "client/samples/04_bus.jpg"
    cases = [
        (
            "ResNet ONNX v1 / HTTP / all samples",
            _case_arguments("classify", sample_directory, "resnet50_onnx", "1", "http", 4),
        ),
        (
            "ResNet ONNX v2 / gRPC",
            _case_arguments("classify", sample, "resnet50_onnx", "2", "grpc", 1),
        ),
        (
            "ResNet TensorRT v1 / HTTP",
            _case_arguments("classify", sample, "resnet50_tensorrt", "1", "http", 1),
        ),
        (
            "YOLO ONNX v1 / HTTP / all samples",
            _case_arguments("detect", sample_directory, "yolo11n_onnx", "1", "http", 2),
        ),
        (
            "YOLO ONNX v1 / gRPC",
            _case_arguments("detect", sample, "yolo11n_onnx", "1", "grpc", 1),
        ),
    ]
    for label, arguments in cases:
        _run(label, arguments, transcript)


def verify() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    samples_manifest = json.loads(SAMPLES_MANIFEST_PATH.read_text(encoding="utf-8"))
    controller = RepositoryController("127.0.0.1:8000", 120.0)
    health_transport = HttpTransport("127.0.0.1:8000", 30.0)
    _wait_for_server_health(health_transport, 30.0)
    initial = controller.ready_set()
    _validate_initial_state(initial, contract)
    initially_ready_models = {model for model, _ in initial}
    transcript: list[str] = []
    EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.unlink(missing_ok=True)
    JSONL_PATH.unlink(missing_ok=True)
    CSV_PATH.unlink(missing_ok=True)
    PREDICTIONS_PATH.unlink(missing_ok=True)
    try:
        _runtime_matrix(transcript)
        _run(
            "CSV export",
            [
                "export-logs",
                "--input-log",
                "docs/evidence/step-5/inference-log.jsonl",
                "--output-csv",
                "docs/evidence/step-5/inference-log.csv",
            ],
            transcript,
        )
        _write(PREDICTIONS_PATH, "\n".join(transcript).rstrip() + "\n")
    finally:
        current = controller.ready_set()
        loaded_models = {model for model, _ in current} - initially_ready_models
        for model in sorted(loaded_models):
            controller.unload(model)
    final = controller.ready_set()
    if final != initial:
        raise VerificationError(
            f"READY state was not restored: initial={sorted(initial)}, final={sorted(final)}"
        )
    _wait_for_server_health(health_transport, 30.0)
    events = read_events(JSONL_PATH)
    with CSV_PATH.open(encoding="utf-8", newline="") as stream:
        csv_rows = sum(1 for _ in csv.DictReader(stream))
    detections = sum(
        len(prediction["items"])
        for event in events
        if event["model"] == "yolo11n_onnx" and event["status"] == "success"
        for prediction in event["predictions"]
    )
    evidence = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_fingerprint_sha256": source_fingerprint(),
        "client_contract_sha256": sha256(CONTRACT_PATH),
        "samples_manifest_sha256": sha256(SAMPLES_MANIFEST_PATH),
        "requirements_sha256": sha256(REQUIREMENTS_PATH),
        "samples_count": len(samples_manifest["samples"]),
        "classification": {"http": "passed", "grpc": "passed", "tensorrt": "passed"},
        "detection": {"http": "passed", "grpc": "passed", "detections": detections},
        "logging": {
            "jsonl_events": len(events),
            "csv_rows": csv_rows,
            "jsonl_sha256": sha256(JSONL_PATH),
            "csv_sha256": sha256(CSV_PATH),
            "predictions_sha256": sha256(PREDICTIONS_PATH),
        },
        "initial_ready": _ready_rows(initial),
        "final_ready": _ready_rows(final),
        "runtime_state_restored": True,
    }
    _write(EVIDENCE_PATH, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main() -> int:
    try:
        evidence = verify()
    except Exception as error:
        print(f"[FAIL] Client runtime verification: {error}", file=sys.stderr)
        return 1
    print(
        "[OK] Client runtime verification passed: "
        f"{evidence['logging']['jsonl_events']} events, "
        f"{evidence['detection']['detections']} detections, READY state restored."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
