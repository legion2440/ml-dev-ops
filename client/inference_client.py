#!/usr/bin/env python3
"""Production image client for Triton HTTP/gRPC inference and persistent logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != SCRIPT_DIRECTORY]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import yaml

from client.input_loader import InputError, batches, discover_images, load_image
from client.logging.csv_export import export_csv
from client.logging.writer import LogError, append_event, sanitize_error
from client.postprocessing import classification_predictions, detection_predictions
from client.preprocessing import preprocess_classification, preprocess_detection
from client.transport import RepositoryController, TransportError, create_transport

CONFIG_PATH = REPOSITORY_ROOT / "client/client-config.yaml"
CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"


class ClientError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClientError(f"{path.name} must contain a mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClientError(f"{path.name} must contain a JSON object")
    return value


def _milliseconds(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 6)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _model_contract(
    contract: dict[str, Any], model: str, task: str, version: str | None, batch_size: int
) -> dict[str, Any]:
    models = contract.get("models", {})
    if model not in models:
        raise ClientError(f"Unknown client model: {model}")
    entry = models[model]
    if entry["task"] != task:
        raise ClientError(f"Model {model} is not a {task} model")
    if version is not None and version not in entry["versions"]:
        raise ClientError(
            f"Version {version} is not available for {model}; expected one of {entry['versions']}"
        )
    if not 1 <= batch_size <= int(entry["max_batch_size"]):
        raise ClientError(
            f"Batch size {batch_size} exceeds {model} maximum {entry['max_batch_size']}"
        )
    return entry


def _transport(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    http_url = args.http_url or config["endpoints"]["http"]
    grpc_url = args.grpc_url or config["endpoints"]["grpc"]
    return create_transport(args.protocol, http_url, grpc_url, args.timeout)


def _prediction_sets(names: list[str], rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {"input_name": name, "items": predictions}
        for name, predictions in zip(names, rows, strict=True)
    ]


def _sanitized_request_error(error: Exception, images: list[Any]) -> str:
    message = str(error)
    for image in images:
        candidates = {str(image.path), image.path.as_posix(), str(image.path.resolve())}
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate:
                message = message.replace(candidate, image.name)
    return sanitize_error(message)


def _event(
    *,
    request_id: str,
    model: str,
    requested_version: str | None,
    resolved_version: str | None,
    protocol: str,
    images: list[Any],
    timing: dict[str, float],
    status: str,
    predictions: list[dict[str, Any]] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": _timestamp(),
        "request_id": request_id,
        "model": model,
        "requested_version": requested_version,
        "resolved_version": resolved_version,
        "protocol": protocol,
        "inputs": [image.log_metadata() for image in images],
        "batch_size": len(images),
        "timing_ms": timing,
        "status": status,
        "predictions": predictions,
        "error": error,
    }


def _run_inference(
    args: argparse.Namespace,
    config: dict[str, Any],
    contract: dict[str, Any],
    task: str,
) -> list[dict[str, Any]]:
    entry = _model_contract(contract, args.model, task, args.version, args.batch_size)
    if args.timeout <= 0:
        raise ClientError("Timeout must be positive")
    if task == "classification":
        if not 1 <= args.top_k <= len(entry["labels"]):
            raise ClientError(f"top-k must be between 1 and {len(entry['labels'])}")
    elif (
        not 0 <= args.confidence <= 1
        or not 0 <= args.iou <= 1
        or args.max_detections < 1
    ):
        raise ClientError("Detection thresholds must be within [0, 1] and limit positive")
    paths = discover_images(Path(args.input))
    images = [load_image(path) for path in paths]
    transport = _transport(args, config)
    http_url = args.http_url or config["endpoints"]["http"]
    controller = RepositoryController(http_url, args.timeout)
    controller.ensure_ready(
        transport,
        args.model,
        args.version,
        auto_load=not args.no_auto_load,
    )
    log_path = _resolve_path(args.log_file)
    all_predictions: list[dict[str, Any]] = []
    for image_batch in batches(images, args.batch_size):
        total_start = time.perf_counter()
        preprocessing_start = time.perf_counter()
        if task == "classification":
            tensor = preprocess_classification(image_batch, entry)
            geometry = None
        else:
            tensor, geometry = preprocess_detection(image_batch, entry)
        preprocessing_ms = _milliseconds(preprocessing_start)
        request_id = str(uuid.uuid4())
        request_ms = 0.0
        postprocessing_ms = 0.0
        try:
            request_start = time.perf_counter()
            result = transport.infer(
                model=args.model,
                version=args.version,
                input_name=entry["input"]["name"],
                output_name=entry["output"]["name"],
                tensor=tensor,
                request_id=request_id,
            )
            request_ms = _milliseconds(request_start)
            if result.model_name != args.model:
                raise ClientError("Triton returned an unexpected model name")
            if result.model_version not in entry["versions"]:
                raise ClientError("Triton returned a version outside the client contract")
            if args.version is not None and result.model_version != args.version:
                raise ClientError("Triton returned a version different from the requested version")
            postprocessing_start = time.perf_counter()
            if task == "classification":
                prediction_rows = classification_predictions(
                    result.output, entry["labels"], args.top_k
                )
            else:
                prediction_rows = detection_predictions(
                    result.output,
                    geometry,
                    entry["labels"],
                    args.confidence,
                    args.iou,
                    args.max_detections,
                )
            postprocessing_ms = _milliseconds(postprocessing_start)
            prediction_sets = _prediction_sets(
                [image.name for image in image_batch], prediction_rows
            )
            timing = {
                "preprocessing": preprocessing_ms,
                "request": request_ms,
                "postprocessing": postprocessing_ms,
                "total": _milliseconds(total_start),
            }
            append_event(
                log_path,
                _event(
                    request_id=request_id,
                    model=args.model,
                    requested_version=args.version,
                    resolved_version=result.model_version,
                    protocol=args.protocol,
                    images=image_batch,
                    timing=timing,
                    status="success",
                    predictions=prediction_sets,
                    error=None,
                ),
            )
            all_predictions.extend(prediction_sets)
        except Exception as error:
            timing = {
                "preprocessing": preprocessing_ms,
                "request": request_ms,
                "postprocessing": postprocessing_ms,
                "total": _milliseconds(total_start),
            }
            append_event(
                log_path,
                _event(
                    request_id=request_id,
                    model=args.model,
                    requested_version=args.version,
                    resolved_version=None,
                    protocol=args.protocol,
                    images=image_batch,
                    timing=timing,
                    status="error",
                    predictions=None,
                    error=_sanitized_request_error(error, image_batch),
                ),
            )
            raise
    return all_predictions


def _print_predictions(task: str, predictions: list[dict[str, Any]]) -> None:
    for prediction_set in predictions:
        print(prediction_set["input_name"])
        if not prediction_set["items"]:
            print("  no detections")
            continue
        for item in prediction_set["items"]:
            if task == "classification":
                print(f"  {item['rank']}. {item['label']:<28} {item['probability']:.4f}")
            else:
                box = ", ".join(f"{value:.1f}" for value in item["box_xyxy"])
                print(f"  {item['label']:<20} {item['confidence']:.4f} [{box}]")


def _add_connection_options(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    parser.add_argument(
        "--protocol", choices=("http", "grpc"), default=config["defaults"]["protocol"]
    )
    parser.add_argument("--timeout", type=float, default=config["defaults"]["timeout_seconds"])
    parser.add_argument("--http-url", help=argparse.SUPPRESS)
    parser.add_argument("--grpc-url", help=argparse.SUPPRESS)


def _add_inference_parser(
    subparsers: Any, command: str, task: str, config: dict[str, Any]
) -> None:
    parser = subparsers.add_parser(command, help=f"Run {task} inference")
    parser.set_defaults(action="infer", task=task)
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--model", default=config["models"][task])
    parser.add_argument("--version")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-auto-load", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-file", default=config["defaults"]["log_path"])
    _add_connection_options(parser, config)
    if task == "classification":
        parser.add_argument("--top-k", type=int, default=config["classification"]["top_k"])
    else:
        parser.add_argument(
            "--confidence",
            type=float,
            default=config["detection"]["confidence_threshold"],
        )
        parser.add_argument("--iou", type=float, default=config["detection"]["iou_threshold"])
        parser.add_argument(
            "--max-detections",
            type=int,
            default=config["detection"]["max_detections"],
        )


def _parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    health = subparsers.add_parser("health", help="Check Triton live and ready state")
    health.set_defaults(action="health")
    _add_connection_options(health, config)
    metadata = subparsers.add_parser("metadata", help="Read model metadata")
    metadata.set_defaults(action="metadata")
    metadata.add_argument("--model", required=True)
    metadata.add_argument("--version")
    _add_connection_options(metadata, config)
    _add_inference_parser(subparsers, "classify", "classification", config)
    _add_inference_parser(subparsers, "detect", "detection", config)
    export = subparsers.add_parser("export-logs", help="Export JSONL events to CSV")
    export.set_defaults(action="export")
    export.add_argument("--input-log", default=config["defaults"]["log_path"])
    export.add_argument("--output", default="logs/inference.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    config = _load_yaml(CONFIG_PATH)
    parser = _parser(config)
    args = parser.parse_args(argv)
    if not hasattr(args, "action"):
        parser.print_help()
        return 0
    try:
        if args.action == "health":
            print(json.dumps(_transport(args, config).health(), indent=2, sort_keys=True))
            return 0
        if args.action == "metadata":
            print(
                json.dumps(
                    _transport(args, config).metadata(args.model, args.version),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "export":
            count = export_csv(_resolve_path(args.input_log), _resolve_path(args.output))
            print(f"Exported {count} inference events to {args.output}")
            return 0
        contract = _load_json(CONTRACT_PATH)
        predictions = _run_inference(args, config, contract, args.task)
        if args.json:
            print(
                json.dumps(
                    {"model": args.model, "protocol": args.protocol, "predictions": predictions},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_predictions(args.task, predictions)
        return 0
    except Exception as error:
        print(f"ERROR: {sanitize_error(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
