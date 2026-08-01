"""Reproducibly prepare, inspect, and clean the step 3 model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared.triton_model_config import (  # noqa: E402
    build_model_config,
    render_pbtxt,
)
SPEC_PATH = REPOSITORY_ROOT / "models/model-spec.yaml"
LOCK_PATH = REPOSITORY_ROOT / "scripts/model_preparation/requirements.lock"
MANIFEST_PATH = REPOSITORY_ROOT / "models/model-manifest.json"
CLIENT_CONTRACT_PATH = REPOSITORY_ROOT / "shared/client-model-contracts.json"
STEP3_MANIFEST_SNAPSHOT_PATH = (
    REPOSITORY_ROOT / "docs/evidence/step-3/model-manifest-v1.json"
)
PREPARATION_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "docs/evidence/step-3/preparation.json"
)
CACHE_DIRECTORY = REPOSITORY_ROOT / ".cache/model-preparation"
SOURCE_DIRECTORY = CACHE_DIRECTORY / "sources"
INSPECTION_PATH = CACHE_DIRECTORY / "onnx-inspection.json"
EXPORTER_DOCKERFILE = (
    REPOSITORY_ROOT / "scripts/model_preparation/Dockerfile.exporter"
)
IMAGE_REFERENCE = re.compile(
    r"^(?P<repository>[a-z0-9./_-]+):(?P<tag>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"@sha256:(?P<digest>[0-9a-f]{64})$"
)
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+) \\$")
MODEL_KEYS = ("resnet50", "yolo11n")
SERVING_MODELS = ("resnet50_onnx", "resnet50_tensorrt", "yolo11n_onnx")
SERVING_MAPPING = {
    "resnet50_onnx": ("resnet50", "onnx", "onnxruntime_onnx"),
    "resnet50_tensorrt": ("resnet50", "tensorrt", "tensorrt_plan"),
    "yolo11n_onnx": ("yolo11n", "onnx", "onnxruntime_onnx"),
}


class PreparationError(RuntimeError):
    """A reproducibility or artifact preparation invariant failed."""


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PreparationError(f"Cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{path} must contain a YAML object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_client_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project the artifact manifest into a repository-independent client contract."""
    models: dict[str, Any] = {}
    for name, entry in manifest["models"].items():
        labels_path = REPOSITORY_ROOT / "models" / name / entry["labels"]["filename"]
        labels = labels_path.read_text(encoding="utf-8").splitlines()
        if len(labels) != entry["labels"]["count"]:
            raise PreparationError(f"{name} client label count is stale")
        output_shape = entry["output"]["shape"]
        if len(output_shape) == 2 and output_shape[-1] == len(labels):
            task = "classification"
            output_semantics = {"kind": "logits"}
            preprocessing = {
                **entry["preprocessing"],
                "channel_order": "RGB",
                "tensor_layout": "CHW",
            }
        elif len(output_shape) == 3 and output_shape[1] == len(labels) + 4:
            task = "detection"
            output_semantics = {
                "kind": "yolo_xywh_class_scores",
                "box_format": "xywh",
                "class_scores_start": 4,
                "has_objectness": False,
                "class_aware_nms": True,
            }
            preprocessing = {
                **entry["preprocessing"],
                "resize_mode": "letterbox",
                "letterbox_center": True,
                "padding_value": 114,
                "tensor_layout": "CHW",
            }
        else:
            raise PreparationError(f"Cannot derive client task semantics for {name}")
        models[name] = {
            "task": task,
            "max_batch_size": entry["max_batch_size"],
            "versions": sorted(entry["versions"], key=int),
            "input": entry["input"],
            "output": entry["output"],
            "preprocessing": preprocessing,
            "labels": labels,
            "output_semantics": output_semantics,
        }
    return {
        "schema_version": 1,
        "source_manifest_sha256": hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest(),
        "models": models,
    }


def generate_client_contract() -> None:
    if not MANIFEST_PATH.is_file():
        raise PreparationError("models/model-manifest.json is required")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _write_text(CLIENT_CONTRACT_PATH, canonical_json(render_client_contract(manifest)))
    print("[OK] Generated repository-independent client model contract.")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _run(
    arguments: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if check and process.returncode != 0:
        detail = ""
        if capture_output:
            detail = process.stderr.strip() or process.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise PreparationError(f"Command failed ({process.returncode}): {arguments[0]}{suffix}")
    return process


def _docker_mount() -> str:
    return f"{REPOSITORY_ROOT.resolve()}:/workspace"


def _source_cache_path(source: dict[str, Any]) -> Path:
    return SOURCE_DIRECTORY / str(source["filename"])


def _label_source_path(labels: dict[str, Any]) -> Path:
    filename = Path(urllib.parse.urlparse(str(labels["source_url"])).path).name
    return SOURCE_DIRECTORY / filename


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ml-dev-ops-step-3"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_download(url: str, destination: Path, expected_sha256: str | None) -> str:
    if not destination.is_file():
        print(f"[DOWNLOAD] {url}")
        _download(url, destination)
    actual_sha256 = sha256_file(destination)
    if expected_sha256 and actual_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise PreparationError(
            f"SHA-256 mismatch for {destination.name}: {actual_sha256}, "
            f"expected {expected_sha256}"
        )
    return actual_sha256


def discover_sources(spec: dict[str, Any]) -> None:
    """Download source candidates and print hashes without changing the spec."""
    for model_key in MODEL_KEYS:
        model = spec["models"][model_key]
        source = model["source"]
        source_hash = _ensure_download(
            str(source["url"]), _source_cache_path(source), None
        )
        print(f"{model_key}.source sha256={source_hash}")
        labels = model["labels"]
        labels_hash = _ensure_download(
            str(labels["source_url"]), _label_source_path(labels), None
        )
        print(f"{model_key}.labels.source sha256={labels_hash}")


def download_sources(spec: dict[str, Any]) -> None:
    for model_key in MODEL_KEYS:
        model = spec["models"][model_key]
        source = model["source"]
        if source.get("hash_status") != "resolved" or not FULL_SHA256.fullmatch(
            str(source.get("sha256", ""))
        ):
            raise PreparationError(
                f"{model_key} source hash must be explicitly accepted as resolved"
            )
        _ensure_download(
            str(source["url"]),
            _source_cache_path(source),
            str(source["sha256"]),
        )
        labels = model["labels"]
        _ensure_download(
            str(labels["source_url"]),
            _label_source_path(labels),
            str(labels["source_sha256"]),
        )
    print("[OK] Source weights and label sources match accepted SHA-256 values.")


def serving_version_path(serving: dict[str, Any], version: str = "1") -> str:
    return str(serving["versions"][str(version)]["artifact_path"])


def serving_artifact_paths(spec: dict[str, Any]) -> list[str]:
    return [
        str(version["artifact_path"])
        for model in spec["models"].values()
        for serving in model["serving"].values()
        for version in serving["versions"].values()
    ]


def serving_model_config(spec: dict[str, Any], serving_name: str) -> dict[str, Any]:
    try:
        logical_key, serving_kind, platform = SERVING_MAPPING[serving_name]
    except KeyError as error:
        raise PreparationError(f"Unknown serving model: {serving_name}") from error
    model = spec["models"][logical_key]
    serving = model["serving"][serving_kind]
    capability = (
        str(spec["build"]["target"]["compute_capability"])
        if serving_kind == "tensorrt"
        else None
    )
    return build_model_config(
        model, serving, platform=platform, compute_capability=capability
    )


def render_config(spec: dict[str, Any], serving_name: str) -> str:
    return render_pbtxt(serving_model_config(spec, serving_name))


def _generated_label_content(
    model_key: str, source_path: Path, expected_count: int
) -> str:
    if model_key == "resnet50":
        lines = source_path.read_text(encoding="utf-8").splitlines()
    else:
        import yaml

        try:
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise PreparationError(f"Cannot parse COCO labels: {error}") from error
        names = source.get("names") if isinstance(source, dict) else None
        if not isinstance(names, dict):
            raise PreparationError("COCO label source does not contain a names mapping")
        lines = [str(names[index]) for index in range(expected_count)]
    return "".join(f"{line}\n" for line in lines)


def generate_repository_text(spec: dict[str, Any]) -> None:
    for serving_name in SERVING_MODELS:
        config_path = REPOSITORY_ROOT / "models" / serving_name / "config.pbtxt"
        _write_text(config_path, render_config(spec, serving_name))

    label_targets = {
        "resnet50": ("resnet50_onnx", "resnet50_tensorrt"),
        "yolo11n": ("yolo11n_onnx",),
    }
    for model_key, serving_names in label_targets.items():
        labels = spec["models"][model_key]["labels"]
        content = _generated_label_content(
            model_key, _label_source_path(labels), int(labels["count"])
        )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash != labels["generated_sha256"]:
            raise PreparationError(f"Generated {model_key} labels have an unexpected SHA-256")
        if len(content.splitlines()) != labels["count"]:
            raise PreparationError(f"Generated {model_key} label count is incorrect")
        for serving_name in serving_names:
            _write_text(
                REPOSITORY_ROOT / "models" / serving_name / labels["filename"],
                content,
            )
    print("[OK] Triton configs and model-local label files were generated from the spec.")


def _internal_export(spec: dict[str, Any]) -> None:
    import onnx
    import torch
    import torchvision
    from torchvision.models import resnet50

    from scripts.model_preparation.yolo_export_adapter import export_yolo11n

    resnet = spec["models"]["resnet50"]
    weights_path = _source_cache_path(resnet["source"])
    model = resnet50(weights=None).eval().float()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    resnet_serving = resnet["serving"]["onnx"]
    onnx_path = REPOSITORY_ROOT / serving_version_path(resnet_serving, "1")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, *resnet["input"]["shape"][1:]), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            export_params=True,
            opset_version=int(spec["build"]["onnx_opset"]),
            do_constant_folding=True,
            input_names=[resnet["input"]["name"]],
            output_names=[resnet["output"]["name"]],
            dynamic_axes={
                resnet["input"]["name"]: {0: "batch"},
                resnet["output"]["name"]: {0: "batch"},
            },
        )

    version_two_path = REPOSITORY_ROOT / serving_version_path(resnet_serving, "2")
    version_two = onnx.load(str(onnx_path))
    public_output = version_two.graph.output[0].name
    internal_output = f"{public_output}__identity_v2_input"
    producers = [
        (node, index)
        for node in version_two.graph.node
        for index, output in enumerate(node.output)
        if output == public_output
    ]
    if len(producers) != 1:
        raise PreparationError("ResNet public ONNX output must have exactly one producer")
    if any(public_output in node.input for node in version_two.graph.node):
        raise PreparationError("ResNet public ONNX output must be a terminal tensor")
    producer, output_index = producers[0]
    producer.output[output_index] = internal_output
    version_two.graph.node.append(
        onnx.helper.make_node(
            "Identity",
            inputs=[internal_output],
            outputs=[public_output],
            name="serving_revision_identity_output",
        )
    )
    onnx.checker.check_model(version_two)
    version_two_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(version_two, str(version_two_path))
    if sha256_file(onnx_path) == sha256_file(version_two_path):
        raise PreparationError("ResNet ONNX versions must have different artifacts")

    yolo = spec["models"]["yolo11n"]
    export_yolo11n(
        _source_cache_path(yolo["source"]),
        REPOSITORY_ROOT / serving_version_path(yolo["serving"]["onnx"], "1"),
        opset=int(spec["build"]["onnx_opset"]),
        input_name=str(yolo["input"]["name"]),
        output_name=str(yolo["output"]["name"]),
        input_shape=list(yolo["input"]["shape"]),
        output_shape=list(yolo["output"]["shape"]),
    )
    print(f"[OK] Exported with torch={torch.__version__}, torchvision={torchvision.__version__}.")


def _onnx_dimensions(value_info: Any) -> list[int | str]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(int(dimension.dim_value))
    return dimensions


def _internal_inspect_onnx(spec: dict[str, Any]) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    inspection: dict[str, Any] = {}
    sessions: dict[str, Any] = {}
    mapping = [
        ("resnet50_onnx", "1", spec["models"]["resnet50"]),
        ("resnet50_onnx", "2", spec["models"]["resnet50"]),
        ("yolo11n_onnx", "1", spec["models"]["yolo11n"]),
    ]
    for serving_name, version, model_spec in mapping:
        serving = model_spec["serving"]["onnx"]
        artifact_path = REPOSITORY_ROOT / serving_version_path(serving, version)
        model = onnx.load(str(artifact_path))
        onnx.checker.check_model(model)
        graph_input = model.graph.input[0]
        graph_output = model.graph.output[0]
        input_shape = _onnx_dimensions(graph_input)
        output_shape = _onnx_dimensions(graph_output)
        expected_input = ["batch", *model_spec["input"]["shape"][1:]]
        expected_output = ["batch", *model_spec["output"]["shape"][1:]]
        if input_shape != expected_input or output_shape != expected_output:
            raise PreparationError(
                f"{serving_name} graph contract {input_shape}->{output_shape} does not match "
                f"{expected_input}->{expected_output}"
            )
        session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
        sessions[f"{serving_name}:{version}"] = session
        for batch in model_spec["smoke_batches"]:
            input_array = np.linspace(
                0.0,
                1.0,
                num=int(np.prod([batch, *model_spec["input"]["shape"][1:]])),
                dtype=np.float32,
            ).reshape([batch, *model_spec["input"]["shape"][1:]])
            output_array = session.run(
                [model_spec["output"]["name"]],
                {model_spec["input"]["name"]: input_array},
            )[0]
            if list(output_array.shape) != [batch, *model_spec["output"]["shape"][1:]]:
                raise PreparationError(
                    f"{serving_name}:{version} ONNX Runtime output shape is incorrect"
                )
            if not np.isfinite(output_array).all():
                raise PreparationError(
                    f"{serving_name}:{version} ONNX Runtime output is not finite"
                )
        inspection[f"{serving_name}:{version}"] = {
            "ir_version": int(model.ir_version),
            "opset": max(int(item.version) for item in model.opset_import),
            "input": {"name": graph_input.name, "shape": input_shape},
            "output": {"name": graph_output.name, "shape": output_shape},
            "onnx_checker": "passed",
            "onnxruntime_batches": list(model_spec["smoke_batches"]),
        }

    resnet = spec["models"]["resnet50"]
    version_parity: dict[str, Any] = {"status": "passed", "batches": {}}
    tolerances = resnet["version_parity"]
    for batch in resnet["smoke_batches"]:
        input_array = np.linspace(
            0.0,
            1.0,
            num=int(np.prod([batch, *resnet["input"]["shape"][1:]])),
            dtype=np.float32,
        ).reshape([batch, *resnet["input"]["shape"][1:]])
        outputs = [
            sessions[f"resnet50_onnx:{version}"].run(
                [resnet["output"]["name"]], {resnet["input"]["name"]: input_array}
            )[0]
            for version in ("1", "2")
        ]
        difference = np.abs(outputs[0] - outputs[1])
        denominator = float(np.linalg.norm(outputs[0].ravel()) * np.linalg.norm(outputs[1].ravel()))
        metrics = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "cosine_similarity": float(np.dot(outputs[0].ravel(), outputs[1].ravel()) / denominator),
            "top1_agreement": float(np.mean(np.argmax(outputs[0], axis=1) == np.argmax(outputs[1], axis=1))),
        }
        if (
            metrics["max_abs_error"] > tolerances["max_abs_error"]
            or metrics["mean_abs_error"] > tolerances["mean_abs_error"]
            or metrics["cosine_similarity"] < tolerances["minimum_cosine_similarity"]
            or metrics["top1_agreement"] < tolerances["minimum_top1_agreement"]
        ):
            raise PreparationError(f"ResNet ONNX version parity failed for batch {batch}")
        version_parity["batches"][str(batch)] = metrics
    version_parity["tolerances"] = tolerances
    inspection["version_parity"] = version_parity

    parity_batch = int(resnet["serving"]["tensorrt"]["profile"]["opt"][0])
    parity_input = np.linspace(
        0.0,
        1.0,
        num=int(np.prod([parity_batch, *resnet["input"]["shape"][1:]])),
        dtype=np.float32,
    ).reshape([parity_batch, *resnet["input"]["shape"][1:]])
    parity_output = sessions["resnet50_onnx:1"].run(
        [resnet["output"]["name"]], {resnet["input"]["name"]: parity_input}
    )[0]
    np.save(CACHE_DIRECTORY / "resnet-parity-input.npy", parity_input)
    np.save(CACHE_DIRECTORY / "resnet-parity-onnx.npy", parity_output)
    parity_contract = {
        "engine_path": serving_version_path(resnet["serving"]["tensorrt"], "1"),
        "input_name": resnet["input"]["name"],
        "output_name": resnet["output"]["name"],
        "input_shape": list(parity_input.shape),
        "output_shape": list(parity_output.shape),
        "compute_capability": spec["build"]["target"]["compute_capability"],
        "onnx_source_sha256": sha256_file(
            REPOSITORY_ROOT / serving_version_path(resnet["serving"]["onnx"], "1")
        ),
        "tolerances": resnet["parity"],
    }
    _write_text(CACHE_DIRECTORY / "resnet-parity-contract.json", canonical_json(parity_contract))
    _write_text(INSPECTION_PATH, canonical_json(inspection))
    print("[OK] ONNX checker, fixed-spatial regression, and synthetic inference passed.")


def build_exporter_image(spec: dict[str, Any]) -> None:
    exporter_image = str(spec["build"]["exporter_image"])
    local_image = str(spec["build"]["exporter_local_image"])
    _run(
        [
            "docker",
            "build",
            "--build-arg",
            f"EXPORTER_IMAGE={exporter_image}",
            "--file",
            str(EXPORTER_DOCKERFILE),
            "--tag",
            local_image,
            ".",
        ]
    )


def _run_exporter(spec: dict[str, Any], internal_command: str) -> None:
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--volume",
            _docker_mount(),
            "--workdir",
            "/workspace",
            str(spec["build"]["exporter_local_image"]),
            "python",
            "scripts/model_preparation/prepare_models.py",
            internal_command,
        ]
    )


def export_models(spec: dict[str, Any]) -> None:
    build_exporter_image(spec)
    _run_exporter(spec, "_export")
    _run_exporter(spec, "_inspect-onnx")


def _internal_prepare_tensorrt_onnx(spec: dict[str, Any]) -> None:
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    resnet = spec["models"]["resnet50"]
    source_path = REPOSITORY_ROOT / serving_version_path(
        resnet["serving"]["onnx"], "1"
    )
    target_path = CACHE_DIRECTORY / "resnet50-fp16.onnx"
    graph = onnx.load(str(source_path))
    converted = float16.convert_float_to_float16(
        graph,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.checker.check_model(converted)
    if converted.graph.input[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise PreparationError("Mixed-precision ResNet input must remain FP32")
    if converted.graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise PreparationError("Mixed-precision ResNet output must remain FP32")
    fp16_initializers = sum(
        initializer.data_type == TensorProto.FLOAT16 for initializer in converted.graph.initializer
    )
    if fp16_initializers < 1:
        raise PreparationError("Mixed-precision ResNet contains no FP16 initializers")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(target_path))
    metadata = {
        "path": ".cache/model-preparation/resnet50-fp16.onnx",
        "source_path": serving_version_path(resnet["serving"]["onnx"], "1"),
        "source_sha256": sha256_file(source_path),
        "sha256": sha256_file(target_path),
        "fp16_initializers": fp16_initializers,
        "input_dtype": "FP32",
        "output_dtype": "FP32",
    }
    _write_text(CACHE_DIRECTORY / "tensorrt-onnx-metadata.json", canonical_json(metadata))
    print("[OK] Created strongly typed FP16 ResNet ONNX with FP32 external I/O.")


def _internal_validate_tensorrt() -> None:
    import numpy as np
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    contract_path = CACHE_DIRECTORY / "resnet-parity-contract.json"
    if not contract_path.is_file():
        raise PreparationError("ONNX parity contract is missing; inspect ONNX artifacts first")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_array = np.load(CACHE_DIRECTORY / "resnet-parity-input.npy")
    reference = np.load(CACHE_DIRECTORY / "resnet-parity-onnx.npy")
    engine_path = REPOSITORY_ROOT / contract["engine_path"]

    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise PreparationError("TensorRT engine cannot be deserialized")
    context = engine.create_execution_context()
    if context is None:
        raise PreparationError("TensorRT execution context cannot be created")

    input_name = str(contract["input_name"])
    output_name = str(contract["output_name"])
    tensor_names = [
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    ]
    if tensor_names != [input_name, output_name]:
        raise PreparationError(
            f"TensorRT I/O tensors are {tensor_names}, expected {[input_name, output_name]}"
        )
    if engine.get_tensor_mode(input_name) != trt.TensorIOMode.INPUT:
        raise PreparationError("TensorRT input tensor mode is invalid")
    if engine.get_tensor_mode(output_name) != trt.TensorIOMode.OUTPUT:
        raise PreparationError("TensorRT output tensor mode is invalid")
    if engine.get_tensor_dtype(input_name) != trt.DataType.FLOAT:
        raise PreparationError("TensorRT input I/O precision must remain FP32")
    if engine.get_tensor_dtype(output_name) != trt.DataType.FLOAT:
        raise PreparationError("TensorRT output I/O precision must remain FP32")
    if not context.set_input_shape(input_name, tuple(int(value) for value in input_array.shape)):
        raise PreparationError("TensorRT rejected the parity input shape")
    output_shape = tuple(int(value) for value in context.get_tensor_shape(output_name))
    if list(output_shape) != contract["output_shape"]:
        raise PreparationError(f"TensorRT output shape {output_shape} is incorrect")
    output_array = np.empty(output_shape, dtype=np.float32)

    allocations: list[int] = []
    stream = 0

    def cuda_result(result: tuple[Any, ...], action: str) -> tuple[Any, ...]:
        if result[0] != cudart.cudaError_t.cudaSuccess:
            raise PreparationError(f"CUDA {action} failed with {result[0]}")
        return result[1:]

    try:
        (stream,) = cuda_result(cudart.cudaStreamCreate(), "stream creation")
        (device_input,) = cuda_result(cudart.cudaMalloc(input_array.nbytes), "input allocation")
        allocations.append(device_input)
        (device_output,) = cuda_result(cudart.cudaMalloc(output_array.nbytes), "output allocation")
        allocations.append(device_output)
        cuda_result(
            cudart.cudaMemcpyAsync(
                device_input,
                input_array.ctypes.data,
                input_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "input copy",
        )
        context.set_tensor_address(input_name, int(device_input))
        context.set_tensor_address(output_name, int(device_output))
        if not context.execute_async_v3(stream):
            raise PreparationError("TensorRT asynchronous execution failed")
        cuda_result(
            cudart.cudaMemcpyAsync(
                output_array.ctypes.data,
                device_output,
                output_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "output copy",
        )
        cuda_result(cudart.cudaStreamSynchronize(stream), "stream synchronization")
    finally:
        for allocation in allocations:
            cudart.cudaFree(allocation)
        if stream:
            cudart.cudaStreamDestroy(stream)

    if not np.isfinite(output_array).all():
        raise PreparationError("TensorRT parity output contains non-finite values")
    difference = np.abs(reference - output_array)
    maximum_error = float(difference.max())
    mean_error = float(difference.mean())
    denominator = float(np.linalg.norm(reference.ravel()) * np.linalg.norm(output_array.ravel()))
    cosine_similarity = float(np.dot(reference.ravel(), output_array.ravel()) / denominator)
    top1_agreement = float(
        np.mean(np.argmax(reference, axis=1) == np.argmax(output_array, axis=1))
    )
    tolerances = contract["tolerances"]
    failures = []
    if maximum_error > tolerances["max_abs_error"]:
        failures.append(f"max abs error {maximum_error}")
    if mean_error > tolerances["mean_abs_error"]:
        failures.append(f"mean abs error {mean_error}")
    if cosine_similarity < tolerances["minimum_cosine_similarity"]:
        failures.append(f"cosine similarity {cosine_similarity}")
    if top1_agreement < tolerances["minimum_top1_agreement"]:
        failures.append(f"top-1 agreement {top1_agreement}")
    if failures:
        raise PreparationError("TensorRT parity failed: " + ", ".join(failures))
    validation = {
        "engine_deserialization": "passed",
        "engine_sha256": sha256_file(engine_path),
        "onnx_source_sha256": contract.get("onnx_source_sha256"),
        "input": {"name": input_name, "dtype": "FP32", "shape": list(input_array.shape)},
        "output": {"name": output_name, "dtype": "FP32", "shape": list(output_array.shape)},
        "parity": {
            "max_abs_error": maximum_error,
            "mean_abs_error": mean_error,
            "cosine_similarity": cosine_similarity,
            "top1_agreement": top1_agreement,
            "tolerances": tolerances,
            "status": "passed",
        },
    }
    _write_text(CACHE_DIRECTORY / "tensorrt-validation.json", canonical_json(validation))
    print("[OK] TensorRT engine inspection and exporter-level ResNet parity passed.")


def validate_artifacts(spec: dict[str, Any]) -> None:
    build_exporter_image(spec)
    _run_exporter(spec, "_inspect-onnx")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--volume",
            _docker_mount(),
            "--workdir",
            "/workspace",
            str(spec["build"]["tensorrt_builder_image"]),
            "python",
            "scripts/model_preparation/prepare_models.py",
            "_validate-tensorrt",
        ]
    )
    validation_path = CACHE_DIRECTORY / "tensorrt-validation.json"
    if not validation_path.is_file():
        raise PreparationError("TensorRT validation did not produce its result")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("parity", {}).get("status") != "passed":
        raise PreparationError("TensorRT parity result is not successful")
    print("[OK] ONNX and TensorRT artifact validation completed without Triton.")


def _trtexec_path() -> str:
    return "/usr/src/tensorrt/bin/trtexec"


def _gpu_query(spec: dict[str, Any]) -> dict[str, str]:
    image = str(spec["build"]["tensorrt_builder_image"])
    process = _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            image,
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    first_line = process.stdout.strip().splitlines()[-1]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        raise PreparationError(f"Unexpected nvidia-smi output: {first_line}")
    return {"name": parts[0], "driver_version": parts[1], "compute_capability": parts[2]}


def build_tensorrt(spec: dict[str, Any]) -> None:
    image = str(spec["build"]["tensorrt_builder_image"])
    gpu = _gpu_query(spec)
    expected_cc = str(spec["build"]["target"]["compute_capability"])
    if gpu["compute_capability"] != expected_cc:
        raise PreparationError(
            f"TensorRT target requires compute capability {expected_cc}, found "
            f"{gpu['compute_capability']}"
        )
    build_exporter_image(spec)
    _run_exporter(spec, "_prepare-tensorrt-onnx")
    version_check = _run(
        ["docker", "run", "--rm", "--gpus", "all", image, _trtexec_path(), "--version"],
        capture_output=True,
        check=False,
    )
    version_output = version_check.stdout + version_check.stderr
    if "TensorRT v" not in version_output:
        raise PreparationError("trtexec --version did not report a TensorRT version")
    resnet = spec["models"]["resnet50"]
    profile = resnet["serving"]["tensorrt"]["profile"]
    input_name = str(resnet["input"]["name"])

    def shape(profile_name: str) -> str:
        return "x".join(str(value) for value in profile[profile_name])

    engine_relative_path = serving_version_path(resnet["serving"]["tensorrt"], "1")
    engine_path = REPOSITORY_ROOT / engine_relative_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--volume",
            _docker_mount(),
            "--workdir",
            "/workspace",
            image,
            _trtexec_path(),
            "--onnx=/workspace/.cache/model-preparation/resnet50-fp16.onnx",
            f"--saveEngine=/workspace/{engine_relative_path}",
            f"--minShapes={input_name}:{shape('min')}",
            f"--optShapes={input_name}:{shape('opt')}",
            f"--maxShapes={input_name}:{shape('max')}",
            "--builderOptimizationLevel=3",
            "--skipInference",
        ]
    )
    if not engine_path.is_file():
        relative_path = engine_path.relative_to(REPOSITORY_ROOT).as_posix()
        raise PreparationError(f"TensorRT build did not create {relative_path}")
    print(f"[OK] Built FP16 TensorRT plan for compute capability {expected_cc}.")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        match = PACKAGE_REQUIREMENT.match(line)
        if match:
            versions[match.group(1)] = match.group(2)
    if not versions:
        raise PreparationError("requirements.lock does not contain exact package versions")
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _tensorrt_environment(spec: dict[str, Any]) -> dict[str, str]:
    image = str(spec["build"]["tensorrt_builder_image"])
    process = _run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            image,
            "python",
            "-c",
            (
                "import json, os, tensorrt as trt; "
                "print(json.dumps({'tensorrt_version': trt.__version__, "
                "'cuda_version': os.environ.get('CUDA_VERSION', 'unknown')}))"
            ),
        ],
        capture_output=True,
    )
    json_line = next(
        (line for line in reversed(process.stdout.splitlines()) if line.startswith("{")),
        "",
    )
    if not json_line:
        raise PreparationError("Could not inspect TensorRT container versions")
    value = json.loads(json_line)
    return {str(key): str(item) for key, item in value.items()}


def _manifest_model_entry(
    spec: dict[str, Any],
    logical_key: str,
    serving_kind: str,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    model = spec["models"][logical_key]
    serving = model["serving"][serving_kind]
    versions: dict[str, Any] = {}
    for version, details in serving["versions"].items():
        path = REPOSITORY_ROOT / details["artifact_path"]
        version_entry: dict[str, Any] = {
            "artifact": {
                "path": details["artifact_path"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            },
            "revision": details["revision"],
        }
        for optional in ("derived_from", "transform"):
            if optional in details:
                version_entry[optional] = details[optional]
        inspection_key = f"{serving['name']}:{version}"
        if inspection_key in inspection:
            version_entry["onnx"] = inspection[inspection_key]
        versions[version] = version_entry
    entry: dict[str, Any] = {
        "logical_model_id": logical_key,
        "source": {**model["source"]},
        "preprocessing": model["preprocessing"],
        "labels": model["labels"],
        "input": model["input"],
        "output": model["output"],
        "precision": serving["precision"],
        "max_batch_size": serving["max_batch_size"],
        "smoke_batches": model["smoke_batches"],
        "version_policy": serving["version_policy"],
        "scheduling": serving["scheduling"],
        "model_config": serving_model_config(spec, serving["name"]),
        "versions": versions,
    }
    if logical_key == "resnet50":
        entry["parity_tolerances"] = model["parity"]
        if serving_kind == "onnx":
            entry["version_parity_tolerances"] = model["version_parity"]
    if serving_kind == "tensorrt":
        entry["compute_precision"] = serving["compute_precision"]
        entry["io_precision"] = serving["io_precision"]
        entry["profile"] = serving["profile"]
        entry["compute_capability"] = spec["build"]["target"]["compute_capability"]
    return entry


def create_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    if not INSPECTION_PATH.is_file():
        build_exporter_image(spec)
        _run_exporter(spec, "_inspect-onnx")
    inspection = json.loads(INSPECTION_PATH.read_text(encoding="utf-8"))
    artifact_paths = serving_artifact_paths(spec)
    missing = [path for path in artifact_paths if not (REPOSITORY_ROOT / path).is_file()]
    if missing:
        raise PreparationError("Cannot create manifest; missing artifacts: " + ", ".join(missing))
    validation_path = CACHE_DIRECTORY / "tensorrt-validation.json"
    resnet = spec["models"]["resnet50"]
    engine_path = REPOSITORY_ROOT / serving_version_path(
        resnet["serving"]["tensorrt"], "1"
    )
    onnx_path = REPOSITORY_ROOT / serving_version_path(resnet["serving"]["onnx"], "1")
    validation: dict[str, Any] = {}
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("engine_sha256") != sha256_file(engine_path)
        or validation.get("onnx_source_sha256") != sha256_file(onnx_path)
        or validation.get("parity", {}).get("status") != "passed"
    ):
        validate_artifacts(spec)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    mixed_precision_metadata = json.loads(
        (CACHE_DIRECTORY / "tensorrt-onnx-metadata.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "spec_path": "models/model-spec.yaml",
        "spec_sha256": sha256_file(SPEC_PATH),
        "requirements_lock": str(spec["requirements_lock"]),
        "requirements_sha256": sha256_file(LOCK_PATH),
        "build": {
            "exporter_image": spec["build"]["exporter_image"],
            "tensorrt_builder_image": spec["build"]["tensorrt_builder_image"],
            "onnx_opset": spec["build"]["onnx_opset"],
            "simplify": spec["build"]["simplify"],
            "exporter_packages": _package_versions(),
            "tensorrt_environment": _tensorrt_environment(spec),
            "gpu": _gpu_query(spec),
            "tensorrt_input": mixed_precision_metadata,
        },
        "artifact_validation": {
            "onnx": inspection,
            "tensorrt": validation,
        },
        "models": {
            "resnet50_onnx": _manifest_model_entry(spec, "resnet50", "onnx", inspection),
            "resnet50_tensorrt": _manifest_model_entry(
                spec, "resnet50", "tensorrt", inspection
            ),
            "yolo11n_onnx": _manifest_model_entry(spec, "yolo11n", "onnx", inspection),
        },
    }
    _write_text(MANIFEST_PATH, canonical_json(manifest))
    generate_client_contract()
    create_preparation_evidence()
    print("[OK] Wrote artifact-complete model manifest and preparation evidence.")
    return manifest


def create_preparation_evidence() -> None:
    if not STEP3_MANIFEST_SNAPSHOT_PATH.is_file():
        raise PreparationError("immutable step-3 manifest snapshot is missing")
    evidence = {
        "manifest_path": "docs/evidence/step-3/model-manifest-v1.json",
        "manifest_sha256": sha256_file(STEP3_MANIFEST_SNAPSHOT_PATH),
    }
    _write_text(PREPARATION_EVIDENCE_PATH, canonical_json(evidence))


def manifest_staleness(spec: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 2:
        errors.append("manifest schema version is stale")
    if manifest.get("spec_sha256") != sha256_file(SPEC_PATH):
        errors.append("manifest spec SHA-256 is stale")
    if manifest.get("requirements_sha256") != sha256_file(LOCK_PATH):
        errors.append("manifest requirements SHA-256 is stale")
    if manifest.get("build", {}).get("onnx_opset") != spec["build"]["onnx_opset"]:
        errors.append("manifest ONNX opset is stale")
    expected_images = {
        "exporter_image": spec["build"]["exporter_image"],
        "tensorrt_builder_image": spec["build"]["tensorrt_builder_image"],
    }
    for key, expected in expected_images.items():
        if manifest.get("build", {}).get(key) != expected:
            errors.append(f"manifest {key} is stale")
    model_mapping = {
        "resnet50_onnx": ("resnet50", "onnx"),
        "resnet50_tensorrt": ("resnet50", "tensorrt"),
        "yolo11n_onnx": ("yolo11n", "onnx"),
    }
    entries = manifest.get("models", {})
    if set(entries) != set(model_mapping):
        errors.append("manifest model set is stale")
        return errors
    for serving_name, (logical_key, serving_kind) in model_mapping.items():
        model_spec = spec["models"][logical_key]
        serving = model_spec["serving"][serving_kind]
        entry = entries[serving_name]
        comparisons = {
            "logical_model_id": logical_key,
            "preprocessing": model_spec["preprocessing"],
            "input": model_spec["input"],
            "output": model_spec["output"],
            "precision": serving["precision"],
            "max_batch_size": serving["max_batch_size"],
            "smoke_batches": model_spec["smoke_batches"],
            "version_policy": serving["version_policy"],
            "scheduling": serving["scheduling"],
            "model_config": serving_model_config(spec, serving_name),
        }
        if logical_key == "resnet50":
            comparisons["parity_tolerances"] = model_spec["parity"]
            if serving_kind == "onnx":
                comparisons["version_parity_tolerances"] = model_spec["version_parity"]
        for key, expected in comparisons.items():
            if entry.get(key) != expected:
                errors.append(f"manifest {serving_name}.{key} is stale")
        if entry.get("source", {}).get("sha256") != model_spec["source"]["sha256"]:
            errors.append(f"manifest {serving_name} source SHA-256 is stale")
        if entry.get("source") != model_spec["source"]:
            errors.append(f"manifest {serving_name} source metadata is stale")
        if entry.get("labels") != model_spec["labels"]:
            errors.append(f"manifest {serving_name} label metadata is stale")
        actual_versions = entry.get("versions", {})
        if set(actual_versions) != set(serving["versions"]):
            errors.append(f"manifest {serving_name} version set is stale")
        for version, version_spec in serving["versions"].items():
            actual_version = actual_versions.get(version, {})
            expected_metadata = {
                key: value
                for key, value in version_spec.items()
                if key != "artifact_path"
            }
            for key, expected in expected_metadata.items():
                if actual_version.get(key) != expected:
                    errors.append(
                        f"manifest {serving_name} version {version} {key} is stale"
                    )
            if actual_version.get("artifact", {}).get("path") != version_spec["artifact_path"]:
                errors.append(f"manifest {serving_name} version {version} path is stale")
        if serving_kind == "tensorrt":
            if entry.get("profile") != serving["profile"]:
                errors.append("manifest TensorRT profile is stale")
            expected_capability = spec["build"]["target"]["compute_capability"]
            if entry.get("compute_capability") != expected_capability:
                errors.append("manifest TensorRT compute capability is stale")
    validation = manifest.get("artifact_validation", {})
    tensorrt_validation = validation.get("tensorrt", {})
    if tensorrt_validation.get("parity", {}).get("status") != "passed":
        errors.append("manifest TensorRT artifact parity is not passed")
    expected_tolerances = spec["models"]["resnet50"]["parity"]
    if tensorrt_validation.get("parity", {}).get("tolerances") != expected_tolerances:
        errors.append("manifest TensorRT artifact parity tolerances are stale")
    if tensorrt_validation.get("engine_sha256") != entries.get("resnet50_tensorrt", {}).get(
        "versions", {}
    ).get("1", {}).get("artifact", {}).get("sha256"):
        errors.append("manifest TensorRT validation engine SHA-256 is stale")
    if tensorrt_validation.get("onnx_source_sha256") != entries.get("resnet50_onnx", {}).get(
        "versions", {}
    ).get("1", {}).get("artifact", {}).get("sha256"):
        errors.append("manifest TensorRT validation ONNX SHA-256 is stale")
    for onnx_name in ("resnet50_onnx", "yolo11n_onnx"):
        for version, version_entry in entries.get(onnx_name, {}).get("versions", {}).items():
            expected_onnx = version_entry.get("onnx")
            key = f"{onnx_name}:{version}"
            if validation.get("onnx", {}).get(key) != expected_onnx:
                errors.append(f"manifest {key} validation metadata is stale")
    expected_version_parity = entries.get("resnet50_onnx", {}).get(
        "version_parity_tolerances"
    )
    version_parity = validation.get("onnx", {}).get("version_parity", {})
    if version_parity.get("status") != "passed":
        errors.append("manifest ResNet version parity is not passed")
    if version_parity.get("tolerances") != expected_version_parity:
        errors.append("manifest ResNet version parity tolerances are stale")
    tensorrt_input = manifest.get("build", {}).get("tensorrt_input", {})
    resnet_artifact = entries.get("resnet50_onnx", {}).get("versions", {}).get(
        "1", {}
    ).get("artifact", {})
    if tensorrt_input.get("source_sha256") != resnet_artifact.get("sha256"):
        errors.append("manifest strongly typed TensorRT input is stale")
    return errors


def check_generated(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for serving_name in SERVING_MODELS:
        path = REPOSITORY_ROOT / "models" / serving_name / "config.pbtxt"
        expected = render_config(spec, serving_name)
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            errors.append(f"stale generated config: {relative_path}")
    label_targets = {
        "resnet50": ("resnet50_onnx", "resnet50_tensorrt"),
        "yolo11n": ("yolo11n_onnx",),
    }
    for model_key, serving_names in label_targets.items():
        labels = spec["models"][model_key]["labels"]
        for serving_name in serving_names:
            path = REPOSITORY_ROOT / "models" / serving_name / labels["filename"]
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            if not path.is_file():
                errors.append(f"missing generated labels: {relative_path}")
                continue
            if sha256_file(path) != labels["generated_sha256"]:
                errors.append(f"stale generated labels: {relative_path}")
            if len(path.read_text(encoding="utf-8").splitlines()) != labels["count"]:
                errors.append(f"wrong generated label count: {relative_path}")
    if not MANIFEST_PATH.is_file():
        errors.append("missing generated manifest: models/model-manifest.json")
    else:
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            errors.extend(manifest_staleness(spec, manifest))
        except (json.JSONDecodeError, OSError, TypeError) as error:
            errors.append(f"cannot read model manifest: {error}")
    if not PREPARATION_EVIDENCE_PATH.is_file():
        errors.append("missing generated preparation evidence")
    elif STEP3_MANIFEST_SNAPSHOT_PATH.is_file():
        try:
            evidence = json.loads(PREPARATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
            expected = {
                "manifest_path": "docs/evidence/step-3/model-manifest-v1.json",
                "manifest_sha256": sha256_file(STEP3_MANIFEST_SNAPSHOT_PATH),
            }
            if evidence != expected:
                errors.append("preparation evidence is stale")
        except (json.JSONDecodeError, OSError, TypeError) as error:
            errors.append(f"cannot read preparation evidence: {error}")
    if not CLIENT_CONTRACT_PATH.is_file():
        errors.append("missing generated shared/client-model-contracts.json")
    elif MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            expected_contract = canonical_json(render_client_contract(manifest))
            if CLIENT_CONTRACT_PATH.read_text(encoding="utf-8") != expected_contract:
                errors.append("stale generated client model contract")
        except (json.JSONDecodeError, OSError, TypeError) as error:
            errors.append(f"cannot validate client model contract: {error}")
    return errors


def clean_models(spec: dict[str, Any]) -> None:
    for relative_path in serving_artifact_paths(spec):
        path = (REPOSITORY_ROOT / relative_path).resolve()
        if REPOSITORY_ROOT.resolve() not in path.parents:
            raise PreparationError(f"Refusing to clean path outside repository: {path}")
        path.unlink(missing_ok=True)
    cache = CACHE_DIRECTORY.resolve()
    if cache.parent != (REPOSITORY_ROOT / ".cache").resolve():
        raise PreparationError(f"Refusing to clean unexpected cache path: {cache}")
    if cache.is_dir():
        shutil.rmtree(cache)
    print("[OK] Removed only ignored model binaries, source weights, and preparation cache.")


def _validate_repository(structure_only: bool) -> None:
    arguments = [sys.executable, "scripts/validate_model_repository.py"]
    if structure_only:
        arguments.append("--structure-only")
    _run(arguments)


def prepare(spec: dict[str, Any]) -> None:
    _run(["docker", "version"], capture_output=True)
    download_sources(spec)
    generate_repository_text(spec)
    export_models(spec)
    build_tensorrt(spec)
    create_manifest(spec)
    _validate_repository(structure_only=False)
    print("[OK] Model preparation reached artifact-complete state.")


def prepare_versions(spec: dict[str, Any]) -> None:
    """Rebuild ONNX serving versions and refresh the manifest when artifacts exist."""
    download_sources(spec)
    generate_repository_text(spec)
    export_models(spec)
    missing = [
        path for path in serving_artifact_paths(spec) if not (REPOSITORY_ROOT / path).is_file()
    ]
    if missing:
        raise PreparationError(
            "Serving versions were prepared, but manifest refresh needs: " + ", ".join(missing)
        )
    create_manifest(spec)


def _validate_images(spec: dict[str, Any]) -> None:
    for key in ("exporter_image", "tensorrt_builder_image"):
        reference = str(spec["build"].get(key, ""))
        match = IMAGE_REFERENCE.fullmatch(reference)
        if not match or match.group("tag").lower() == "latest":
            raise PreparationError(f"{key} must use an exact tag@sha256 reference")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check tracked generated files")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "prepare",
            "prepare-versions",
            "discover",
            "download",
            "generate",
            "export",
            "build-tensorrt",
            "validate",
            "clean",
            "manifest",
            "client-contract",
            "_export",
            "_inspect-onnx",
            "_prepare-tensorrt-onnx",
            "_validate-tensorrt",
        ),
    )
    arguments = parser.parse_args()
    try:
        if arguments.command == "_validate-tensorrt":
            _internal_validate_tensorrt()
            return 0
        spec = load_spec()
        _validate_images(spec)
        if arguments.check:
            errors = check_generated(spec)
            if errors:
                for error in errors:
                    print(f"[ERROR] {error}", file=sys.stderr)
                return 1
            print(
                "[OK] Generated model configs, labels, manifest, and preparation "
                "evidence are current."
            )
            return 0
        if not arguments.command:
            parser.error("a command is required unless --check is used")
        commands = {
            "prepare": lambda: prepare(spec),
            "prepare-versions": lambda: prepare_versions(spec),
            "discover": lambda: discover_sources(spec),
            "download": lambda: download_sources(spec),
            "generate": lambda: (download_sources(spec), generate_repository_text(spec)),
            "export": lambda: export_models(spec),
            "build-tensorrt": lambda: build_tensorrt(spec),
            "validate": lambda: _validate_repository(structure_only=False),
            "clean": lambda: clean_models(spec),
            "manifest": lambda: create_manifest(spec),
            "client-contract": generate_client_contract,
            "_export": lambda: _internal_export(spec),
            "_inspect-onnx": lambda: _internal_inspect_onnx(spec),
            "_prepare-tensorrt-onnx": lambda: _internal_prepare_tensorrt_onnx(spec),
        }
        commands[arguments.command]()
        return 0
    except (OSError, ValueError, TypeError, KeyError, PreparationError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
