"""Build and render a complete Triton ModelConfig from plain mappings."""

from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


def _version_numbers(serving: Mapping[str, Any]) -> list[int]:
    return sorted(int(version) for version in serving["version_policy"]["specific"])


def build_model_config(
    model: Mapping[str, Any],
    serving: Mapping[str, Any],
    *,
    platform: str,
    compute_capability: str | None = None,
) -> dict[str, Any]:
    """Return one complete neutral ModelConfig used by every renderer."""
    versions = serving["versions"]
    filenames = {
        PurePosixPath(details["artifact_path"]).name for details in versions.values()
    }
    if len(filenames) != 1:
        raise ValueError("all versions of a serving model must use one artifact filename")
    filename = next(iter(filenames))
    output = {
        "name": model["output"]["name"],
        "data_type": f"TYPE_{model['output']['dtype']}",
        "dims": list(model["output"]["shape"][1:]),
    }
    label_filename = model.get("labels", {}).get("filename")
    if label_filename and model.get("labels", {}).get("count") == model["output"]["shape"][-1]:
        output["label_filename"] = label_filename

    config: dict[str, Any] = {
        "name": serving["name"],
        "platform": platform,
        "max_batch_size": serving["max_batch_size"],
        "version_policy": {"specific": {"versions": _version_numbers(serving)}},
        "dynamic_batching": {
            "preferred_batch_size": list(
                serving["scheduling"]["dynamic_batching"]["preferred_batch_sizes"]
            ),
            "max_queue_delay_microseconds": serving["scheduling"]["dynamic_batching"][
                "max_queue_delay_microseconds"
            ],
        },
        "input": [
            {
                "name": model["input"]["name"],
                "data_type": f"TYPE_{model['input']['dtype']}",
                "dims": list(model["input"]["shape"][1:]),
            }
        ],
        "output": [output],
    }
    if platform == "tensorrt_plan":
        if not compute_capability:
            raise ValueError("TensorRT config requires compute_capability")
        config["cc_model_filenames"] = {compute_capability: filename}
    else:
        config["default_model_filename"] = filename
    return config


def validate_contract_relationships(config: Mapping[str, Any]) -> list[str]:
    """Validate relationships inside a complete ModelConfig without I/O."""
    errors: list[str] = []
    max_batch_size = config.get("max_batch_size")
    batching = config.get("dynamic_batching", {})
    preferred = batching.get("preferred_batch_size", [])
    if not isinstance(max_batch_size, int) or max_batch_size < 1:
        errors.append("max_batch_size must be positive")
    if (
        not isinstance(preferred, list)
        or not preferred
        or any(not isinstance(value, int) or value < 1 for value in preferred)
    ):
        errors.append("preferred batch sizes must be positive integers")
    elif isinstance(max_batch_size, int) and any(
        value > max_batch_size for value in preferred
    ):
        errors.append("preferred batch size exceeds max_batch_size")
    delay = batching.get("max_queue_delay_microseconds")
    if not isinstance(delay, int) or delay < 0:
        errors.append("max_queue_delay_microseconds must be nonnegative")
    versions = config.get("version_policy", {}).get("specific", {}).get("versions", [])
    if (
        not isinstance(versions, list)
        or not versions
        or versions != sorted(set(versions))
        or any(not isinstance(version, int) or version < 1 for version in versions)
    ):
        errors.append("specific version policy must contain sorted positive versions")
    if config.get("platform") == "tensorrt_plan":
        if "default_model_filename" in config:
            errors.append("TensorRT config must not define default_model_filename")
        mappings = config.get("cc_model_filenames")
        if not isinstance(mappings, dict) or len(mappings) != 1:
            errors.append("TensorRT config must define one capability-qualified plan")
    return errors


def _tensor_pbtxt(kind: str, tensors: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"{kind} ["]
    for tensor in tensors:
        lines.extend(
            [
                "  {",
                f'    name: "{tensor["name"]}"',
                f"    data_type: {tensor['data_type']}",
                "    dims: [ " + ", ".join(str(value) for value in tensor["dims"]) + " ]",
            ]
        )
        if tensor.get("label_filename"):
            lines.append(f'    label_filename: "{tensor["label_filename"]}"')
        lines.append("  }")
    lines.append("]")
    return "\n".join(lines) + "\n"


def render_pbtxt(config: Mapping[str, Any]) -> str:
    """Render the canonical tracked protobuf-text ModelConfig."""
    errors = validate_contract_relationships(config)
    if errors:
        raise ValueError("; ".join(errors))
    lines = [
        f'name: "{config["name"]}"',
        f'platform: "{config["platform"]}"',
        f"max_batch_size: {config['max_batch_size']}",
    ]
    if "default_model_filename" in config:
        lines.append(f'default_model_filename: "{config["default_model_filename"]}"')
    for capability, filename in sorted(config.get("cc_model_filenames", {}).items()):
        lines.extend(
            [
                "cc_model_filenames {",
                f'  key: "{capability}"',
                f'  value: "{filename}"',
                "}",
            ]
        )
    lines.extend(["version_policy {", "  specific {"])
    for version in config["version_policy"]["specific"]["versions"]:
        lines.append(f"    versions: {version}")
    lines.extend(["  }", "}", "dynamic_batching {"])
    for batch_size in config["dynamic_batching"]["preferred_batch_size"]:
        lines.append(f"  preferred_batch_size: {batch_size}")
    lines.extend(
        [
            "  max_queue_delay_microseconds: "
            f"{config['dynamic_batching']['max_queue_delay_microseconds']}",
            "}",
        ]
    )
    return (
        "\n".join(lines)
        + "\n"
        + _tensor_pbtxt("input", config["input"])
        + _tensor_pbtxt("output", config["output"])
    )


def render_load_config_json(
    config: Mapping[str, Any], versions: Sequence[int]
) -> dict[str, dict[str, str]]:
    """Render the HTTP repository-load wrapper with a full ModelConfig JSON string."""
    override = copy.deepcopy(dict(config))
    override["version_policy"] = {
        "specific": {"versions": sorted(int(version) for version in versions)}
    }
    errors = validate_contract_relationships(override)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "parameters": {
            "config": json.dumps(
                override, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        }
    }
