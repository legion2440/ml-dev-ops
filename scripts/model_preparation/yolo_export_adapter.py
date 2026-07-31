"""Export YOLO11 with only the batch dimension marked dynamic."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_yolo11n(
    weights_path: Path,
    output_path: Path,
    *,
    opset: int,
    input_name: str,
    output_name: str,
    input_shape: list[int],
    output_shape: list[int],
) -> None:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights_path)).model.eval().float()

    class DetectionTensor(torch.nn.Module):
        def __init__(self, detector: torch.nn.Module) -> None:
            super().__init__()
            self.detector = detector

        def forward(self, images: Any) -> Any:
            result = self.detector(images)
            return result[0] if isinstance(result, (tuple, list)) else result

    wrapper = DetectionTensor(model).eval()
    example = torch.zeros((1, *input_shape[1:]), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        )

    # Ultralytics shape arithmetic leaves non-batch output dimensions symbolic
    # even with fixed spatial input, so copy the canonical spec dimensions into
    # ValueInfo while retaining only the batch dimension as dynamic.
    import onnx

    graph = onnx.load(str(output_path))
    output_dimensions = graph.graph.output[0].type.tensor_type.shape.dim
    for dimension, value in zip(output_dimensions[1:], output_shape[1:], strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output_path))
