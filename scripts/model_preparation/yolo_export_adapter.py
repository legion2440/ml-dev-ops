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
    example = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
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

    # Ultralytics shape arithmetic leaves the two non-batch output dimensions
    # symbolic even with fixed spatial input. They are contractually fixed for
    # 640x640 detection without NMS, so make that fact explicit in ValueInfo.
    import onnx

    graph = onnx.load(str(output_path))
    output_dimensions = graph.graph.output[0].type.tensor_type.shape.dim
    for dimension, value in zip(output_dimensions[1:], (84, 8400), strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output_path))
