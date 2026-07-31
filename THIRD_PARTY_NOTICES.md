# Third-party notices

This educational repository is licensed under AGPL-3.0-only. That repository license does not replace the licenses or terms of third-party software, pretrained weights, datasets, container images, or GPU runtimes used by the reproducible workflow.

## Models and model tooling

| Component | Reproducibility record | License or terms |
|---|---|---|
| Ultralytics exporter and YOLO weights | Package version and wheel hash in the lock; model identity and source hash in the spec | AGPL-3.0 under the open-source option. A separate Ultralytics Enterprise license is required for uses that cannot comply with AGPL. |
| PyTorch | Package version and wheel hash in the lock | BSD-3-Clause for the software. |
| TorchVision | Package version and wheel hash in the lock | BSD-3-Clause for the software. |
| ResNet50 pretrained weights | Model identity and source hash in the spec | ImageNet-derived pretrained weights. Dataset and weight terms require independent review; this repository does not claim that the weights are BSD-3-Clause. |
| ONNX | Package version and wheel hash in the lock | Apache-2.0. |
| ONNX Runtime | Package version and wheel hash in the lock | MIT. |
| ONNX Converter Common | Package version and wheel hash in the lock | MIT. |
| NumPy | Package version and wheel hash in the lock | BSD-3-Clause. |
| PyYAML | Package version and wheel hash in the lock | MIT. |

Exact package versions and wheel hashes are recorded only in `scripts/model_preparation/requirements.lock`. Model source URLs, source hashes, label sources, and license metadata are recorded only in `models/model-spec.yaml`; the generated manifest captures the values used for the verified artifacts.

## Serving and GPU runtime

| Component | Selected image | License or terms |
|---|---|---|
| NVIDIA Triton Inference Server | `nvcr.io/nvidia/tritonserver:26.07-py3` | Triton source is BSD-3-Clause; the NVIDIA container and bundled components are also subject to the NVIDIA Deep Learning Container License and their component terms. |
| NVIDIA TensorRT | `nvcr.io/nvidia/tensorrt:26.07-py3` | NVIDIA Software License Agreement and NVIDIA AI Product-Specific Terms. |
| NVIDIA DCGM Exporter | Pin in `.env.example` | NVIDIA and bundled component terms. |
| Prometheus | Pin in `.env.example` | Apache-2.0. |
| Grafana | Pin in `.env.example` | AGPL-3.0. |

Primary references:

- Ultralytics licensing: https://www.ultralytics.com/license
- PyTorch and TorchVision model terms: https://docs.pytorch.org/vision/main/models.html
- Triton server repository and license: https://github.com/triton-inference-server/server
- NVIDIA software agreements: https://www.nvidia.com/en-us/agreements/enterprise-software/
- ImageNet terms: https://www.image-net.org/download.php

This notice is an inventory for reproducibility and is not legal advice.
