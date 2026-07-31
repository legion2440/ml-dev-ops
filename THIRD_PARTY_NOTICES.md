# Third-party notices

This educational repository is licensed under AGPL-3.0-only. That repository license does not replace the licenses or terms of third-party software, pretrained weights, datasets, container images, or GPU runtimes used by the reproducible workflow.

## Models and model tooling

| Component | Selected version or artifact | License or terms |
|---|---|---|
| Ultralytics | `ultralytics==8.3.0`, `yolo11n.pt` | AGPL-3.0 under the open-source option. A separate Ultralytics Enterprise license is required for uses that cannot comply with AGPL. |
| PyTorch | `torch==2.4.1+cpu` | BSD-3-Clause for the software. |
| TorchVision | `torchvision==0.19.1+cpu` | BSD-3-Clause for the software. |
| ResNet50 IMAGENET1K_V2 weights | `resnet50-11ad3fa6.pth` | ImageNet-derived pretrained weights. Dataset and weight terms require independent review; this repository does not claim that the weights are BSD-3-Clause. |
| ONNX | `onnx==1.16.2` | Apache-2.0. |
| ONNX Runtime | `onnxruntime==1.19.2` | MIT. |
| ONNX Converter Common | `onnxconverter-common==1.16.0` | MIT. |
| NumPy | `numpy==1.26.4` | BSD-3-Clause. |
| PyYAML | `PyYAML==6.0.2` | MIT. |

Exact wheel hashes are recorded in `scripts/model_preparation/requirements.lock`. Model source URLs, source hashes, label sources, and license metadata are recorded in `models/model-spec.yaml`.

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
