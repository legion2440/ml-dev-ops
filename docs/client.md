# Client and inference logging

## Current status

Step 5 is runtime-verified. `client/inference_client.py` sends real images to Triton
over HTTP or gRPC, performs reusable preprocessing and postprocessing, and writes a
schema-validated JSONL request history with a deterministic CSV projection.

## Client contract

The client accepts one JPG, JPEG, or PNG file or a directory. Directory traversal is
deterministic. Batch limits, tensor names and shapes, preprocessing, supported
versions, tasks, and labels come only from the generated
`shared/client-model-contracts.json` contract. Client code never reads the model
specification or model manifest directly.

```text
python client/inference_client.py health
python client/inference_client.py classify client/samples/01_dog.jpg
python client/inference_client.py classify client/samples/ --model resnet50_tensorrt --protocol grpc --batch-size 4
python client/inference_client.py detect client/samples/ --protocol http --batch-size 2
python client/inference_client.py metadata --model resnet50_onnx --version 2
python client/inference_client.py export-logs --input-log logs/inference.jsonl --output-csv logs/inference.csv
```

If a requested model/version is not READY, the client uses the HTTP repository API
to load its tracked serving configuration and waits for readiness. `--no-auto-load`
turns this behavior into a client-side error. A production request does not unload
the model afterward.

ResNet uses the generated RGB, resize-shortest-side, centered-crop, scale, and
ImageNet normalization contract with the exact pinned torchvision resize and crop
geometry. Its output semantics must identify logits before stable softmax and top-K
label lookup run. YOLO uses centered letterbox resizing and contract-driven output
semantics: `xywh` boxes, class scores beginning at index 4, no objectness field, and
class-aware NMS. Postprocessing maps boxes back to the source image, clips them, and
bounds the detection count.

## Logging contract

Each inference event includes a timestamp, request ID, requested and resolved model
version, protocol, sanitized input metadata, per-stage timing, status, final
prediction summaries, and sanitized error text. One line represents one attempted
Triton request, including failures after request-ID creation. Raw pixels, logits,
YOLO candidate tensors, secrets, and absolute host paths are never logged.

JSONL is the primary history format. CSV is a derived export for offline analysis.
Operational files default to `logs/inference.jsonl` and `logs/inference.csv` and are
ignored by Git. Tracked proof under `docs/evidence/step-5` contains 11 requests: ten
images classified in batches of 4, ten detected in batches of 2, plus explicit gRPC,
ONNX v2, and TensorRT cases. Validate it with
`python scripts/validate_client_evidence.py`.
