# Model versioning

## Implemented policy

`models/model-spec.yaml` is the editable source for every version and policy.
`resnet50_onnx` exposes versions `1` and `2`; the TensorRT and YOLO variants expose
version `1`. Generated `config.pbtxt` files use an explicit `specific` policy, so
Triton never relies on its implicit latest-version default.

ResNet version 2 is a serving graph revision, not new weights or a new training
checkpoint. Preparation deterministically renames the terminal tensor and appends an
Identity node while preserving public names, shapes, dtypes, preprocessing, and
weights. ONNX Checker, ONNX Runtime batches from the spec, distinct artifact hashes,
and strict v1/v2 output parity are required before the manifest is generated.

The runtime verifier exercises policy overrides through the HTTP repository API:

- only version 1, with default inference selecting `1`;
- only version 2, with default inference selecting `2`;
- tracked policy `1+2`, with explicit requests to both and default selection of `2`;
- a genuine load/reload while the model is already loaded, without a prior unload.

Model control is HTTP-only in the verifier. HTTP and gRPC are both used for metadata
and binary inference. Cleanup unloads all models and confirms Triton remains live and
ready with an empty READY set.

TensorRT remains strictly capability-qualified through `cc_model_filenames`. Its
config has no `default_model_filename`, and `model.plan` is absent, so another GPU
capability cannot fall back to the CC 8.9 engine.
