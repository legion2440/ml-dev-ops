# Model preparation

## Current status

Model download, export, optimization, and validation are planned.

## Contract

Preparation will consume pinned upstream weights and configuration, produce
validated ONNX models, build TensorRT plans on the compatible target GPU, inspect
input and output metadata, and populate the Triton model repository.

The workflow must be reproducible and must record enough environment information to
explain an artifact. TensorRT plans are not treated as universally portable model
files.

The specific YOLO release, weight storage mechanism, runtime versions, and target
GPU parameters remain deferred.
