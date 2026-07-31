# Model versioning

## Current state

Step 3 implements only model version `1` for `resnet50_onnx`, `resnet50_tensorrt`, and `yolo11n_onnx`. Triton runs with explicit model control, and `deployment/triton/smoke_models.py` verifies load, readiness, inference, repository state, and unload.

Tracked `config.pbtxt` files intentionally contain no `version_policy`. Triton reports its default latest-version policy at runtime, but there is only one numeric version. Directories `0`, zero-padded names such as `01`, and versions other than `1` are rejected by the step 3 structure validator.

TensorRT uses the strict filename `model_cc89.plan` and a `cc_model_filenames` mapping for compute capability 8.9. No generic `model.plan` fallback is produced.

## Deferred step 4 scope

Step 4 will introduce an additional version with the same public tensor contract, explicit version-policy behavior, requests to a selected version, default selection, reload, and rollback verification. Dynamic batching also remains deferred even though the current models accept batched tensors.
