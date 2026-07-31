# Model preparation

## Runtime-verified scope

Step 3 produces and verifies three serving artifacts:

| Triton model | Artifact | Contract source |
|---|---|---|
| `resnet50_onnx` | ONNX FP32 | ResNet tensor and preprocessing sections in the model spec |
| `resnet50_tensorrt` | TensorRT FP16 compute, FP32 I/O | Same ResNet contract plus the spec profile and target capability |
| `yolo11n_onnx` | ONNX FP32, no NMS | YOLO tensor and preprocessing sections in the model spec |

Only the batch dimension is dynamic. Dynamic request batching, model version switching, user-image preprocessing, detection decoding, NMS, and performance benchmarking remain outside this step.

## Sources of truth

`models/model-spec.yaml` is the only editable source for model identities, source URLs and hashes, licenses, tensor contracts, preprocessing, opset, build images, artifact paths, TensorRT profile, and parity tolerances.

`scripts/model_preparation/requirements.lock` is the only source for exact exporter package versions and Linux CPython 3.11 wheel hashes. `config.pbtxt`, model-local labels, `models/model-manifest.json`, and preparation evidence are generated and checked for staleness.

The accepted full source hashes live only in the model spec and are copied into the generated artifact manifest for auditability.

`discover` downloads candidates and prints hashes but never edits the spec:

```text
python scripts/model_preparation/prepare_models.py discover
```

Artifact preparation refuses an unresolved or mismatched source hash.

## Reproducible build

The exporter runs in the pinned Ultralytics CPU image declared as `tag@sha256` in the spec. It exports ResNet50 and YOLO11n with the spec opset, runs ONNX Checker, checks graph metadata, and performs ONNX Runtime synthetic inference. The YOLO adapter copies the fixed output dimensions from the spec into ONNX metadata; only batch remains symbolic.

The selected TensorRT toolchain uses strongly typed networks and does not accept the historical `trtexec --fp16` flag. The workflow therefore creates an ignored intermediate ResNet ONNX with FP16 internal weights and tensors plus FP32 boundary casts, then builds the engine in the pinned container. Its `model_cc<capability>.plan` name and `cc_model_filenames` mapping are derived from the spec. The TensorRT config deliberately omits `default_model_filename`, and no generic `model.plan` is produced, so another capability cannot fall back to this engine.

Run the complete artifact workflow:

```text
make prepare-models
```

Direct equivalent:

```text
python scripts/model_preparation/prepare_models.py prepare
```

Preparation ends at artifact-complete. It does not call Triton.

## Validation levels

Tracked structure, spec schemas, generated text, manifest references, Git tracking policy, and evidence staleness can be checked without Docker:

```text
make validate-model-structure
python scripts/validate_model_repository.py --structure-only
```

Full artifact validation requires Docker and the target GPU, but not Triton. It runs ONNX Checker and ONNX Runtime again, deserializes the TensorRT plan, verifies FP32 external I/O, and compares TensorRT output against the exporter-level ONNX reference:

```text
make validate-models
python scripts/validate_model_repository.py
```

Runtime verification requires the running step 2 Triton service:

```text
make smoke-models
python deployment/triton/smoke_models.py --env-file .env.example
```

The smoke performs explicit load, readiness, metadata/config checks, ResNet inference for batches 1, 4, and 8, YOLO inference for batches 1 and 2, ResNet ONNX/TensorRT parity, and explicit unload. It writes sanitized evidence under `docs/evidence/step-3`.

## Local artifacts and disk use

Source weights and intermediates live under `.cache/model-preparation`. Serving binaries are:

- `models/resnet50_onnx/1/model.onnx`;
- the capability-qualified TensorRT plan declared in the model spec;
- `models/yolo11n_onnx/1/model.onnx`.

They are ignored by Git. A clean checkout retains the spec, configs, labels, manifest, and evidence, while the three binaries must be reproduced locally for full validation or serving. Allow roughly 1 GB for model sources and generated artifacts, plus local Docker storage for the exporter, TensorRT, and Triton images. The container images dominate disk use and can require several tens of gigabytes.

Remove only ignored model binaries, source weights, and preparation cache:

```text
make clean-models
python scripts/model_preparation/prepare_models.py clean
```

The command does not remove committed configs, labels, manifest, or evidence.

## Licensing and portability

The repository is AGPL-3.0-only because the selected Ultralytics YOLO11 open-source workflow is AGPL-3.0. Closed commercial use requires replacing the model/tooling or obtaining an appropriate commercial license.

TorchVision code is BSD-3-Clause, but the ResNet pretrained weights are ImageNet-derived and require independent review of upstream dataset and weight terms. See `THIRD_PARTY_NOTICES.md`.

The generated manifest records the exact GPU, compute capability, TensorRT version, and driver/runtime used for the plan. It is not represented as portable to another compute capability or TensorRT runtime. Rebuild it on the target environment.
