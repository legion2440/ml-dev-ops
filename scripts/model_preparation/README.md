# Model preparation

`prepare_models.py` is the public artifact-preparation entrypoint. It downloads hash-accepted sources, generates configs and labels from `models/model-spec.yaml`, exports two ONNX graphs in a pinned container, builds one `model.plan` on the GPU selected by `--gpu-device`, records host/toolchain provenance, creates the generated manifest, and runs Triton-free artifact validation on the same device.

```text
python scripts/model_preparation/prepare_models.py prepare
```

`--gpu-device` accepts one host GPU index or UUID and defaults to `0`.

Additional commands are `discover`, `download`, `generate`, `export`, `build-tensorrt`, `validate`, `manifest`, `client-contract`, and `clean`. `--check` is daemon-free and verifies tracked generated files, including the client-facing projection in `shared/client-model-contracts.json`.

The exporter image is built from `Dockerfile.exporter`; package versions and wheel hashes come only from `requirements.lock`. Runtime model smoke belongs to `deployment/triton/smoke_models.py`, not this module.
