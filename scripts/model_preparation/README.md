# Model preparation

`prepare_models.py` is the public artifact-preparation entrypoint. It downloads hash-accepted sources, generates configs and labels from `models/model-spec.yaml`, exports two ONNX graphs in a pinned container, builds the CC 8.9 TensorRT plan, creates the generated manifest, and runs Triton-free artifact validation.

```text
python scripts/model_preparation/prepare_models.py prepare
```

Additional commands are `discover`, `download`, `generate`, `export`, `build-tensorrt`, `validate`, `manifest`, and `clean`. `--check` is daemon-free and verifies tracked generated files.

The exporter image is built from `Dockerfile.exporter`; package versions and wheel hashes come only from `requirements.lock`. Runtime model smoke belongs to `deployment/triton/smoke_models.py`, not this module.
