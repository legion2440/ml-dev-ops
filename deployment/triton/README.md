# Triton serving

Triton startup policy is owned by `docker-compose.yml` and exposed through `deployment/scripts/run_triton.sh`. The server uses explicit model control, disables backend config auto-completion, publishes HTTP, gRPC, and metrics endpoints, and mounts `models` read-only.

Disabling auto-completion is required for the step 3 contract: the TensorRT backend otherwise adds a dynamic scheduler when `max_batch_size` is greater than one. All model configs are therefore complete and generated from `models/model-spec.yaml`.

Run the model runtime verification after artifact preparation and Triton startup:

```text
python deployment/triton/smoke_models.py --env-file .env.example
```

The smoke explicitly loads all three version-1 models, checks metadata and config, performs synthetic inference, compares ResNet ONNX and TensorRT, records evidence, unloads the models, and confirms that none remains ready.
