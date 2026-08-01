# Inference client

`inference_client.py` is the production CLI for real-image classification and
detection through Triton. It supports HTTP and gRPC binary tensors, explicit model
versions, bounded input batches, model auto-load, metadata, health checks, JSON
output, persistent JSONL history, and CSV export.

```text
python client/inference_client.py
python client/inference_client.py classify client/samples/01_dog.jpg
python client/inference_client.py detect client/samples/ --protocol grpc --batch-size 2
```

The client reads `client-config.yaml` and the generated
`shared/client-model-contracts.json`. It intentionally has no dependency on the
model repository. `verify_runtime.py` runs the complete live protocol/model matrix
and restores the initial READY state.
