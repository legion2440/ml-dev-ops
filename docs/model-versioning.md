# Model versioning

## Current status

No model artifacts or Triton version policies exist yet.

## Planned contract

At least one model will expose two Triton versions with identical input and output
contracts. The initial plan uses a baseline ResNet50 ONNX version and an updated or
graph-optimized ONNX version.

Documentation and smoke tests will cover repository indexing, explicit model
control, load, unload, reload, a request to a specific version, and default version
selection through version policy.
