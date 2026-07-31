# Model versioning

## Current status

No model artifacts or model version policies exist yet. The step 2 Triton runtime
already uses explicit model control so later load, unload, and reload operations do
not require an infrastructure-mode change.

## Planned contract

At least one model will expose two Triton versions with identical input and output
contracts. The initial plan uses a baseline ResNet50 ONNX version and an updated or
graph-optimized ONNX version.

Documentation and smoke tests will cover repository indexing, explicit model
control, load, unload, reload, a request to a specific version, and default version
selection through version policy.
