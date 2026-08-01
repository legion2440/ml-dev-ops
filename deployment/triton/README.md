# Triton serving

Triton startup policy is owned by `docker-compose.yml`. The server uses explicit
model control, disables backend config auto-completion, exposes HTTP, gRPC, and
metrics, and mounts `models` read-only. Complete generated ModelConfig data comes
from `models/model-spec.yaml` through `shared/triton_model_config.py` for both tracked
protobuf text and HTTP load overrides.

Run the step 4 verifier against an existing Compose server:

```text
make verify-serving
```

The profile-only `triton-verifier` service uses the pinned official SDK image, needs
no GPU, publishes no ports, and cannot write outside `docs/evidence/step-4`. It checks
the required server extensions, HTTP/gRPC metadata and binary inference, exact
cross-protocol results, dynamic-batching statistics, ResNet version switching, a
reload without unload, and final cleanup.

`deployment/triton/smoke_models.py` and immutable `docs/evidence/step-3` remain the
historical single-version proof. This verifier is not the production image client;
preprocessing, postprocessing, user-facing requests, and inference logging belong to
step 5.
