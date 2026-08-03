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
no GPU, publishes no ports, and cannot write outside `docs/evidence/portability`. It checks
the required server extensions, HTTP/gRPC metadata and binary inference, exact
cross-protocol results, dynamic-batching statistics, ResNet version switching, a
reload without unload, and final cleanup. Cleanup checks every model and version
readiness endpoint in addition to the repository index; batching records and
validates `attempts_used` against the bounded attempt history.

The original `docs/evidence/step-4` bundle is immutable and self-contained through
its runtime model-spec and manifest snapshots. New serving verification writes only
the separate portability proof.

`deployment/triton/smoke_models.py --check` only validates immutable
`docs/evidence/step-3`; it has no runtime or write path. This verifier is not the production image client;
preprocessing, postprocessing, user-facing requests, and inference logging belong to
step 5.
