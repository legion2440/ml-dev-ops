# Troubleshooting

## Triton image pull fails or stalls

The required artifact is the Docker image selected by `TRITON_IMAGE` in
`.env.example`; it does not belong under `models` or any other repository directory.
If Docker reports an authorization error, authenticate to NVIDIA NGC with a personal
API key that has NGC Catalog access:

```text
docker login nvcr.io --username '$oauthtoken'
docker compose --project-directory . --file docker-compose.yml --env-file .env.example build --pull triton
```

Paste the API key only at Docker's password prompt. An authorization failure is
normally reported immediately; a pull that transfers layers for a long time is more
likely limited by image size, network throughput, proxy behavior, or Docker Desktop
storage.

## Architecture validation

If a generated graph is stale, run:

```text
python scripts/generate_dependency_graph.py
```

If Make is unavailable, use the direct Python commands documented in `README.md`.

An `implemented` path must exist. A `generated` path must exist and its generator
must pass `--check`. A `planned` path may be absent.

## Docker daemon unavailable

Confirm that Docker Desktop or Docker Engine is running:

```text
docker version
docker info
```

On Windows, select the WSL2 backend and Linux containers.

## NVIDIA runtime or GPU unavailable

Check the host driver first:

```text
nvidia-smi
docker info
```

On Windows, update WSL and confirm Docker Desktop GPU support. A successful host
`nvidia-smi` does not replace container runtime verification.

## Port already allocated

Change only the host-side port in `.env`. Container ports and service-to-service
targets remain fixed. Re-run canonical Compose validation before startup.

## Triton cannot read the model repository

`MODEL_REPOSITORY_PATH` must be a repository-relative directory. The mount is
read-only by design. The three model directories and four versioned binaries are
declared in `models/model-spec.yaml`. Before verification, run `make prepare-models`.

Inspect:

```text
docker compose --project-directory . --file docker-compose.yml --env-file .env.example logs triton
```

## Prometheus target is down

Inspect `http://127.0.0.1:9090/targets` and service logs. Internal targets must be
`triton:8002` and `dcgm-exporter:9400`, not loopback addresses.

## Grafana datasource is absent

Check that the datasource provisioning file is mounted read-only and inspect
Grafana logs. The expected datasource UID is `prometheus`, with URL
`http://prometheus:9090`.

## DCGM metrics are absent

Inspect DCGM Exporter logs, confirm NVIDIA runtime availability, and query
`http://127.0.0.1:9400/metrics`. A valid response must contain `DCGM_` metrics.
The distroless image has no shell, curl, or wget.

## Volume permissions or stale data

Normal shutdown preserves `prometheus-data` and `grafana-data`. Use
`stop_environment.sh --purge` only when intentionally removing those named volumes.
The command does not remove host directories.

## Source weight hash mismatch

Do not replace the accepted hash automatically. Run the non-mutating discovery
command and compare the candidate with the upstream release:

```text
python scripts/model_preparation/prepare_models.py discover
```

Only an intentional review should change `models/model-spec.yaml`. A partial or
mismatched download is removed by the preparation workflow.

## TensorRT reports `Unknown option: --fp16`

TensorRT 11.1 removed `trtexec --fp16` because networks are strongly typed. The
step 3 workflow converts the ResNet graph to an ignored FP16 internal ONNX with
FP32 boundary casts, then invokes `trtexec` without that historical flag. Use the
repository command rather than calling an older recipe manually:

```text
python scripts/model_preparation/prepare_models.py build-tensorrt
```

## TensorRT plan is rejected on another GPU

The capability-qualified plan is intentionally bound to the compute capability
and TensorRT version recorded in the spec and generated manifest. Rebuild it on
the target host. Do not rename it to `model.plan` or represent it as a portable
engine.

## Triton model load fails

If `make verify-serving` spends a long time before creating a container, Docker is
usually downloading the large `TRITON_SDK_IMAGE`. This requires Docker storage but
does not require placing any file in the repository. Pull the exact SDK pin from
`.env.example` to separate download problems from verifier failures.

If batching evidence fails, do not weaken the criterion. The verifier retries at
most three times and requires statistics deltas with fewer executions than batch-1
inferences plus an observed batch size greater than one.

Inspect the generated config, local binary, and server logs:

```text
python scripts/validate_model_repository.py
docker compose --project-directory . --file docker-compose.yml --env-file .env.example logs triton
```

Triton config auto-completion must remain disabled. The current verifier performs
best-effort unload on failure and proves model/version readiness is false on success.

## Model evidence is stale

Step 3 evidence is immutable and must not be regenerated against the v2 manifest.
Check the historical snapshot, then run current verification separately:

```text
python scripts/model_preparation/prepare_models.py manifest
python deployment/triton/smoke_models.py --check
make verify-serving
```

## Client rejects a model or batch before connecting

The runtime client validates task, version, and maximum batch size against
`shared/client-model-contracts.json` before image discovery or a network request.
Regenerate tracked projections after an intentional model-manifest change:

```text
python scripts/model_preparation/prepare_models.py client-contract
python scripts/model_preparation/prepare_models.py --check
```

Do not make the client read `models/model-spec.yaml` or
`models/model-manifest.json`; that would violate the declared module boundary.

## Client reports that a model is not READY

Auto-load is enabled by default and uses Triton's HTTP repository API even when
inference uses gRPC. Confirm Triton is reachable and the local artifacts exist:

```text
python client/inference_client.py health
python scripts/validate_model_repository.py
```

If `--no-auto-load` was supplied, load the model through the serving workflow or
remove that option. A normal client request intentionally leaves the model READY.

## Inference log or CSV is invalid

JSONL is the primary append-only history. Do not edit the CSV and expect the client
to consume it. Validate operational history by exporting it again:

```text
python client/inference_client.py export-logs --input-log logs/inference.jsonl --output-csv logs/inference.csv
```

A malformed image is rejected before request-ID creation. Failures after a request
starts produce a sanitized error event without raw inputs or absolute host paths.

## Step 5 evidence is stale

Source, contract, sample-manifest, dependency, JSONL, CSV, and transcript hashes are
validated without contacting Triton. Refresh the snapshot only against a live,
prepared server:

```text
make verify-client
python scripts/validate_client_evidence.py
```

The verifier refuses a partially READY model state it cannot reproduce exactly and
unloads only models that it loaded itself.
