# Troubleshooting

## Triton image pull fails or stalls

The required artifact is the Docker image
`nvcr.io/nvidia/tritonserver:26.07-py3`; it does not belong under `models` or any
other repository directory. Authenticate to NVIDIA NGC with a personal API key that
has NGC Catalog access:

```text
docker login nvcr.io --username '$oauthtoken'
docker pull nvcr.io/nvidia/tritonserver:26.07-py3
docker image inspect nvcr.io/nvidia/tritonserver:26.07-py3
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
read-only by design. During step 2 the directory should contain no Triton model
directories, and the smoke test expects an empty repository index.

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

## Models, inference, and benchmarks

Troubleshooting for model export, inference, version policy, client behavior, and
benchmarks will be added with their implementation scopes.
