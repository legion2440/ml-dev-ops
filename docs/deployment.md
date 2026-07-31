# Deployment

## Current status

The Docker infrastructure is runtime-verified on the reference host as of
2026-07-31. Static validators and canonical Compose configuration pass, all four
services reach Compose health `healthy`, GPU passthrough works, and the complete
host-side smoke test succeeds.

Step 2 evidence was captured while the repository was empty. Step 3 now provides
three model directories, but Triton still starts with no models loaded because
explicit model control is enabled. Model-serving claims are recorded separately in
`docs/evidence/step-3`.

## Version matrix

`.env.example` is the editable source of truth for container images.

| Component | Pinned version |
| --- | --- |
| Triton image | `TRITON_IMAGE` in `.env.example` |
| Triton Server | `2.71.0` |
| Triton Ubuntu base | `24.04` |
| CUDA in Triton | `13.3.4.1` |
| TensorRT in Triton | `11.1.0.106` |
| ONNX Runtime in Triton | `1.27.0` |
| Prometheus image | `PROMETHEUS_IMAGE` in `.env.example` |
| Grafana image | `GRAFANA_IMAGE` in `.env.example` |
| DCGM Exporter image | `DCGM_EXPORTER_IMAGE` in `.env.example` |

CUDA, TensorRT, and ONNX Runtime versions are properties of the selected Triton
image. They are documented but are not independent Compose variables. TensorRT
engines remain target-GPU and runtime dependent and will be built in a compatible
container during model preparation.

## Host requirements

- Linux `amd64` container support
- Docker Engine and Docker Compose v2
- NVIDIA GPU with compute capability 7.5 or later
- NVIDIA driver compatible with CUDA 13
- NVIDIA Container Toolkit
- Python 3.10 or newer for validation and smoke checks
- Bash for lifecycle scripts

CUDA 13 minor-version compatibility requires driver branch 580 or newer. The
selected CUDA 13.3 toolkit corresponds to driver `610.43.02` or newer; runtime
execution remains the authoritative compatibility test.

On Windows, Docker Desktop exposes GPUs only through its WSL2 backend. Use an
up-to-date Windows installation, NVIDIA WSL driver, WSL version and kernel, and
Linux-container mode.

Host diagnostics:

```text
docker version
docker compose version
docker info
nvidia-smi
wsl --version
```

## Configuration selection

Lifecycle scripts choose `.env` when it exists and otherwise use `.env.example`.
They do not execute either file. Every Compose invocation includes
`--project-directory`, `--file`, and `--env-file`.

Canonical clean-checkout validation:

```text
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

Critical Compose substitutions fail with a clear message when absent.

## Topology

All services share the project-scoped `backend` bridge network:

- `triton`
- `prometheus`
- `grafana`
- `dcgm-exporter`

Inter-container traffic uses these DNS names and fixed container ports. Published
host ports bind to `127.0.0.1` only.

| Service | Container port | Default host port |
| --- | ---: | ---: |
| Triton HTTP | 8000 | 8000 |
| Triton gRPC | 8001 | 8001 |
| Triton metrics | 8002 | 8002 |
| Prometheus | 9090 | 9090 |
| Grafana | 3000 | 3000 |
| DCGM metrics | 9400 | 9400 |

## Storage

- `models` is mounted into Triton at `/models` read-only.
- Prometheus data uses the `prometheus-data` named volume.
- Grafana data uses the `grafana-data` named volume.
- Prometheus configuration is a read-only bind mount.
- Grafana provisioning and dashboard directories are read-only bind mounts.
- Triton logs only to stdout and stderr.

Inspect Triton logs:

```text
docker compose --project-directory . --file docker-compose.yml --env-file .env.example logs triton
```

## Lifecycle

### NVIDIA NGC authentication

The Triton base is a Docker image and must not be copied into the repository. Docker
Desktop stores its layers in the local image store. If registry authentication is
required, create an NGC personal API key with NGC Catalog access, then log in with
the literal username `$oauthtoken` and paste the key at the password prompt:

```text
docker login nvcr.io --username '$oauthtoken'
docker compose --project-directory . --file docker-compose.yml --env-file .env.example build --pull triton
```

Never put the API key in `.env`, `.env.example`, a script, or shell history.

```text
bash deployment/scripts/run_environment.sh
bash deployment/scripts/check_environment.sh
python deployment/scripts/smoke_environment.py
bash deployment/scripts/stop_environment.sh
```

Run only Triton:

```text
bash deployment/scripts/run_triton.sh
```

Explicitly remove only Compose named volumes:

```text
bash deployment/scripts/stop_environment.sh --purge
```

The purge mode never deletes `models`, monitoring configuration, benchmark data, or
other repository-owned host directories.

## Health and startup policy

Services start in parallel without hard `depends_on` relationships.

- Triton health uses `/v2/health/live`.
- Prometheus health uses `/-/healthy`.
- Grafana health uses `/api/health`.
- The distroless DCGM image is checked by directly executing
  `/usr/bin/dcgm-exporter --version`; no shell is assumed.

The host-side infrastructure smoke separately checks Triton readiness and metrics,
that no model is loaded before model smoke, Prometheus targets, the Grafana
datasource, and real `DCGM_` metrics.

Triton runs with `--model-control-mode=explicit` and
`--disable-auto-complete-config`. The latter prevents the TensorRT backend from
silently enabling dynamic batching; complete configs are generated from the model
spec. Step 3 verifies explicit load and unload for all three models.

## Completion gates

Code-complete requires:

```text
python scripts/validate_structure.py
python scripts/validate_module_map.py
python scripts/validate_deployment.py
python scripts/validate_runtime_evidence.py
python -m unittest discover -s tests/unit -p "test_*.py"
python scripts/generate_dependency_graph.py --check
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

Runtime-verified additionally requires:

- GPU visibility inside containers;
- all four expected containers running;
- acceptable Compose health;
- successful `deployment/scripts/smoke_environment.py`;
- Prometheus targets up;
- provisioned Grafana datasource;
- real DCGM GPU metrics;
- no model in ready state before the dedicated model smoke.

The 2026-07-31 reference run passed every runtime requirement with an NVIDIA
GeForce RTX 4080 Laptop GPU, driver 610.88, and compute capability 8.9. Runtime
evidence is host-specific and must be regenerated after material host, driver,
Docker, image, or Compose changes.

Capture a sanitized snapshot after a successful runtime check:

```text
python deployment/scripts/capture_runtime_evidence.py
```

The command writes `smoke.json`, `compose-ps.txt`, and `environment.txt` under
`docs/evidence/step-2`. The snapshot contains no secrets, container IDs, hostnames,
or workspace paths. `scripts/validate_runtime_evidence.py` checks the recorded
services and smoke results and detects image evidence stale relative to
`.env.example` without requiring Docker.

## Recovery and cleanup

Configuration errors should be corrected before restarting:

```text
python scripts/validate_deployment.py
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

Inspect service state and logs:

```text
docker compose --project-directory . --file docker-compose.yml --env-file .env.example ps
docker compose --project-directory . --file docker-compose.yml --env-file .env.example logs
```

Restarting with `run_environment.sh` is idempotent. Persistent metric and Grafana
data survive normal `down`.

## References

- [Triton release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/)
- [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [DCGM Exporter installation](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html)
- [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)

## Boundary

Deployment orchestrates containers and configuration. It does not download models,
export ONNX, build TensorRT engines, preprocess images, or contain inference
postprocessing.
