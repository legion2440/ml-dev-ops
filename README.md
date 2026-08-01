# ML DevOps with NVIDIA Triton

This project will demonstrate production-style serving of pretrained
computer-vision models with NVIDIA Triton Inference Server. The target includes
classification and object detection, ONNX and TensorRT optimization, reproducible
containers, structured inference history, benchmarking, Prometheus metrics, Grafana
dashboards, GPU monitoring, alerting, and model version management.

## Current status

**Step 4 runtime-verified: batching, HTTP/gRPC, and model-version control.**

The repository defines pinned Triton, Prometheus, Grafana, and DCGM Exporter
services; GPU reservations; loopback-only ports; persistent metrics volumes;
read-only configuration mounts; lifecycle commands; and infrastructure smoke
checks.

The reference Windows 11, Docker Desktop/WSL2, and NVIDIA GPU host verifies the
three serving models through both HTTP and gRPC. Runtime statistics prove dynamic
batching for a concrete version of every model. ResNet50 ONNX versions 1 and 2 are
selected explicitly, loaded together, and reloaded without a preliminary unload;
cleanup leaves no model READY. The production image client, dashboards, alerts, and
benchmarks remain planned. Batching values are functional defaults, not performance
tuning results.

Model binaries are reproducible local artifacts and are ignored by Git. The model
specification, configs, labels, manifest, and sanitized runtime evidence are tracked.

## Planned architecture

An image is preprocessed by a Python client and sent to Triton over REST or gRPC.
Triton loads versioned ONNX or TensorRT models from its model repository. Triton and
GPU metrics flow to Prometheus and Grafana. The client records JSONL history that can
be exported to CSV.

See `ARCHITECTURE.md` for component contracts and
`docs/generated/dependency-graph.md` for the generated dependency graph.

## Repository layout

```text
ml-dev-ops/
├── models/                 Triton model repository
├── deployment/             Container and service lifecycle
├── monitoring/             Prometheus, Grafana, GPU metrics, and alerts
├── client/                 Inference client, samples, and logging
├── benchmarks/             Load profiles, raw data, results, and reports
├── scripts/                Architecture checks and model preparation
├── shared/                 Cross-module schemas and DTOs only
├── tests/                  Unit, integration, and fixture data
├── schemas/                JSON Schema for architecture metadata
├── docs/                   Design, operations, and audit documentation
├── module-map.json         Module navigation metadata
└── dependency-graph.json   Allowed and forbidden dependencies
```

## Deployment requirements

- Python 3.10 or newer
- packages from `requirements.txt`
- Docker Engine with Docker Compose v2
- an NVIDIA GPU with a compatible driver
- NVIDIA Container Toolkit
- Linux `amd64` containers
- Bash for lifecycle scripts; Git Bash is supported on Windows
- Docker Desktop with the WSL2 backend for GPU containers on Windows

The exporter and TensorRT images require additional local Docker storage. Together
with Triton they can consume several tens of gigabytes.

The pinned image and runtime matrix is documented in `docs/deployment.md`.

## Environment configuration

`.env.example` is the canonical configuration. A local `.env` may override it and
is ignored by Git. Lifecycle scripts select `.env` when present and otherwise use
`.env.example`; neither file is executed as shell code.

Validate the clean-checkout configuration:

```text
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

## Start and inspect the infrastructure

With Make:

```text
make up
make status
make smoke
make down
```

Direct equivalents:

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

## Prepare and verify models

Build all three local artifacts and reach artifact-complete state:

```text
make prepare-models
```

With Triton running, reach runtime-verified state:

```text
make smoke-models
make verify-serving
```

Structure-only validation works on a clean checkout without Docker or model binaries:

```text
make validate-model-structure
```

Full artifact validation requires Docker and the target GPU declared in the model spec:

```text
make validate-models
```

See `docs/model-preparation.md` for source hashes, exact contracts, cleanup,
portability limits, and direct Python equivalents.

Remove only the Prometheus and Grafana named volumes:

```text
bash deployment/scripts/stop_environment.sh --purge
```

The purge command does not delete repository-owned host directories.

## Local endpoints

Default ports are bound to loopback:

| Service | Address |
| --- | --- |
| Triton HTTP | `http://127.0.0.1:8000` |
| Triton gRPC | `127.0.0.1:8001` |
| Triton metrics | `http://127.0.0.1:8002/metrics` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |
| DCGM metrics | `http://127.0.0.1:9400/metrics` |

Grafana's Prometheus datasource is provisioned automatically. The dashboard file is
planned for step 7.

## Validation

Install validation dependencies:

```text
python -m pip install -r requirements.txt
```

Run all code-complete checks:

```text
make validate
```

On systems without Make, run:

```text
python scripts/validate_structure.py
python scripts/validate_module_map.py
python scripts/validate_deployment.py
python scripts/validate_runtime_evidence.py
python scripts/validate_model_repository.py --structure-only
python scripts/validate_model_evidence.py
python scripts/validate_serving.py --structure-only
python scripts/validate_serving_evidence.py
python -m unittest discover -s tests/unit -p "test_*.py"
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

Regenerate or check the derived dependency documentation:

```text
make architecture
make check-architecture
```

Equivalent direct commands are:

```text
python scripts/generate_dependency_graph.py
python scripts/generate_dependency_graph.py --check
```

`scripts/validate_deployment.py` reports `[SKIP]` when `promtool` is unavailable.
The Compose and Python YAML checks remain mandatory. A successful GPU smoke test is
required before the deployment is considered runtime-verified.

The committed runtime snapshot is under `docs/evidence/step-2`. Refresh it only
after a successful live run:

```text
python deployment/scripts/capture_runtime_evidence.py
```

Model preparation and Triton evidence is under `docs/evidence/step-3`. Refresh it
with `prepare_models.py manifest` after an artifact rebuild and with
`deployment/triton/smoke_models.py` after a successful live model run.

Step 4 evidence is under `docs/evidence/step-4` and is refreshed only by a
successful `make verify-serving` run. The verifier uses the official SDK image and
mounts the repository read-only except for that evidence directory.

## License

This educational repository is licensed under AGPL-3.0-only because it uses the
open-source Ultralytics YOLO11 workflow. Closed commercial use requires a different
model/toolchain or an appropriate commercial license. Third-party software,
pretrained weights, datasets, containers, and NVIDIA runtimes retain their own terms;
see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Delivery roadmap

1. Architecture, repository structure, and agent navigation
2. Reproducible Docker infrastructure
3. Model preparation and Triton model repository
4. Triton serving, batching, and model versioning
5. Python client, sample images, and inference logging
6. Baseline and optimized benchmarks
7. Prometheus, Grafana, GPU monitoring, and alerting
8. Automated tests and quality gates
9. Documentation and audit evidence

Each implementation step includes its code, scoped tests, documentation, and
architecture metadata updates.
