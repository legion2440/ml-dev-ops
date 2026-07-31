# ML DevOps with NVIDIA Triton

This project will demonstrate production-style serving of pretrained
computer-vision models with NVIDIA Triton Inference Server. The target includes
classification and object detection, ONNX and TensorRT optimization, reproducible
containers, structured inference history, benchmarking, Prometheus metrics, Grafana
dashboards, GPU monitoring, alerting, and model version management.

## Current status

**Step 2 runtime-verified: reproducible Docker infrastructure.**

The repository defines pinned Triton, Prometheus, Grafana, and DCGM Exporter
services; GPU reservations; loopback-only ports; persistent metrics volumes;
read-only configuration mounts; lifecycle commands; and infrastructure smoke
checks.

The complete infrastructure smoke test passed on the reference Windows 11,
Docker Desktop/WSL2, and NVIDIA GPU host on 2026-07-31. No model artifacts,
inference implementation, dynamic batching, dashboards, alerts, or benchmarks are
present yet. Triton starts with the expected empty model repository.

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
