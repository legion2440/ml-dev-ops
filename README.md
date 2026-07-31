# ML DevOps with NVIDIA Triton

This project will demonstrate production-style serving of pretrained
computer-vision models with NVIDIA Triton Inference Server. The target includes
classification and object detection, ONNX and TensorRT optimization, reproducible
containers, structured inference history, benchmarking, Prometheus metrics, Grafana
dashboards, GPU monitoring, alerting, and model version management.

## Current status

**Step 1: architecture and repository scaffolding.**

The repository currently contains module boundaries, machine-readable architecture
metadata, generated dependency documentation, and structural validators. Triton,
model artifacts, Docker services, the Python inference client, monitoring, and
benchmarks are not implemented yet.

`docker-compose.yml` is intentionally a valid empty scaffold. It does not start any
services.

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

## Step 1 requirements

- Python 3.10 or newer
- packages from `requirements.txt`
- Make is optional

GPU drivers, Docker, NVIDIA Container Toolkit, and runtime image requirements will
be pinned during the deployment scope.

## Architecture checks

Install the current validation dependency:

```text
python -m pip install -r requirements.txt
```

Run all step 1 checks:

```text
make validate
```

On systems without Make, run:

```text
python scripts/validate_structure.py
python scripts/validate_module_map.py
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
