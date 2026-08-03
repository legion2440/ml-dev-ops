# ML DevOps with NVIDIA Triton

This project will demonstrate production-style serving of pretrained
computer-vision models with NVIDIA Triton Inference Server. The target includes
classification and object detection, ONNX and TensorRT optimization, reproducible
containers, structured inference history, benchmarking, Prometheus metrics, Grafana
dashboards, GPU monitoring, alerting, and model version management.

## Current status

**Step 7 runtime-verified: Prometheus, Grafana dashboard, GPU monitoring, and alert rules passed.**

The repository defines pinned Triton, Prometheus, Grafana, and DCGM Exporter
services; GPU reservations; loopback-only ports; persistent metrics volumes;
read-only configuration mounts; lifecycle commands; and infrastructure smoke
checks.

The reference Windows 11, Docker Desktop/WSL2, and NVIDIA GPU host verifies the
three serving models through both HTTP and gRPC. The production client accepts real
JPG/PNG files or directories, performs contract-driven ResNet and YOLO preprocessing,
decodes predictions, auto-loads an unavailable model, appends one JSONL event per
Triton request, and exports the history to CSV. The formal benchmark now compares
the ResNet50 ONNX and TensorRT implementations through the pinned SDK Perf Analyzer;
the provisioned Grafana dashboard visualizes live Triton and DCGM data, and
Prometheus loads the two project alert rules. Client timing remains operational
diagnostics, not benchmark evidence.

Cleanup evidence covers both the repository READY set and every model-level and
version-specific readiness endpoint. Batching evidence records `attempts_used` and
permits no more than three attempts.

Step 5 runtime evidence covers all ten tracked sample images, ONNX versions 1 and 2,
TensorRT, HTTP and gRPC, 11 logged requests, and exact restoration of the READY set.

Step 6 measures 16 clean Perf Analyzer runs: four paired AB/BA repetitions for
latency and throughput. PASS requires a positive median paired improvement and at
least three of four pairs improving for each primary metric. Five percent labels a
strong result but is not a gate. Windows-host process telemetry permits a same-slot
replacement only for objectively attributed foreign GPU activity. PA stability,
thermal/power drift, and clocks do not exclude a valid measurement. The earlier
5%-stability/5%-improvement contract is superseded. Superseded diagnostic runs are
intentionally excluded from committed Step 6 evidence and may exist only in the
local ignored benchmark cache. The published formal bundle is self-contained.

The published reference run passed 4/4 pairs in both scenarios, with median paired
improvements of 19.32% for mean client latency and 114.11% for throughput. Two
objectively attributed contaminated attempts remain in the evidence beside their
same-slot replacements. Those attempts were classified conservatively from Windows
host activity and were directionally consistent with the published TensorRT gain;
they were not required to establish the optimization conclusion.

Step 7 verifies the complete Triton -> Prometheus -> Grafana chain through Grafana's
provisioned datasource proxy. A 35-second classification workload crossed two scrape
intervals, all five dashboard queries returned numeric data, both alert definitions
were loaded, GPU identity matched `nvidia-smi`, and the initial empty READY set was
restored. GPU utilization may validly be zero; a positive `max_over_time` observation
is retained only as supporting evidence, not an acceptance gate.

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

## Run the production client

Install the single host dependency set and start Triton with prepared models:

```text
python -m pip install -r requirements.txt
python client/inference_client.py health
```

Run classification, detection, metadata, and CSV export:

```text
python client/inference_client.py classify client/samples/01_dog.jpg
python client/inference_client.py classify client/samples/ --model resnet50_tensorrt --protocol grpc --batch-size 4
python client/inference_client.py detect client/samples/ --protocol http --batch-size 2
python client/inference_client.py metadata --model resnet50_onnx --version 2
python client/inference_client.py export-logs --input-log logs/inference.jsonl --output-csv logs/inference.csv
```

Models are loaded through Triton's HTTP repository API when needed and remain READY
after a normal client request. Operational `logs/*.jsonl` and `logs/*.csv` files are
ignored by Git. See `docs/client.md` for the CLI and event contracts.

## Run the formal benchmark

With prepared artifacts and healthy Triton:

```text
make benchmark
```

The Windows host records process-attributed GPU telemetry while the SDK container
writes only to ignored `.cache/benchmarking` during measurement. The host publishes
tracked results only after both gates and the independent raw-data validator pass.
Each PA pass records an acknowledged WDDM sequence range and explicit-version
Triton request/queue/compute deltas for diagnosis; those fields never alter formal
classification. See `docs/benchmarking.md` for scenarios, boundary handshakes,
isolation, metrics, and direct commands.

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

Grafana's Prometheus datasource and the `ML DevOps Inference` dashboard are
provisioned automatically. To generate a short controlled workload, verify the
complete monitoring chain, and then inspect the dashboard:

```text
make up
make verify-monitoring
```

Open `http://127.0.0.1:3000/d/ml-dev-ops-inference/ml-dev-ops-inference` and use the
credentials selected by `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD`. The
verifier writes only a temporary ignored log, waits at least two scrape intervals,
and restores the exact initial READY set.

## Validation

Install the validation and production-client dependencies:

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
python scripts/validate_client.py
python scripts/validate_client_evidence.py
python scripts/validate_benchmark.py
python scripts/validate_benchmark_evidence.py
python scripts/validate_monitoring.py
python scripts/validate_repository_hygiene.py
python scripts/generate_dependency_graph.py --check
python -m unittest discover -s tests/unit -t . -p "test_*.py"
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

The committed Step 2 snapshot is under `docs/evidence/step-2`. Its
`runtime-integrity.json` separates the exact source manifest captured for the run
from the current four-service compatibility projection. Refresh it only after a
successful live run:

```text
python deployment/scripts/capture_runtime_evidence.py
```

`python scripts/validate_runtime_evidence.py --check` validates historical
integrity plus current compatibility. `--historical-only` validates the immutable
snapshot alone. Both modes are read-only.

Step 3 runtime evidence under `docs/evidence/step-3` is immutable and bound to its
manifest v1 snapshot. `make smoke-models` is retained only as a read-only historical
evidence check; it never contacts Triton or rewrites evidence.

Step 4 evidence is under `docs/evidence/step-4` and is refreshed only by a
successful `make verify-serving` run. The verifier uses the official SDK image and
mounts the repository read-only except for that evidence directory.

Step 5 evidence is under `docs/evidence/step-5`. Refresh it only against live Triton
with `make verify-client`; validate the tracked snapshot without a daemon with
`make validate-client-evidence`. The verifier unloads only models it loaded and
requires the final READY set to equal the initial set.

Step 6 evidence is under `docs/evidence/step-6`. Refresh it only through `make
benchmark`; validate it without a daemon with `make validate-benchmark-evidence`.
Failed runs remain only in ignored cache and do not modify the last passing bundle.
The stored `runtime_source_fingerprint_sha256` and per-file source manifest describe
the source state used by the run. The separately hashed semantic projection is the
only current-compatibility gate. Its PA command, aggregation, guard classification,
and replacement fields are outputs of deterministic probes against the production
code, not parallel declarations. Unrelated repository evolution is reported as
non-gating provenance drift. Use `--historical-only` to audit the run without the
current compatibility check.

Step 7 evidence is under `docs/evidence/step-7`. Refresh both compact JSON files only
against the live four-service stack with `make verify-monitoring`; validate the
tracked snapshot without contacting any service with `make validate-monitoring`.

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
