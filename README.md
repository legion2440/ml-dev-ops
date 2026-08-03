# ML DevOps with NVIDIA Triton

Production-style computer-vision inference with NVIDIA Triton Inference Server, ONNX and TensorRT models, reproducible Docker deployment, benchmarking, Prometheus metrics, Grafana dashboards, GPU monitoring, persistent inference logs, and explicit model version management.

The repository serves ResNet50 classification and YOLO11n object detection, compares an ONNX baseline with an FP16 TensorRT optimization, exposes inference over HTTP and gRPC, and keeps committed runtime evidence for the main assignment requirements.

· [Русская версия](README_RU.md)

## 📋 TOC

- [🚀 Quick start](#-quick-start)
- [📝 About](#-about)
- [✨ Features](#-features)
- [🔄 Architecture](#-architecture)
- [🧠 Models](#-models)
- [⚡ Triton serving](#-triton-serving)
- [📈 Benchmark](#-benchmark)
- [📊 Monitoring](#-monitoring)
- [🖼️ Python client and samples](#️-python-client-and-samples)
- [🧾 Logging and CSV export](#-logging-and-csv-export)
- [🔎 Runtime evidence](#-runtime-evidence)
- [🧰 Technology stack](#-technology-stack)
- [🧪 Tests and checks](#-tests-and-checks)
- [📁 Project structure](#-project-structure)
- [⚠️ Notes](#️-notes)
- [🧑‍💻 Author](#-author)

## 🚀 Quick start

### Prerequisites

- Python `3.10+`
- Docker Engine with Docker Compose v2
- NVIDIA GPU with a compatible driver
- NVIDIA Container Toolkit
- Bash; Git Bash is supported on Windows
- Docker Desktop with the WSL2 backend when running GPU containers on Windows

Model export and NVIDIA runtime images are large. Keep several tens of gigabytes of free Docker storage available.

### Clone and install

```bash
git clone https://github.com/legion2440/ml-dev-ops.git
cd ml-dev-ops

python -m pip install -r requirements.txt
```

`.env.example` is the canonical configuration. A local `.env` is optional and ignored by Git.

Check the clean configuration:

```bash
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

### Prepare models

```bash
python scripts/model_preparation/prepare_models.py prepare
```

This creates the local model artifacts used by Triton. Model binaries are reproducible build artifacts and are intentionally not committed.

### Start the stack

```bash
bash deployment/scripts/run_environment.sh
```

Check the environment:

```bash
bash deployment/scripts/check_environment.sh
python deployment/scripts/smoke_environment.py
```

### Run inference

Health:

```bash
python client/inference_client.py health
```

ResNet50 classification:

```bash
python client/inference_client.py classify client/samples/01_dog.jpg
```

TensorRT classification over gRPC:

```bash
python client/inference_client.py classify client/samples/ \
    --model resnet50_tensorrt \
    --protocol grpc \
    --batch-size 4
```

YOLO detection:

```bash
python client/inference_client.py detect client/samples/ \
    --protocol http \
    --batch-size 2
```

Stop the stack:

```bash
bash deployment/scripts/stop_environment.sh
```

If GNU Make is available, the same workflows are exposed through targets such as `make prepare-models`, `make up`, `make verify-serving`, `make verify-monitoring`, `make benchmark`, and `make validate`.

## 📝 About

The project implements a complete ML inference delivery path around NVIDIA Triton rather than a standalone model script.

The model repository contains versioned ResNet50 and YOLO11n serving contracts. A reusable Python client performs preprocessing, sends requests over Triton HTTP or gRPC, decodes predictions, and records one structured event per request. Triton and GPU metrics are scraped by Prometheus and visualized in a provisioned Grafana dashboard.

A formal benchmark compares the same ResNet50 workload in ONNX Runtime and TensorRT. Historical runtime evidence is committed separately from current semantic compatibility checks, so later documentation or monitoring changes do not rewrite the original benchmark run.

The repository is designed so that code, configuration, generated contracts, runtime evidence, and audit documentation can be checked independently.

## ✨ Features

### Model serving

- NVIDIA Triton Inference Server in Docker;
- explicit model control;
- read-only Triton model repository mount;
- ResNet50 classification;
- YOLO11n object detection;
- ONNX Runtime and TensorRT backends;
- HTTP and gRPC inference;
- explicit model version selection;
- load, unload, reload, and default-version behavior;
- dynamic batching with runtime evidence.

### Model optimization

- ResNet50 ONNX FP32 baseline;
- TensorRT FP16 compute with FP32 public I/O;
- shared source weights between baseline and optimized variants;
- parity checks before benchmark publication;
- deterministic benchmark input;
- committed baseline, optimized, comparison, raw, and report artifacts.

### Observability

- Triton Prometheus metrics;
- DCGM GPU metrics;
- provisioned Grafana datasource;
- provisioned `ML DevOps Inference` dashboard;
- inference throughput;
- request rate;
- average request latency;
- GPU utilization;
- failed request count;
- Prometheus rules for high latency and inference failures.

### Client and evidence

- reusable Python inference client;
- contract-driven preprocessing and postprocessing;
- ten tracked real sample images with provenance;
- JSONL inference history;
- deterministic CSV export;
- sanitized committed runtime evidence;
- tamper and staleness validation;
- repository hygiene checks.

## 🔄 Architecture

```text
                         +----------------------+
                         |  JPG / PNG samples   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Python client     |
                         | preprocess / decode  |
                         +----+------------+----+
                              |            |
                         HTTP |            | gRPC
                              v            v
                    +-----------------------------+
                    | NVIDIA Triton Inference     |
                    | Server                      |
                    |                             |
                    | ResNet50 ONNX v1 / v2       |
                    | ResNet50 TensorRT v1        |
                    | YOLO11n ONNX v1             |
                    +------+----------------------+
                           |
              +------------+-------------+
              |                          |
              v                          v
    +-------------------+       +-------------------+
    | Triton /metrics   |       | JSONL request log |
    +---------+---------+       +---------+---------+
              |                           |
              v                           v
    +-------------------+       +-------------------+
    | Prometheus        |       | CSV export        |
    +---------+---------+       +-------------------+
              |
        +-----+--------------------+
        |                          |
        v                          v
+---------------+          +---------------+
| Grafana       |          | Alert rules   |
| dashboard     |          | latency/error |
+---------------+          +---------------+

DCGM Exporter --------------------> Prometheus --------------------> Grafana
```

Triton uses explicit model control, so model lifecycle is observable and testable rather than hidden behind automatic repository scanning.

See `ARCHITECTURE.md` and `docs/generated/dependency-graph.md` for module ownership and dependency boundaries.

## 🧠 Models

| Model               | Backend      | Versions | Precision | Task                                    |
| ------------------- | ------------ | -------: | --------- | --------------------------------------- |
| `resnet50_onnx`     | ONNX Runtime | `1`, `2` | FP32      | ImageNet-1K classification              |
| `resnet50_tensorrt` | TensorRT     | `1`      | FP16 compute, FP32 I/O | ImageNet-1K classification |
| `yolo11n_onnx`      | ONNX Runtime | `1`      | FP32      | COCO object detection                   |

ResNet50 ONNX and TensorRT use the same source weights. Preparation selects one GPU, builds and validates a canonical `model.plan` on that device, and records host provenance separately from the portable model semantics.

The repository tracks model specifications, Triton configs, labels, source hashes, licenses, generated manifests, and runtime evidence. Large model binaries remain local.

Prepare artifacts:

```bash
python scripts/model_preparation/prepare_models.py prepare
```

Structure-only validation does not require model binaries:

```bash
python scripts/validate_model_repository.py --structure-only
```

See `docs/model-preparation.md` and `docs/model-versioning.md`.

## ⚡ Triton serving

The serving layer verifies:

- server liveness and readiness;
- model metadata and configuration;
- HTTP inference;
- gRPC inference;
- numerical protocol parity;
- explicit model versions;
- default version selection;
- load and unload;
- in-place reload;
- dynamic batching;
- final READY-state cleanup.

Inspect metadata for ResNet50 ONNX v2:

```bash
python client/inference_client.py metadata \
    --model resnet50_onnx \
    --version 2
```

The client can load an unavailable model through Triton's repository-control API before inference.

The immutable Step 4 serving evidence and the current GPU-portability proof are stored separately:

```text
docs/evidence/step-4/
docs/evidence/portability/
```

## 📈 Benchmark

The formal benchmark compares:

```text
baseline:  resnet50_onnx:v1
optimized: resnet50_tensorrt:v1
```

Both variants use the same ResNet50 weights and public FP32 tensor contract.

### Formal scenarios

| Scenario   | Batch | Concurrency | Primary metric      |
| ---------- | ----: | ----------: | ------------------- |
| Latency    | `1`   | `1`         | mean client latency |
| Throughput | `8`   | `4`         | inferences / second |

The published run uses four paired repetitions in balanced order:

```text
ONNX -> TensorRT
TensorRT -> ONNX
ONNX -> TensorRT
TensorRT -> ONNX
```

### Published result

| Metric                               | Result      |
| ------------------------------------ | ----------: |
| Median paired latency improvement    | **19.32%**  |
| Latency pairs improving              | **4 / 4**   |
| Median paired throughput improvement | **114.11%** |
| Throughput pairs improving           | **4 / 4**   |
| Valid formal slots                   | **16 / 16** |

Two host-activity-contaminated attempts are retained in the raw evidence together with their same-slot replacements. They are excluded by the predeclared environment guard and are not needed to establish the TensorRT result.

Run the benchmark:

```bash
python benchmarks/run_benchmark.py run --env-file .env.example
```

Validate committed evidence without rerunning inference:

```bash
python scripts/validate_benchmark_evidence.py --check
```

Validate only the immutable historical run:

```bash
python scripts/validate_benchmark_evidence.py --historical-only
```

The compatibility gate is semantic rather than full-tree byte equality. It derives Perf Analyzer command behavior, aggregation behavior, environment-guard classification, replacement behavior, the model pair, methodology, and the benchmark-relevant deployment projection from production code.

See `docs/benchmarking.md` and `benchmarks/report.md`.

## 📊 Monitoring

The stack contains:

- Prometheus;
- Grafana;
- NVIDIA DCGM Exporter;
- Triton native metrics.

Default local endpoints:

| Service        | Address                         |
| -------------- | ------------------------------- |
| Triton HTTP    | `http://127.0.0.1:8000`         |
| Triton gRPC    | `127.0.0.1:8001`                |
| Triton metrics | `http://127.0.0.1:8002/metrics` |
| Prometheus     | `http://127.0.0.1:9090`         |
| Grafana        | `http://127.0.0.1:3000`         |
| DCGM metrics   | `http://127.0.0.1:9400/metrics` |

The provisioned Grafana dashboard UID is:

```text
ml-dev-ops-inference
```

Dashboard:

```text
http://127.0.0.1:3000/d/ml-dev-ops-inference/ml-dev-ops-inference
```

Its five main panels are:

1. inference throughput;
2. request rate;
3. average request latency;
4. GPU utilization;
5. failed requests.

Prometheus also loads two rules:

- `HighInferenceLatency`;
- `InferenceRequestFailures`.

Verify the complete Triton -> Prometheus -> Grafana path:

```bash
python monitoring/verify_runtime.py --env-file .env.example
```

The verifier generates a short controlled inference workload, waits across at least two Prometheus scrape intervals, queries the dashboard expressions through Grafana's Prometheus datasource proxy, checks GPU identity, verifies the alert definitions, and restores the original READY set.

See `docs/monitoring.md`.

## 🖼️ Python client and samples

The client accepts individual images or directories.

Classification:

```bash
python client/inference_client.py classify client/samples/
```

Explicit ONNX version:

```bash
python client/inference_client.py classify client/samples/01_dog.jpg \
    --model resnet50_onnx \
    --version 2
```

TensorRT over gRPC:

```bash
python client/inference_client.py classify client/samples/ \
    --model resnet50_tensorrt \
    --protocol grpc \
    --batch-size 4
```

YOLO over HTTP:

```bash
python client/inference_client.py detect client/samples/ \
    --protocol http \
    --batch-size 2
```

The repository contains ten tracked JPG sample images under `client/samples/`. Their source, license/provenance, dimensions, and SHA-256 values are recorded in `client/samples/manifest.json`.

The runtime client evidence covers all ten samples and both serving protocols.

See `docs/client.md`.

## 🧾 Logging and CSV export

Each Triton request appends one structured JSONL event.

Operational logs are written outside committed evidence and are ignored by Git.

Export a JSONL history to CSV:

```bash
python client/inference_client.py export-logs \
    --input-log logs/inference.jsonl \
    --output-csv logs/inference.csv
```

Committed Step 5 evidence contains:

```text
docs/evidence/step-5/inference-log.jsonl
docs/evidence/step-5/inference-log.csv
docs/evidence/step-5/predictions.txt
docs/evidence/step-5/client-runtime.json
```

The committed reference run records successful classification/detection requests without raw tensors, secrets, or host-specific paths.

## 🔎 Runtime evidence

README claims are not treated as runtime proof. The repository keeps machine-checkable evidence for the main live stages.

| Step | Evidence                | What it proves                                                                       |
| ---- | ----------------------- | ------------------------------------------------------------------------------------ |
| 2    | `docs/evidence/step-2/` | Docker stack, GPU visibility, service health, Prometheus targets, Grafana datasource |
| 3    | `docs/evidence/step-3/` | prepared model contracts and runtime model smoke                                     |
| 4    | `docs/evidence/step-4/` | HTTP/gRPC serving, batching, versions, lifecycle, cleanup                            |
| 5    | `docs/evidence/step-5/` | real-image client, predictions, JSONL/CSV logging, READY restoration                 |
| 6    | `docs/evidence/step-6/` + `benchmarks/results/` | ONNX vs TensorRT benchmark and raw measurement evidence      |
| 7    | `docs/evidence/step-7/` | Prometheus/Grafana/DCGM data path and loaded alert rules                             |
| GPU portability | `docs/evidence/portability/` | selected-GPU TensorRT build provenance, parity-gated manifest, and current serving proof |

Step 2 and Step 6 separate:

```text
historical integrity
current semantic compatibility
```

A historical runtime snapshot is not rewritten merely because unrelated repository files change.

The complete requirement-to-evidence matrix is in:

```text
docs/audit-evidence.md
```

## 🧰 Technology stack

| Area                  | Technology                              |
| --------------------- | --------------------------------------- |
| Inference server      | NVIDIA Triton Inference Server `2.71.0` |
| Container runtime     | Docker + Docker Compose                 |
| Classification        | ResNet50                                |
| Detection             | YOLO11n                                 |
| Baseline runtime      | ONNX Runtime                            |
| Optimized runtime     | TensorRT                                |
| Client                | Python                                  |
| Protocols             | Triton HTTP and gRPC                    |
| Metrics               | Triton Prometheus metrics               |
| GPU telemetry         | NVIDIA DCGM Exporter                    |
| Metrics storage/query | Prometheus                              |
| Dashboard             | Grafana                                 |
| Benchmark tool        | Triton Perf Analyzer                    |
| Contracts             | JSON / JSON Schema / YAML               |
| Tests                 | Python `unittest`                       |

Pinned container versions and model/export dependencies are stored in repository configuration rather than floating `latest` tags.

## 🧪 Tests and checks

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

With GNU Make:

```bash
make validate
```

Direct validation:

```bash
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

The validation layer covers:

- repository structure;
- module and dependency metadata;
- Docker configuration;
- runtime-evidence integrity;
- model repository structure;
- model and serving evidence;
- client contracts and evidence;
- benchmark arithmetic and raw-data recomputation;
- behavioral compatibility probes;
- monitoring configuration and evidence;
- deterministic generation;
- read-only check modes;
- tracked-file hygiene;
- host-path and secret-like evidence leakage.

`promtool` is used when available. Its absence does not replace the repository's mandatory YAML and semantic validation.

## 📁 Project structure

```text
ml-dev-ops/
├── benchmarks/
│   ├── configs/
│   ├── results/
│   ├── aggregate_results.py
│   ├── environment_guard.py
│   ├── report.md
│   └── run_benchmark.py
├── client/
│   ├── logging/
│   ├── samples/
│   ├── inference_client.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── transport.py
├── deployment/
│   ├── docker/
│   ├── scripts/
│   ├── triton/
│   └── runtime_evidence.py
├── docs/
│   ├── evidence/
│   ├── generated/
│   ├── audit-evidence.md
│   ├── benchmarking.md
│   ├── client.md
│   ├── deployment.md
│   ├── model-preparation.md
│   ├── model-versioning.md
│   └── monitoring.md
├── models/
│   ├── resnet50_onnx/
│   ├── resnet50_tensorrt/
│   ├── yolo11n_onnx/
│   ├── model-manifest.json
│   └── model-spec.yaml
├── monitoring/
│   ├── grafana/
│   ├── prometheus/
│   └── verify_runtime.py
├── schemas/
├── scripts/
├── shared/
├── tests/
├── ARCHITECTURE.md
├── dependency-graph.json
├── docker-compose.yml
├── Makefile
├── module-map.json
├── README.md
└── README_RU.md
```

`module-map.json` documents module ownership. `dependency-graph.json` defines allowed and forbidden dependencies, and `docs/generated/dependency-graph.md` is generated from it.

## ⚠️ Notes

- Model binaries are intentionally ignored by Git and must be prepared locally.
- TensorRT engines are hardware-specific; the portable workflow rebuilds `model.plan` on the selected GPU while the build record and manifest capture host provenance.
- The reference runtime evidence was produced on an NVIDIA GeForce RTX 4080 Laptop GPU with compute capability `8.9`.
- Runtime evidence proves the recorded reference runs; it does not claim every future host will reproduce identical performance numbers.
- Perf Analyzer's internal stability text is diagnostic and is not the benchmark acceptance criterion.
- GPU utilization may legitimately be `0%`; monitoring validity requires a numeric series with the correct GPU identity, not a forced non-zero value.
- Grafana and Prometheus persistent data live in Docker volumes and are not committed.
- `.env`, local cache directories, model binaries, operational logs, and runtime junk are ignored by Git.
- The repository is licensed under `AGPL-3.0-only` because the model-preparation workflow uses the open-source Ultralytics YOLO toolchain. Third-party components retain their own licenses; see `THIRD_PARTY_NOTICES.md`.

## 🧑‍💻 Author

Nazar Yestayev (@nyestaye)
