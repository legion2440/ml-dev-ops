# Architecture

## Status

Architecture contracts, the step 2 Docker infrastructure, and the step 3 model
repository are implemented. Structure validation is daemon-free; full artifact
validation is Triton-free but uses pinned Docker images and the target GPU. The
four-service infrastructure and three-model runtime smoke passed on the reference
host on 2026-07-31.

Version-1 ResNet50 ONNX FP32, ResNet50 TensorRT FP16, and YOLO11n ONNX FP32 are
runtime-verified. Dynamic request batching, additional model versions, the production
client, dashboards, alerts, logging, and benchmarks remain planned.

## System context

The target system serves two computer-vision workloads through NVIDIA Triton:
ImageNet classification and COCO object detection. A Python client owns
model-specific preprocessing and postprocessing. Triton owns inference scheduling,
dynamic batching, version selection, and GPU execution.

```mermaid
flowchart LR
    Sample["Sample image"] --> Client["Python inference client"]
    Client -->|"REST or gRPC"| Triton["NVIDIA Triton"]
    Triton --> Repository["Triton model repository"]
    Repository --> ONNX["ONNX models"]
    Repository --> TensorRT["TensorRT engines"]
    Client --> History["JSONL inference history"]
    History --> CSV["CSV export"]
    Triton --> Metrics["Triton metrics"]
    GPU["GPU metrics"] --> MetricsStore["Prometheus"]
    Metrics --> MetricsStore
    MetricsStore --> Grafana["Grafana"]
```

## Components

### Model preparation

- Responsibility: download pinned pretrained weights, export ONNX, build TensorRT
  engines on the target GPU, validate models, and populate the model repository.
- Inputs: preparation configuration, upstream weights, export parameters, shared
  tensor contracts, and target GPU capabilities.
- Outputs: validated ONNX files, TensorRT plans, model metadata, and checksums.
- Public entrypoint: `scripts/model_preparation/prepare_models.py` (`implemented`).
- Root: `scripts/model_preparation`.
- Tests: spec and manifest staleness, layout, tensor contracts, ONNX validation,
  TensorRT deserialization, and exporter-level parity.

### Model repository

- Responsibility: own Triton `config.pbtxt` files, version directories, and generated
  model artifacts.
- Inputs: artifacts produced by model preparation.
- Outputs: a Triton-compatible repository mounted read-only by the serving runtime.
- Public entrypoint: none; this is an artifact module.
- Root: `models`.
- Tests: repository layout, model metadata, version policy, and artifact presence.

### Triton serving

- Responsibility: expose health, repository control, metadata, REST inference, gRPC
  inference, metrics, batching, GPU instances, and version policy.
- Inputs: the model repository and shared inference contracts.
- Outputs: predictions, service health, model metadata, and Prometheus metrics.
- Public entrypoint: `deployment/scripts/run_triton.sh` (`implemented`).
- Root: `deployment/triton`.
- Tests: step 2 smoke covers health and metrics. Step 3 smoke explicitly loads,
  infers with, compares, and unloads all three models. Version switching remains planned.

### Deployment

- Responsibility: build, start, stop, connect, and health-check containers while
  preserving declared volumes.
- Inputs: Compose configuration, environment variables, runtime images, and service
  configuration.
- Outputs: a reproducible container network containing serving and observability
  services.
- Public entrypoint: `deployment/scripts/run_environment.sh` (`implemented`).
- Root: `deployment`.
- Tests: `scripts/validate_deployment.py` provides static validation and
  `deployment/scripts/smoke_environment.py` provides runtime verification.

### Inference client

- Responsibility: preprocess images, build Triton payloads, use REST or gRPC, parse
  predictions, render classifications or detections, and handle failures.
- Inputs: one image or a directory, model selection, version, protocol, and timeout.
- Outputs: human-readable predictions and structured inference events.
- Public entrypoint: `client/inference_client.py` (`planned`).
- Root: `client`.
- Tests: preprocessing, payload generation, transport boundaries, prediction
  parsing, batching, timeouts, and model-version selection.

### Inference logging

- Responsibility: append one JSONL event per request and export history to CSV.
- Inputs: request identity, timing, model metadata, status, prediction summary, and
  error text.
- Outputs: `logs/inference.jsonl` and `logs/inference.csv`.
- Public entrypoint: `client/logging/writer.py` (`planned`).
- Root: `client/logging`.
- Tests: schema stability, append behavior, error records, and CSV export.

### Benchmarking

- Responsibility: execute controlled load profiles and compare baseline ONNX with
  optimized TensorRT serving.
- Inputs: common images, batch sizes, concurrency, warmup, request count, and
  environment metadata.
- Outputs: raw measurements, aggregate CSV files, and a comparison report.
- Public entrypoint: `benchmarks/run_benchmark.py` (`planned`).
- Root: `benchmarks`.
- Tests: aggregation, percent-change calculations, errors, and report generation.

### Observability

- Responsibility: collect Triton, GPU, and availability metrics; provision dashboards;
  and evaluate alert rules.
- Inputs: Triton metrics, DCGM metrics, and container health.
- Outputs: step 2 provides Prometheus targets and a provisioned Grafana datasource;
  dashboards and alert states remain planned.
- Public entrypoint: `monitoring/prometheus/prometheus.yml` (`implemented`).
- Root: `monitoring`.
- Tests: configuration validation and runtime target discovery are implemented.
  Dashboard JSON and alert-rule tests remain planned.

### Shared contracts

- Responsibility: define only schemas and data-transfer objects shared across module
  boundaries.
- Inputs: stable cross-module data requirements.
- Outputs: tensor metadata, request identifiers, prediction summaries, and log event
  contracts.
- Public entrypoint: `shared/contracts.py` (`planned`).
- Root: `shared`.
- Tests: serialization and backward-compatible schema behavior.

## Boundary rules

- Dependency arrows point from a consumer to the module it may use.
- `dependency-graph.json` is the only editable source for allowed and forbidden
  dependencies.
- The client may use shared contracts, Triton APIs, and inference logging.
- Benchmarking may use the public client transport but not client internals.
- Observability consumes metrics and health endpoints, not Python client code.
- Deployment controls containers but contains no model preprocessing.
- Triton consumes the model repository but does not build model artifacts at runtime.
- Shared contracts depend on no higher-level feature.

The complete generated graph is in
`docs/generated/dependency-graph.md`.

## Deployment topology

```mermaid
flowchart TB
    Host["GPU host with NVIDIA runtime"]
    Loopback["Loopback-only published ports"]
    Models["Read-only models bind mount"]
    PromData["Prometheus named volume"]
    GrafanaData["Grafana named volume"]
    subgraph Compose["Project-scoped Compose network"]
        Triton["Triton container"]
        Prometheus["Prometheus container"]
        Grafana["Grafana container"]
        DCGM["DCGM exporter container"]
    end
    Host --> Compose
    Models --> Triton
    Triton --> Prometheus
    DCGM --> Prometheus
    Prometheus --> Grafana
    PromData --> Prometheus
    GrafanaData --> Grafana
    Compose --> Loopback
```

`docker-compose.yml` is the only owner of Triton server arguments. Triton uses
explicit model control, disables backend config auto-completion, and remains PID 1
in a minimal image wrapper. Prometheus and
Grafana data use named volumes; repository configuration uses read-only bind
mounts. Services communicate through Compose DNS names and fixed container ports.

`.env.example` owns pinned image references, host ports, and local defaults.
Lifecycle scripts share the same Compose command and pass either `.env` or
`.env.example` explicitly without executing it.

## Inference sequence

```mermaid
sequenceDiagram
    participant User
    participant Client
    participant Log as Inference logging
    participant Triton
    User->>Client: image, model, version, protocol
    Client->>Client: preprocess and form batch
    Client->>Triton: inference request
    Triton-->>Client: tensors or error
    Client->>Client: postprocess prediction
    Client->>Log: append structured event
    Client-->>User: prediction and latency
```

## Model preparation flow

```mermaid
flowchart LR
    Config["Model spec and hashed package lock"] --> Weights["Verify source SHA-256"]
    Weights --> Export["Export ONNX"]
    Export --> Validate["ONNX Checker and Runtime"]
    Validate --> Cast["Strongly typed FP16 intermediate"]
    Cast --> Optimize["Build CC 8.9 TensorRT plan"]
    Validate --> Repository["Populate versioned repository"]
    Optimize --> Repository
    Repository --> Verify["Explicit load, inference, parity, unload"]
```

TensorRT plans are target-dependent build artifacts and are not assumed to be
portable across arbitrary GPU and runtime combinations.

## Deferred decisions

The YOLO release, source hashes, local binary policy, and CC 8.9 TensorRT profile are
fixed for step 3. CI provider, cross-host artifact distribution, additional model
versions, and benchmark strategy remain later-scope decisions.
