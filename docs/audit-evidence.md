# Audit evidence

This matrix maps audit requirements to concrete implementation and evidence. It
will be expanded as each scope is completed.

| Audit item | Implementation | File or command | Evidence |
| --- | --- | --- | --- |
| Repository structure | Step 1 scaffold and required directories | `python scripts/validate_structure.py` | Successful validation output |
| Module navigation | Machine-readable module map and schema | `python scripts/validate_module_map.py` | Successful validation output |
| Dependency boundaries | Allowed and forbidden graph with generated Mermaid | `python scripts/generate_dependency_graph.py --check` | Current generated graph |
| Pinned Docker topology | Four services, pinned images, GPU policy, mounts, ports, healthchecks | `python scripts/validate_deployment.py` | Successful static validation |
| Dockerfile | Minimal pinned Triton wrapper | `deployment/docker/Dockerfile` | Static validation |
| Compose configuration | Canonical `.env.example` interpolation | `make compose-config` | Successful config output |
| Lifecycle commands | Shared Compose helper and start/stop/status wrappers | `deployment/scripts` | Four services running and healthy on 2026-07-31 |
| Triton infrastructure | Explicit control, disabled config auto-completion, and read-only repository | `deployment/scripts/run_triton.sh` | Live, ready, metrics, and no models loaded before model smoke |
| Minimum observability | Prometheus targets and Grafana datasource provisioning | `monitoring` | Triton/DCGM targets up and datasource provisioned |
| GPU telemetry | Pinned paired DCGM Exporter service | `docker-compose.yml` | Real `DCGM_` metrics; RTX 4080 Laptop visible in Triton |
| Infrastructure smoke | Host-side health, targets, datasource, and empty-repository checks | `docs/evidence/step-2/smoke.json` | All 10 checks passed on 2026-07-31 |
| Runtime service state | Sanitized service, image, health, and port snapshot | `docs/evidence/step-2/compose-ps.txt` | Four running and healthy services; loopback-only ports |
| Runtime environment | Sanitized Docker, Triton image digest, and GPU facts | `docs/evidence/step-2/environment.txt` | Machine-checked reference environment |
| Model source provenance | Accepted URLs, SHA-256, licenses, package lock, and image digests | `models/model-spec.yaml` | Step 3 evidence references its immutable manifest v1 snapshot |
| Two CV workloads | ResNet50 classification and YOLO11n detection | `models/model-manifest.json` | Two ONNX graphs pass checker and synthetic ONNX Runtime inference |
| TensorRT optimization | Strongly typed FP16 ResNet with FP32 I/O; spec-derived capability mapping with no default plan fallback | `models/resnet50_tensorrt/config.pbtxt` and unit regression | Engine deserialization, exporter-level parity, and live explicit load passed |
| Triton model repository | Three generated configs, model-local labels, normalized versions, and explicit policies | `python scripts/validate_model_repository.py --structure-only` | Manifest v2 and generated text are current; binaries are not tracked |
| Runtime model serving | Explicit load, metadata/config, batch inference, parity, and unload | `docs/evidence/step-3/triton-model-smoke.json` | All three models runtime-verified on 2026-07-31 |
| Runtime repository state | Three version-1 models ready during verification | `docs/evidence/step-3/model-repository.txt` | Sanitized repository index captured before explicit unload |
| ONNX and TensorRT benchmark | Four paired AB/BA repetitions for latency and throughput using pinned Perf Analyzer | `benchmarks/report.md`, `benchmarks/results`, and `docs/evidence/step-6/benchmark-runtime.json` | 16/16 valid slots; latency median improvement 19.32%, throughput 114.11%, both 4/4 directional; two attributed contaminated attempts retained with same-slot replacements |
| Full Prometheus and Grafana dashboard | Planned for step 7 | Not implemented | Pending |
| Production image client | Contract-driven ResNet/YOLO preprocessing and postprocessing, bounded batches, auto-load, explicit versions | `client/inference_client.py` | `docs/evidence/step-5/predictions.txt` records real-image HTTP, gRPC, ONNX v1/v2, and TensorRT results |
| Sample image provenance | Ten real CC0/public-domain-marked JPG images with source, attribution, hash, and dimensions | `client/samples/manifest.json` | `python scripts/validate_client.py` decodes and verifies the complete inventory |
| Inference logs and CSV | One schema-validated JSONL event per request plus deterministic CSV export | `docs/evidence/step-5/inference-log.jsonl` and `inference-log.csv` | 11 successful events/rows; final predictions retained without raw tensors or host paths |
| Client runtime matrix | Full ten-image ResNet batch-4 and YOLO batch-2 runs plus explicit protocol/version/TensorRT cases | `docs/evidence/step-5/client-runtime.json` | `http`, `grpc`, and lowercase `tensorrt` classification cases passed; 55 detections; source fingerprints current; initial READY set restored exactly |
| HTTP and gRPC serving | SDK protocol matrix for every model/version | `docs/evidence/step-4/serving-runtime.json` | Metadata, binary inference, finite outputs, and numerical protocol parity passed |
| Dynamic batching | Spec-owned schedulers, bounded attempt history, and per-version statistics deltas | `docs/evidence/step-4/serving-runtime.json` | `attempts_used` matches 1–3 recorded attempts; inference deltas exceed execution deltas and batch sizes above one are observed |
| Model version management | ResNet ONNX v1/v2, load overrides, tracked policy, and in-place reload | `docs/evidence/step-4/repository-versions.txt` | Versions 1 and 2 READY together; default selects v2; cleanup repository empty and every model/version readiness endpoint false |

README statements alone are not accepted as runtime evidence.

The recorded runtime evidence was produced on Windows 11 with Docker Desktop/WSL2,
NVIDIA driver 610.88, and compute capability 8.9. Steps 2–4 were captured on
2026-07-31, step 5 on 2026-08-01, and step 6 on 2026-08-02. These files prove the reference runs, not a
substitute for rerunning verification after environment changes.
Validate the committed evidence with `python scripts/validate_runtime_evidence.py`
`python scripts/validate_model_evidence.py`, and
`python scripts/validate_serving_evidence.py`, and
`python scripts/validate_client_evidence.py`, and
`python scripts/validate_benchmark_evidence.py`.
