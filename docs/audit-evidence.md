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
| ONNX and TensorRT benchmark | Planned for step 6 | Functional parity is implemented, performance benchmark pending | Pending |
| Full Prometheus and Grafana dashboard | Planned for step 7 | Not implemented | Pending |
| Inference logs and CSV | Planned | Not implemented | Pending |
| HTTP and gRPC serving | SDK protocol matrix for every model/version | `docs/evidence/step-4/serving-runtime.json` | Metadata, binary inference, finite outputs, and numerical protocol parity passed |
| Dynamic batching | Spec-owned schedulers, bounded attempt history, and per-version statistics deltas | `docs/evidence/step-4/serving-runtime.json` | `attempts_used` matches 1–3 recorded attempts; inference deltas exceed execution deltas and batch sizes above one are observed |
| Model version management | ResNet ONNX v1/v2, load overrides, tracked policy, and in-place reload | `docs/evidence/step-4/repository-versions.txt` | Versions 1 and 2 READY together; default selects v2; cleanup repository empty and every model/version readiness endpoint false |

README statements alone are not accepted as runtime evidence.

The recorded runtime evidence was produced on Windows 11 with Docker Desktop/WSL2,
NVIDIA driver 610.88, and compute capability 8.9. It is evidence for that reference
run, not a substitute for rerunning the smoke test after environment changes.
Validate the committed evidence with `python scripts/validate_runtime_evidence.py`
`python scripts/validate_model_evidence.py`, and
`python scripts/validate_serving_evidence.py`.
