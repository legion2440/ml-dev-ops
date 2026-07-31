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
| Triton infrastructure | Explicit control and empty read-only repository | `deployment/scripts/run_triton.sh` | Live, ready, metrics, and empty repository checks passed |
| Minimum observability | Prometheus targets and Grafana datasource provisioning | `monitoring` | Triton/DCGM targets up and datasource provisioned |
| GPU telemetry | Pinned paired DCGM Exporter service | `docker-compose.yml` | Real `DCGM_` metrics; RTX 4080 Laptop visible in Triton |
| Infrastructure smoke | Host-side health, targets, datasource, and empty-repository checks | `python deployment/scripts/smoke_environment.py --format json` | All 10 checks passed on 2026-07-31 |
| Two CV models | Planned | Not implemented | Pending |
| ONNX and TensorRT benchmark | Planned | Not implemented | Pending |
| Full Prometheus and Grafana dashboard | Planned for step 7 | Not implemented | Pending |
| Inference logs and CSV | Planned | Not implemented | Pending |
| Model version management | Planned | Not implemented | Pending |

README statements alone are not accepted as runtime evidence.

The recorded runtime evidence was produced on Windows 11 with Docker Desktop/WSL2,
NVIDIA driver 610.88, and compute capability 8.9. It is evidence for that reference
run, not a substitute for rerunning the smoke test after environment changes.
