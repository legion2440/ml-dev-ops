# Audit evidence

This matrix maps audit requirements to concrete implementation and evidence. It
will be expanded as each scope is completed.

| Audit item | Implementation | File or command | Evidence |
| --- | --- | --- | --- |
| Repository structure | Step 1 scaffold and required directories | `python scripts/validate_structure.py` | Successful validation output |
| Module navigation | Machine-readable module map and schema | `python scripts/validate_module_map.py` | Successful validation output |
| Dependency boundaries | Allowed and forbidden graph with generated Mermaid | `python scripts/generate_dependency_graph.py --check` | Current generated graph |
| Triton serving | Planned | Not implemented | Pending |
| Two CV models | Planned | Not implemented | Pending |
| ONNX and TensorRT benchmark | Planned | Not implemented | Pending |
| Prometheus and Grafana | Planned | Not implemented | Pending |
| Inference logs and CSV | Planned | Not implemented | Pending |
| Model version management | Planned | Not implemented | Pending |

README statements alone are not accepted as runtime evidence.
