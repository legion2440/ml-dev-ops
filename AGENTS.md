# Repository instructions

This repository is organized around explicit feature boundaries. Work within the
smallest relevant module and treat architecture metadata as part of the change.

## Navigation order

1. Read `module-map.json`.
2. Find the module that owns the requested feature.
3. Read the matching section in `ARCHITECTURE.md`.
4. Open the module's public entrypoint or interface.
5. Read only the related configuration, implementation, schemas, and tests.
6. Inspect another module only when an allowed dependency makes that necessary.

`dependency-graph.json` is the only editable source of allowed and forbidden
dependencies. `docs/generated/dependency-graph.md` is generated from it and must not
be edited manually.

## Module boundaries

The fixed modules are:

- `model-preparation`
- `model-repository`
- `triton-serving`
- `deployment`
- `inference-client`
- `inference-logging`
- `benchmarking`
- `observability`
- `shared-contracts`

`model-repository` is an artifact module and does not require an executable
entrypoint. `shared-contracts` is reserved for schemas and data-transfer contracts
used by more than one module. Generic helpers do not belong there.

Do not:

- change a neighboring module without an allowed dependency and a task-level need;
- add an architectural dependency without updating `dependency-graph.json`;
- add a package without recording its purpose in dependency metadata;
- hand-edit generated files;
- treat documentation as evidence that a feature works;
- commit secrets, access tokens, local environment files, or large model weights;
- expand the requested scope merely to prepare unrelated future work.

## Path policy

Repository-owned paths in architecture metadata must be repository-relative POSIX
paths. Documentation may show absolute container paths and HTTP API routes when
their context is explicit. Do not use host-specific absolute paths,
parent-directory traversal, or backslashes.

Examples:

- `client/inference_client.py`
- `models/resnet50_onnx/config.pbtxt`
- `tests/unit/client`

## File statuses

Every mapped path has one status:

- `planned`: the path may be absent and must not be presented as implemented;
- `implemented`: the path must exist;
- `generated`: the path must exist and its generator must pass freshness checks.

Every module root must exist regardless of its entrypoint statuses.

## Change workflow

For a feature change:

1. Update implementation and scoped tests together.
2. Update the owning documentation.
3. Update `module-map.json` when paths, entrypoints, interfaces, tests, or artifacts
   change.
4. Update `dependency-graph.json` when architectural dependencies change.
5. Regenerate derived architecture documentation.
6. Run scoped checks first, then repository-wide validation.

New and rewritten text files use LF line endings. `.gitattributes` enforces LF
checkouts for repository text files on every supported host.

## Available commands

The repository currently supports:

```text
make validate
make validate-deployment
make validate-evidence
make architecture
make check-architecture
make compose-config
make up
make down
make status
make smoke
make capture-evidence
make benchmark
make validate-benchmark
make validate-benchmark-evidence
```

The same checks can run without Make:

```text
python scripts/validate_structure.py
python scripts/validate_module_map.py
python scripts/validate_deployment.py
python scripts/validate_runtime_evidence.py
python -m unittest discover -s tests/unit -t . -p "test_*.py"
python scripts/generate_dependency_graph.py
python scripts/generate_dependency_graph.py --check
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
bash deployment/scripts/run_environment.sh
bash deployment/scripts/stop_environment.sh
bash deployment/scripts/check_environment.sh
python deployment/scripts/smoke_environment.py
python deployment/scripts/capture_runtime_evidence.py
python benchmarks/run_benchmark.py run --env-file .env.example
python scripts/validate_benchmark.py
python scripts/validate_benchmark_evidence.py
```

These standard commands are reserved for later scopes and must not be documented as
working until implemented:

```text
make test
make test-feature FEATURE=client
make prepare-models
```

Step 6 benchmarking is launched from the Windows host with `make benchmark`. The
host process owns Windows `GPU Engine(*)` attribution and `nvidia-smi` device
diagnostics, while the SDK container owns Perf Analyzer; their boundary handshake
uses shared-cache sequence acknowledgements.
Every PA pass has a separate marker/ack boundary and measured-model/version Triton
statistics snapshot. Pass decomposition is diagnostic only: it must never remove a
pass, create a replacement, or change the paired acceptance result. Perf Analyzer
stability, thermal/power state, clocks, P-state, and workload-owned utilization are
not validity or PASS criteria. Only attributed foreign GPU activity may replace the
same formal slot.

Deployment has two distinct completion states:

- code-complete: static validation and canonical Compose configuration pass;
- runtime-verified: the GPU services are running and the infrastructure smoke test
  passes.

Do not describe deployment as fully verified without runtime evidence. The current
step intentionally expects an empty Triton model repository.

Step 2 evidence under `docs/evidence/step-2` is a sanitized runtime snapshot. Refresh
it with `capture_runtime_evidence.py` after material image, Compose, driver, Docker,
or GPU changes; never hand-edit it or include credentials, container IDs, hostnames,
or workspace paths.
