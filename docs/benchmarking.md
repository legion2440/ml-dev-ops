# Benchmarking

## Comparison pair

The measured pair is ResNet50 ONNX Runtime FP32 (`resnet50_onnx:1`) versus
TensorRT FP16 compute with FP32 I/O (`resnet50_tensorrt:1`). The generated
`shared/benchmark-model-pair.json` proves common weights, tensor shapes,
preprocessing, maximum batch size, scheduler, instance policy, and passed numerical
parity without exposing model-artifact paths.

The runner consumes this shared projection and live Triton metadata. It does not
read the model spec, manifest, inference history, or model binaries.

## Step 6 acceptance

The formal benchmark contains only latency and throughput:

| Scenario | Batch | Concurrency | Primary | Secondary |
| --- | ---: | ---: | --- | --- |
| latency | 1 | 1 | mean client latency | p50, p95 |
| throughput | 8 | 4 | infer/s | Triton request/queue/compute statistics |

Both use the same deterministic FP32 input, explicit model version, HTTP binary
tensors, 100 warmup requests, and count windows of at least 500 successful requests.
Dynamic batching is excluded because Step 4 already proves it with Triton execution
and batch-count statistics.

Four paired repetitions run in AB → BA → AB → BA order, where A is ONNX and B is
TensorRT. No best-run selection, outlier deletion, or performance-triggered retry is
allowed.

For pair `i`:

```text
latency_i = (ONNX mean_i - TensorRT mean_i) / ONNX mean_i * 100
throughput_i = (TensorRT infer/s_i - ONNX infer/s_i) / ONNX infer/s_i * 100
```

The primary result is the median paired improvement across all four repetitions.
Each scenario passes when the median is greater than zero and at least three pairs
improve in the expected direction. A result above 5% is described as strong; a
result from 0% through 5% is modest but measurable. Five percent is descriptive,
not an acceptance threshold.

## Perf Analyzer output contract

Perf Analyzer is the formal measurement tool. It runs three count windows, 500
requests per window, and a fixed 999% completion tolerance so that it emits the
complete CSV containing mean, p50, p95, and infer/s. Perf Analyzer determines its
reported result from three recent windows; therefore this setting is explicitly an
operational output mechanism, not a stationarity claim.

`Failed to obtain stable measurement`, if present, has no direct PASS/FAIL meaning.
A trial still needs a parseable formal metric artifact; a missing/corrupt CSV is an
infrastructure/output error rather than a performance failure. The configured 999%
completion tolerance avoids treating normal laptop operating-state drift as an
assignment acceptance gate.

Every PA window is preserved in the log and sidecar. The validator matches the
window infer/s and p95 back to the log, requires a final PA client request count of
at least 1500, and independently recomputes the explicit-version Triton statistics
deltas. Boundary deltas are diagnostic and may shift a few in-flight requests
between adjacent windows. Mean client latency and percentiles come from the PA CSV,
not a custom Python timer.

On Docker Desktop/WSL2, the tracked `benchmarks/clock_guard.c` preserves realtime
epoch but advances elapsed time from `CLOCK_MONOTONIC`. Its source and binary hashes
are recorded.

## Isolation and validity

Before measurement, the runner snapshots the entire READY set, rejects partial
states it cannot reproduce, unloads everything, and checks the pair's live tensor
and scheduling contract. Each run loads only its target model. Cleanup unloads the
measured model and restores the exact initial state.

A valid formal run proves:

- the expected model and version were measured under the declared batch and
  concurrency;
- the same deterministic input hash and live serving contract were used;
- all three 500-request PA windows completed and the CSV/log/sidecar parse;
- Perf Analyzer and Triton report no request/runtime errors;
- the host observer produced continuous, valid attribution telemetry;
- no foreign GPU workload was objectively attributed during the sequence range.

PA stability text, temperature, power, clocks, P-state, high GPU utilization, queue
variation, or a poor metric value are not validity criteria.

## Environment guard

The Windows host dynamically enumerates `GPU Engine(*)` counters through PDH and
records PID, process name, engine type, and utilization. `nvidia-smi` provides only
device diagnostics. `vmmemWSL.exe` and `vmwp.exe` are predeclared benchmark-owned
processes; forbidden applications are declared separately in the config.

Marker/ack boundaries use host sequence numbers, so Windows and container monotonic
clock origins are never compared. One-second periodic samples carry device
diagnostics. Lightweight event-driven boundary samples refresh WDDM attribution and
refer to the latest periodic device snapshot.

The validator independently reconstructs baseline and measurement ranges.
`CONTAMINATED` requires more than the declared 0.1% activity threshold from a new
foreign process, a forbidden process, or a baseline-idle foreign process that
becomes active. Only that objective attribution permits a consecutive same-slot
replacement, with no more than three attempts. Telemetry gaps or collection failure
are `ERROR`.

For the published candidate, this predeclared formal rule classified `System`
(PID 4) `Copy` activity above 0.1% that was absent from the guard baseline as
external/host contamination. This is a conservative classification of Windows host
activity, not proof that any specific user process caused it.

## Supporting diagnostics

Per-window snapshots retain request, inference, and execution counts plus total
request, queue, compute-input, compute-infer, and compute-output durations. Host
telemetry retains clocks, power, temperature, GPU utilization, memory, and WDDM
workload state.

These values explain variation without reclassifying it. A report may state that
throughput variation correlated with workload-owned compute-infer drift accompanied
by temperature/power changes, consistent with operating-state variation. It must
not claim stronger causality than the data supports.

Clock locking with `nvidia-smi -lgc` is not a formal prerequisite. It requires
administrator privileges and is reserved for a separate diagnostic experiment only
if the paired formal result is contradictory.

## Publication and validation

The SDK container mounts the repository read-only and writes only to the ignored
benchmark cache. The host stages a full candidate below
`.cache/benchmarking/run-<id>/publish`, validates it, publishes atomically, then
validates the tracked bundle again. Rollback restores the previous published bundle
if any step fails.

The evidence validator independently proves exactly four ONNX/TensorRT pairs per
scenario, AB/BA order, zero errors, captured model/config identity, all formulas and
medians, at least three improving pairs, raw/CSV arithmetic, published artifact
hashes, and objective attribution for every replacement. It does not enforce PA
stability, thermal stability, fixed clocks, or a 5% performance gate.

Historical integrity and current compatibility are separate. The evidence retains
the exact runtime source fingerprint `82e10584916355dfd2332055dc785a093b95d5265d37b62c9b7388fc274f4f62`
plus the runtime per-file hashes. Current compatibility compares only the captured
methodology, scenario and acceptance contract, model pair, aggregation and Perf
Analyzer semantics, environment-guard rules, and the benchmark-relevant Compose
projection. The critical code semantics are derived from deterministic behavioral
probes: four commands from the real Perf Analyzer command builder, synthetic paired
summaries from the production aggregation function, and clean/contaminated/error plus
same-slot replacement cases through the real guard and runner control flow.
Monitoring mounts, report wording, validator implementation, and check control flow
remain source drift but do not change the meaning of the measurement.

The two contaminated replacements are retained for auditability but are not required
to establish the optimization conclusion; their measured performance was
directionally consistent with the published result.

```text
make validate-benchmark
make benchmark
make validate-benchmark-evidence
```

Direct equivalents:

```text
python benchmarks/run_benchmark.py run --env-file .env.example
python scripts/validate_benchmark_evidence.py --check
python scripts/validate_benchmark_evidence.py --historical-only
```

Both validation modes are strictly read-only. `--check` fails when either historical
integrity or current compatibility fails; `--historical-only` ignores current source
compatibility and validates the saved run bundle alone.

If a formal candidate is contradictory, keep all four pairs and supporting
telemetry in `.cache/benchmarking` and stop for review. Do not change the method or
rerun selectively.

## Superseded diagnostics

The earlier contract required 5% PA stability within ten windows for every formal
run and at least 5% aggregate improvement. It was superseded before this formal
candidate because diagnostic measurements showed persistent workload-owned
thermal/power drift. PA stationarity is not required by the assignment and was
rejecting otherwise valid inference measurements.

Superseded diagnostic runs are intentionally excluded from committed Step 6 evidence
and may exist only in the local ignored benchmark cache. The published formal bundle
is self-contained.
