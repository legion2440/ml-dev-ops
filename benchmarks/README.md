# Benchmarking

This module performs the formal ResNet50 ONNX Runtime FP32 versus TensorRT FP16
comparison with equal FP32 I/O and a generated shared tensor/scheduling contract.
NVIDIA Perf Analyzer runs from the pinned Triton SDK image over HTTP.

Run the workflow with Triton healthy and the model artifacts prepared:

```text
make benchmark
```

## Formal contract

Step 6 has two scenarios and four paired repetitions:

| Scenario | Batch | Concurrency | Primary metric | Secondary metrics |
| --- | ---: | ---: | --- | --- |
| latency | 1 | 1 | mean client latency | p50 and p95 client latency |
| throughput | 8 | 4 | infer/s | server request/queue/compute decomposition |

Each repetition is one ONNX/TensorRT pair. The fixed order is ONNX → TensorRT,
TensorRT → ONNX, ONNX → TensorRT, TensorRT → ONNX. Pairing and alternating order
reduce systematic heating-order bias without claiming that the host is a stationary
laboratory performance system.

Perf Analyzer uses count windows of 500 measured requests, 100 warmup requests,
three windows, deterministic binary input, and a 999% operational completion
tolerance. Perf Analyzer needs three recent windows to emit the complete CSV used
for mean/p50/p95 and throughput. The 999% value is not an acceptance threshold or a
claim of statistical stability. PA status text, clocks, power, temperature, P-state,
and workload-owned utilization do not determine Step 6 validity or PASS.

For every repetition, the runner calculates:

```text
latency improvement % = (ONNX mean latency - TensorRT mean latency) / ONNX mean latency * 100
throughput improvement % = (TensorRT infer/s - ONNX infer/s) / ONNX infer/s * 100
```

The formal result uses the median of the four paired improvements. Each primary
metric passes only when its median is positive and at least three of four pairs
improve in the expected direction. Greater than 5% is labelled a strong measurable
improvement; 0–5% is measurable but modest. Five percent is not a PASS gate.

Dynamic batching is not a Step 6 scenario. Step 4 already records real Triton
execution and batch statistics.

## Validity, isolation, and replacement

A formal run is valid only when the correct model/version is exclusively READY, the
Perf Analyzer command and CSV parse, all three 500-request windows are present and
the final PA client request count is at least 1500, errors are zero, input/config
identities match, and the host telemetry is intact. PA statistical stability is not
a validity condition.

Before measuring, the runner snapshots and unloads the complete READY set. It loads
only the measured model for each run and restores the original READY set in
`finally`.

The Windows host samples `GPU Engine(*)` counters for PID/name/engine attribution.
Only objectively attributed foreign GPU activity may classify an attempt as
`CONTAMINATED` and replace the same formal slot. Replacement numbering is
consecutive and capped at three attempts per slot. Thermal drift, power/clock
changes, high workload-owned utilization, or an unfavorable result never justify a
replacement. Telemetry gaps and infrastructure/runtime corruption are errors.

For the published candidate, this predeclared formal rule classified `System`
(PID 4) `Copy` activity above 0.1% that was absent from the guard baseline as
external/host contamination. This is a conservative classification of Windows host
activity, not proof that any specific user process caused it. The two contaminated
replacements are retained for auditability but are not required to establish the
optimization conclusion; their measured performance was directionally consistent
with the published result.

Every PA window has a marker/ack boundary and explicit-version Triton statistics
snapshots. Queue, compute input/infer/output, total request duration, device clocks,
power, temperature, and WDDM state remain supporting diagnostics. They explain
variation but cannot discard a valid clean measurement.

On Docker Desktop/WSL2, the tracked `clock_guard.c` preserves realtime epoch while
deriving elapsed progress from `CLOCK_MONOTONIC`. Source and compiled hashes are
recorded in the raw bundle.

## Publication

Measurement first writes below `.cache/benchmarking/run-<id>/publish`. A candidate
is published only after both paired gates pass and the daemon-free validator
recomputes the bundle. Failed or contradictory candidates stay in ignored cache and
do not replace the last valid evidence.

Tracked outputs are:

- `benchmarks/results/raw/valid/`: all 16 formal PA CSV/log/sidecar triples;
- `benchmarks/results/raw/contaminated/`: compact proof for any replaced attempts;
- `benchmarks/results/raw/environment-telemetry.jsonl`: process attribution and
  supporting device telemetry;
- `benchmarks/results/baseline.csv` and `optimized.csv`: every scenario/repetition;
- `benchmarks/results/comparison.csv`: all eight pairs and recomputed medians;
- `benchmarks/report.md`: generated human-readable report;
- `docs/evidence/step-6/benchmark-runtime.json`: hashed audit evidence.

Full high-frequency diagnostic streams and all unsuccessful candidates remain in
`.cache/benchmarking`; they are not selected as formal results.

Validate the configuration or published evidence without Triton:

```text
python scripts/validate_benchmark.py
python scripts/validate_benchmark_evidence.py
```

## Superseded diagnostic contract

Before the next formal candidate, the earlier requirement for every run to meet 5%
Perf Analyzer stability within ten windows and for both aggregate improvements to
exceed 5% was superseded. Diagnostic measurements showed persistent
workload-owned thermal/power drift. Perf Analyzer statistical stability is not an
assignment requirement and was producing false failure of otherwise valid inference
measurements.

Superseded diagnostic runs are intentionally excluded from committed Step 6 evidence
and may exist only in the local ignored benchmark cache. The published formal bundle
is self-contained.
