# Monitoring and alerting

## Implemented scope

Step 7 completes the Prometheus/Grafana requirement using the four existing
services. It does not add Alertmanager, Loki, OpenTelemetry, another exporter, a
custom telemetry pipeline, or browser automation.

- Triton publishes inference counters at `triton:8002`.
- DCGM Exporter publishes GPU metrics at `dcgm-exporter:9400`.
- Prometheus scrapes both targets every 15 seconds.
- Grafana uses the provisioned Prometheus datasource UID `prometheus` and the
  existing file provider at `/var/lib/grafana/dashboards`.
- The `ML DevOps Inference` dashboard has UID `ml-dev-ops-inference`.
- Prometheus loads exactly two project alert rules.

## Dashboard metrics

The five panels use these expressions:

| Panel | PromQL | Unit |
| --- | --- | --- |
| Inference Throughput | `sum by (model, version) (rate(nv_inference_count[1m]))` | infer/s |
| Request Rate | `sum by (model, version) (rate(nv_inference_request_success[1m]))` | req/s |
| Average Request Latency | `sum by (model, version) (rate(nv_inference_request_duration_us[1m])) / sum by (model, version) (rate(nv_inference_request_success[1m])) / 1000` | ms |
| GPU Utilization | `max by (UUID, gpu, modelName, pci_bus_id) (DCGM_FI_DEV_GPU_UTIL)` | % |
| Failed Requests | `sum by (model, version) (rate(nv_inference_request_failure[1m]))` | failed req/s |

Triton separates successful request count from inference count. Throughput therefore
uses `nv_inference_count`, which preserves batch semantics. Average request latency
uses successful requests as its denominator and intentionally does not use
`clamp_min(..., 1)`: when no requests exist, no latency value is preferable to a
distorted value.

The pinned DCGM Exporter on the reference host exposes `DCGM_FI_DEV_GPU_UTIL` as a
0…100 percentage gauge with GPU UUID, index, model name, and PCI bus ID. A numeric
series at `0%` is valid monitoring data. A positive `max_over_time` value after the
controlled workload is recorded only as a diagnostic observation, never as a PASS
gate.

## Alert rules

`monitoring/prometheus/alerts.yml` contains:

- `HighInferenceLatency`: average request latency above the project-defined 100 ms
  threshold for two minutes, guarded by positive request traffic;
- `InferenceRequestFailures`: at least one increase in
  `nv_inference_request_failure` during five minutes.

The latency threshold is a project operational default, not an assignment
requirement. Step 7 proves that Prometheus loads both rules and exposes their state.
It does not force either alert to fire and does not configure notification delivery.

## Runtime verification

Start the stack and run:

```text
make up
make verify-monitoring
```

The verifier:

1. resolves the ports actually published by the running Compose project;
2. checks Prometheus, Grafana, the datasource, Triton/DCGM targets, the dashboard,
   and both alert definitions;
3. snapshots the exact Triton READY set;
4. calls the existing `client/inference_client.py classify` path for at least two
   15-second scrape intervals and writes only `.cache/monitoring/inference-log.jsonl`;
5. executes all five dashboard expressions through Grafana's datasource proxy;
6. matches the DCGM series to the host `nvidia-smi` UUID, model, and PCI bus ID;
7. restores and verifies the exact initial READY set in `finally`;
8. atomically writes two compact evidence files only after PASS.

The verifier does not call `verify-client`, modify Step 5 evidence, run a benchmark,
or require statistical stability. The reference run used a short workload of a
little over 35 seconds; this is an observability check, not a performance test.

The dashboard is available at:

```text
http://127.0.0.1:3000/d/ml-dev-ops-inference/ml-dev-ops-inference
```

## Evidence and offline validation

Committed evidence is limited to:

- `docs/evidence/step-7/monitoring-runtime.json`: health, targets, dashboard, alerts,
  GPU identity, workload facts, READY restoration, and artifact hashes;
- `docs/evidence/step-7/prometheus-queries.json`: the five Grafana-proxied query
  results plus the non-gating GPU `max_over_time` observation.

Prometheus TSDB data, Grafana's database, screenshots, and temporary inference logs
are not committed. Validate the configuration and tracked evidence without a live
daemon:

```text
python scripts/validate_monitoring.py
```

Use `--config-only` while authoring configuration before runtime evidence exists.
