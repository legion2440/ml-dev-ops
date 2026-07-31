# Monitoring and alerting

## Current status

Prometheus, Grafana, DCGM metrics, dashboards, and alerts are planned.

## Metrics contract

Prometheus will collect Triton request, failure, inference, queue, compute, and
throughput metrics; GPU utilization and memory; and container availability.

Grafana will show service and model state, request rate, latency percentiles,
throughput, failures, GPU behavior, and model or version comparisons.

Planned alert categories are unavailable Triton or models, high latency, increasing
failed requests, high GPU utilization or memory, and missing metrics.

Provisioning must be automatic when the deployment is implemented.
