# Observability

Step 7 completes the existing observability stack without adding another exporter,
telemetry pipeline, Alertmanager, or screenshot automation.

- Prometheus scrapes Triton and DCGM Exporter every 15 seconds and loads two alert
  rules from `prometheus/alerts.yml`.
- Grafana uses the provisioned datasource UID `prometheus` and the unchanged
  file-provider contract.
- `grafana/dashboards/ml-dev-ops.json` contains exactly five panels for inference
  throughput, request rate, average request latency, GPU utilization, and failures.
- `verify_runtime.py` drives the existing classification client for at least two
  scrape intervals, queries panel expressions through Grafana's datasource proxy,
  records compact evidence, and restores the exact initial READY set.

Run `make verify-monitoring` against the live stack. Validate configuration and the
committed evidence offline with `make validate-monitoring`.
