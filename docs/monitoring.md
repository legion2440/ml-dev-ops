# Monitoring and alerting

## Current status

The step 2 observability containers and minimum provisioning are implemented:

- Prometheus scrapes itself, Triton, and DCGM Exporter;
- Grafana receives a Prometheus datasource with UID `prometheus`;
- a dashboard file provider watches `/var/lib/grafana/dashboards`;
- DCGM Exporter publishes GPU metrics on port 9400.

The project dashboard and Prometheus alert rules remain planned for step 7.

The minimum observability runtime was verified on 2026-07-31: both required scrape
targets were up, the Grafana datasource was provisioned, and DCGM returned real GPU
metrics.

## Runtime verification

The infrastructure smoke test requires:

- Prometheus health;
- active and healthy `triton` and `dcgm-exporter` targets;
- Grafana health;
- the provisioned datasource at `http://prometheus:9090`;
- a DCGM response containing metrics with the `DCGM_` prefix.

`monitoring/prometheus/prometheus.yml` and both Grafana provisioning files are
validated statically. If `promtool` is available, deployment validation runs it; an
unavailable binary is reported as `[SKIP]`.

## Planned scope

Step 7 will add latency percentiles, throughput, failures, GPU utilization and
memory panels, model/version comparisons, and alert rules. Their existence or
correctness is not claimed by the step 2 infrastructure.
