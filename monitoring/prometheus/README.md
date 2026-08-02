# Prometheus

`prometheus.yml` defines self, Triton, and DCGM Exporter scrape jobs and loads
`alerts.yml` from a read-only mount. The two Step 7 rules are:

- `HighInferenceLatency`: project-defined average latency above 100 ms for two
  minutes while request traffic is present;
- `InferenceRequestFailures`: at least one failed request in five minutes.

Alertmanager and notification delivery are intentionally outside the assignment
scope. Runtime evidence proves that Prometheus loaded both definitions; neither rule
is required to enter the firing state.
