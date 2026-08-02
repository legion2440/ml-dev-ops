# Grafana

The Prometheus datasource has stable UID `prometheus`. The existing file provider
loads `dashboards/ml-dev-ops.json` as the `ML DevOps Inference` dashboard with UID
`ml-dev-ops-inference`.

The runtime verifier executes every panel expression through Grafana's datasource
proxy. This proves the Grafana-to-Prometheus data path without relying on a browser
screenshot.
