# Deployment

## Current status

Deployment is planned. `docker-compose.yml` is an intentionally empty, valid
scaffold and starts no services.

## Scope

The deployment implementation will define pinned container images, GPU reservation,
health checks, service dependencies, ports, volumes, automatic observability
provisioning, and lifecycle scripts.

The target services are NVIDIA Triton, Prometheus, Grafana, and NVIDIA DCGM
Exporter, with an optional client and benchmark container.

## Boundary

Deployment orchestrates containers and configuration. It does not preprocess
images, export models, or contain inference postprocessing.
