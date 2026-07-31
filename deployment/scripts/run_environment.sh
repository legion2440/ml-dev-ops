#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
# shellcheck source=compose_common.sh
source "${SCRIPT_DIR}/compose_common.sh"

require_docker
require_nvidia_runtime

compose config --quiet
compose up --detach --wait --build

printf '[OK] Environment is running with %s\n' "${COMPOSE_ENV_FILE}"
printf 'Triton HTTP:    http://%s\n' "$(service_address triton 8000)"
printf 'Triton gRPC:    %s\n' "$(service_address triton 8001)"
printf 'Triton metrics: http://%s/metrics\n' "$(service_address triton 8002)"
printf 'Prometheus:     http://%s\n' "$(service_address prometheus 9090)"
printf 'Grafana:        http://%s\n' "$(service_address grafana 3000)"
printf 'DCGM metrics:   http://%s/metrics\n' "$(service_address dcgm-exporter 9400)"
