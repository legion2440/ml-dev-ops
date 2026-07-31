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
compose up --detach --wait --build triton

printf '[OK] Triton is running with an explicitly controlled model repository.\n'
printf 'Triton HTTP: http://%s\n' "$(service_address triton 8000)"
