#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
# shellcheck source=compose_common.sh
source "${SCRIPT_DIR}/compose_common.sh"

require_docker
require_command python

compose ps
python "${SCRIPT_DIR}/smoke_environment.py" --env-file "${COMPOSE_ENV_FILE}" "$@"
