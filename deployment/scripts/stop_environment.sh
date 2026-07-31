#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
# shellcheck source=compose_common.sh
source "${SCRIPT_DIR}/compose_common.sh"

require_docker

case "${1:-}" in
  "")
    compose down --remove-orphans
    printf '[OK] Environment stopped; named volumes were preserved.\n'
    ;;
  --purge)
    compose down --remove-orphans --volumes
    printf '[OK] Environment stopped; Compose named volumes were removed.\n'
    printf '[OK] Repository-owned host directories were preserved.\n'
    ;;
  *)
    die "Usage: deployment/scripts/stop_environment.sh [--purge]"
    ;;
esac
