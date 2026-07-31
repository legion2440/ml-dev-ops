#!/usr/bin/env bash

# Shared host-side Compose command construction. This file never executes .env files.

readonly DEPLOYMENT_SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
readonly REPOSITORY_ROOT="$(
  cd -- "${DEPLOYMENT_SCRIPT_DIR}/../.." >/dev/null 2>&1
  pwd -P
)"
readonly COMPOSE_FILE="${REPOSITORY_ROOT}/docker-compose.yml"

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command is unavailable: $1"
}

resolve_env_file() {
  local candidate="${ML_DEV_OPS_ENV_FILE:-}"
  if [[ -z "${candidate}" ]]; then
    if [[ -f "${REPOSITORY_ROOT}/.env" ]]; then
      candidate="${REPOSITORY_ROOT}/.env"
    else
      candidate="${REPOSITORY_ROOT}/.env.example"
    fi
  elif [[ "${candidate}" != /* && ! "${candidate}" =~ ^[A-Za-z]:[\\/].* ]]; then
    candidate="${REPOSITORY_ROOT}/${candidate}"
  fi

  [[ -f "${candidate}" ]] || die "Compose environment file does not exist: ${candidate}"
  printf '%s\n' "${candidate}"
}

readonly COMPOSE_ENV_FILE="$(resolve_env_file)"
readonly -a COMPOSE_COMMAND=(
  docker
  compose
  --project-directory
  "${REPOSITORY_ROOT}"
  --file
  "${COMPOSE_FILE}"
  --env-file
  "${COMPOSE_ENV_FILE}"
)

compose() {
  "${COMPOSE_COMMAND[@]}" "$@"
}

require_docker() {
  require_command docker
  docker compose version >/dev/null 2>&1 ||
    die "Docker Compose v2 is required."
  docker info >/dev/null 2>&1 ||
    die "Docker daemon is unavailable. Start Docker Desktop or Docker Engine."
}

require_nvidia_runtime() {
  require_command nvidia-smi
  nvidia-smi -L >/dev/null 2>&1 ||
    die "The host NVIDIA driver cannot discover a GPU."

  local runtimes
  runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  [[ "${runtimes}" == *nvidia* ]] ||
    die "Docker does not report the NVIDIA container runtime."
}

service_address() {
  local service="$1"
  local container_port="$2"
  compose port "${service}" "${container_port}"
}
