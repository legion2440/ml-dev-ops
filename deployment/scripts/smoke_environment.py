"""Run host-side smoke checks for the step 2 container infrastructure."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"
EXPECTED_SERVICES = {"triton", "prometheus", "grafana", "dcgm-exporter"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    detail: str | None = None


def _select_env_file(argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
    else:
        local_env = REPOSITORY_ROOT / ".env"
        path = local_env if local_env.is_file() else REPOSITORY_ROOT / ".env.example"
    if not path.is_file():
        raise FileNotFoundError(f"Compose environment file does not exist: {path}")
    return path.resolve()


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path.name}:{line_number} is not a KEY=VALUE assignment")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not key:
            raise ValueError(f"{path.name}:{line_number} has an empty key")
        values[key] = value
    return values


def _required(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    if not value:
        raise ValueError(f"Required environment value is missing: {key}")
    return value


def _compose_command(env_file: Path, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-directory",
        str(REPOSITORY_ROOT),
        "--file",
        str(COMPOSE_FILE),
        "--env-file",
        str(env_file),
        *arguments,
    ]


def _check_containers(env_file: Path) -> CheckResult:
    process = subprocess.run(
        _compose_command(env_file, "ps", "--services", "--status", "running"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        return CheckResult("containers", False, "Expected containers are unavailable", detail)

    running = {line.strip() for line in process.stdout.splitlines() if line.strip()}
    missing = sorted(EXPECTED_SERVICES - running)
    if missing:
        return CheckResult(
            "containers",
            False,
            "Expected containers are unavailable",
            f"not running: {', '.join(missing)}",
        )
    return CheckResult("containers", True, "All four expected containers are running")


def _request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def _eventually(
    name: str,
    success_message: str,
    url: str,
    predicate: Callable[[int, bytes], tuple[bool, str | None]],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> CheckResult:
    deadline = time.monotonic() + timeout
    last_detail = "endpoint did not respond"
    while True:
        try:
            status, body = _request(url, headers=headers, method=method)
            ok, detail = predicate(status, body)
            if ok:
                return CheckResult(name, True, success_message, detail)
            last_detail = detail or f"unexpected HTTP {status}"
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as error:
            last_detail = str(error)

        if time.monotonic() >= deadline:
            return CheckResult(name, False, f"{success_message} check failed", last_detail)
        time.sleep(2)


def _status_ok(status: int, _: bytes) -> tuple[bool, str | None]:
    return status == 200, None


def _empty_repository(status: int, body: bytes) -> tuple[bool, str | None]:
    if status != 200:
        return False, f"unexpected HTTP {status}"
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, list):
        return False, "repository index is not a JSON array"
    if payload:
        names = [str(item.get("name", "<unknown>")) for item in payload if isinstance(item, dict)]
        return False, f"repository is not empty: {', '.join(names) or payload!r}"
    return True, None


def _contains_metric(prefix: str) -> Callable[[int, bytes], tuple[bool, str | None]]:
    def predicate(status: int, body: bytes) -> tuple[bool, str | None]:
        if status != 200:
            return False, f"unexpected HTTP {status}"
        text = body.decode("utf-8", errors="replace")
        if prefix not in text:
            return False, f"response contains no metric with prefix {prefix}"
        return True, None

    return predicate


def _prometheus_targets(status: int, body: bytes) -> tuple[bool, str | None]:
    if status != 200:
        return False, f"unexpected HTTP {status}"
    payload = json.loads(body.decode("utf-8"))
    targets = payload.get("data", {}).get("activeTargets", [])
    health_by_job = {
        target.get("labels", {}).get("job"): target.get("health")
        for target in targets
        if isinstance(target, dict)
    }
    expected = {"triton", "dcgm-exporter"}
    missing = sorted(expected - set(health_by_job))
    unhealthy = sorted(job for job in expected if health_by_job.get(job) != "up")
    if missing:
        return False, f"Prometheus targets missing: {', '.join(missing)}"
    if unhealthy:
        return False, f"Prometheus targets not up: {', '.join(unhealthy)}"
    return True, None


def _grafana_health(status: int, body: bytes) -> tuple[bool, str | None]:
    if status != 200:
        return False, f"unexpected HTTP {status}"
    payload = json.loads(body.decode("utf-8"))
    database = payload.get("database")
    return database == "ok", None if database == "ok" else f"database={database!r}"


def _grafana_datasource(status: int, body: bytes) -> tuple[bool, str | None]:
    if status != 200:
        return False, f"unexpected HTTP {status}"
    payload = json.loads(body.decode("utf-8"))
    uid = payload.get("uid")
    url = payload.get("url")
    ok = uid == "prometheus" and url == "http://prometheus:9090"
    return ok, None if ok else f"uid={uid!r}, url={url!r}"


def _base_url(port: str) -> str:
    return f"http://127.0.0.1:{port}"


def run_checks(env_file: Path, env: dict[str, str], timeout: float) -> list[CheckResult]:
    containers = _check_containers(env_file)
    if not containers.ok:
        return [containers]

    triton_http = _base_url(_required(env, "TRITON_HTTP_PORT"))
    triton_metrics = _base_url(_required(env, "TRITON_METRICS_PORT"))
    prometheus = _base_url(_required(env, "PROMETHEUS_PORT"))
    grafana = _base_url(_required(env, "GRAFANA_PORT"))
    dcgm = _base_url(_required(env, "DCGM_METRICS_PORT"))

    credentials = (
        f"{_required(env, 'GRAFANA_ADMIN_USER')}:{_required(env, 'GRAFANA_ADMIN_PASSWORD')}"
    )
    authorization = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    grafana_headers = {"Authorization": f"Basic {authorization}"}

    return [
        containers,
        _eventually(
            "triton_liveness",
            "Triton is live",
            f"{triton_http}/v2/health/live",
            _status_ok,
            timeout=timeout,
        ),
        _eventually(
            "triton_readiness",
            "Triton is ready",
            f"{triton_http}/v2/health/ready",
            _status_ok,
            timeout=timeout,
        ),
        _eventually(
            "triton_repository",
            "Triton model repository is empty",
            f"{triton_http}/v2/repository/index",
            _empty_repository,
            timeout=timeout,
            method="POST",
        ),
        _eventually(
            "triton_metrics",
            "Triton metrics are available",
            f"{triton_metrics}/metrics",
            _contains_metric("nv_"),
            timeout=timeout,
        ),
        _eventually(
            "prometheus_health",
            "Prometheus is healthy",
            f"{prometheus}/-/healthy",
            _status_ok,
            timeout=timeout,
        ),
        _eventually(
            "prometheus_targets",
            "Prometheus sees Triton and DCGM targets",
            f"{prometheus}/api/v1/targets",
            _prometheus_targets,
            timeout=timeout,
        ),
        _eventually(
            "grafana_health",
            "Grafana is healthy",
            f"{grafana}/api/health",
            _grafana_health,
            timeout=timeout,
        ),
        _eventually(
            "grafana_datasource",
            "Prometheus datasource is provisioned",
            f"{grafana}/api/datasources/uid/prometheus",
            _grafana_datasource,
            timeout=timeout,
            headers=grafana_headers,
        ),
        _eventually(
            "dcgm_metrics",
            "DCGM metrics are available",
            f"{dcgm}/metrics",
            _contains_metric("DCGM_"),
            timeout=timeout,
        ),
    ]


def _render_human(results: list[CheckResult]) -> None:
    for result in results:
        label = "OK" if result.ok else "FAIL"
        print(f"[{label}] {result.message}")
        if result.detail and not result.ok:
            print(f"       {result.detail}")


def _render_json(results: list[CheckResult], env_file: Path) -> None:
    payload: dict[str, Any] = {
        "ok": all(result.ok for result in results),
        "env_file": env_file.name,
        "checks": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Compose environment file.")
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        dest="output_format",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each endpoint.",
    )
    args = parser.parse_args()

    try:
        env_file = _select_env_file(args.env_file)
        env = _load_env(env_file)
        results = run_checks(env_file, env, args.timeout)
    except (OSError, ValueError) as error:
        if args.output_format == "json":
            print(json.dumps({"ok": False, "error": str(error)}, indent=2, sort_keys=True))
        else:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        _render_json(results, env_file)
    else:
        _render_human(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
