"""Triton HTTP/gRPC inference transports and HTTP repository control."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceResult:
    output: np.ndarray
    model_name: str
    model_version: str


class HttpTransport:
    protocol = "http"

    def __init__(self, url: str, timeout: float) -> None:
        import tritonclient.http as httpclient

        self._module = httpclient
        self._timeout = timeout
        self._client = httpclient.InferenceServerClient(
            url=url,
            connection_timeout=timeout,
            network_timeout=timeout,
        )

    def health(self) -> dict[str, bool]:
        return {
            "live": bool(self._client.is_server_live()),
            "ready": bool(self._client.is_server_ready()),
        }

    def metadata(self, model: str, version: str | None) -> dict[str, Any]:
        value = self._client.get_model_metadata(model, model_version=version or "")
        if not isinstance(value, dict):
            raise TransportError("HTTP metadata response is not a JSON object")
        return value

    def is_ready(self, model: str, version: str | None) -> bool:
        return bool(self._client.is_model_ready(model, model_version=version or ""))

    def infer(
        self,
        *,
        model: str,
        version: str | None,
        input_name: str,
        output_name: str,
        tensor: np.ndarray,
        request_id: str,
    ) -> InferenceResult:
        item = self._module.InferInput(input_name, list(tensor.shape), "FP32")
        item.set_data_from_numpy(tensor, binary_data=True)
        requested = self._module.InferRequestedOutput(output_name, binary_data=True)
        result = self._client.infer(
            model,
            [item],
            model_version=version or "",
            outputs=[requested],
            request_id=request_id,
            timeout=int(self._timeout * 1_000_000),
        )
        output = result.as_numpy(output_name)
        response = result.get_response()
        if output is None or not isinstance(response, dict):
            raise TransportError("HTTP inference response is incomplete")
        return InferenceResult(
            output=output,
            model_name=str(response["model_name"]),
            model_version=str(response["model_version"]),
        )


class GrpcTransport:
    protocol = "grpc"

    def __init__(self, url: str, timeout: float) -> None:
        import tritonclient.grpc as grpcclient

        self._module = grpcclient
        self._timeout = timeout
        self._client = grpcclient.InferenceServerClient(url=url)

    def health(self) -> dict[str, bool]:
        return {
            "live": bool(self._client.is_server_live(client_timeout=self._timeout)),
            "ready": bool(self._client.is_server_ready(client_timeout=self._timeout)),
        }

    def metadata(self, model: str, version: str | None) -> dict[str, Any]:
        value = self._client.get_model_metadata(
            model,
            model_version=version or "",
            as_json=True,
            client_timeout=self._timeout,
        )
        if not isinstance(value, dict):
            raise TransportError("gRPC metadata response is not a JSON object")
        return value

    def is_ready(self, model: str, version: str | None) -> bool:
        return bool(
            self._client.is_model_ready(
                model,
                model_version=version or "",
                client_timeout=self._timeout,
            )
        )

    def infer(
        self,
        *,
        model: str,
        version: str | None,
        input_name: str,
        output_name: str,
        tensor: np.ndarray,
        request_id: str,
    ) -> InferenceResult:
        item = self._module.InferInput(input_name, list(tensor.shape), "FP32")
        item.set_data_from_numpy(tensor)
        requested = self._module.InferRequestedOutput(output_name)
        result = self._client.infer(
            model,
            [item],
            model_version=version or "",
            outputs=[requested],
            request_id=request_id,
            client_timeout=self._timeout,
        )
        output = result.as_numpy(output_name)
        response = result.get_response()
        if output is None:
            raise TransportError("gRPC inference response has no requested output")
        return InferenceResult(
            output=output,
            model_name=str(response.model_name),
            model_version=str(response.model_version),
        )


def create_transport(protocol: str, http_url: str, grpc_url: str, timeout: float) -> Any:
    if protocol == "http":
        return HttpTransport(http_url, timeout)
    if protocol == "grpc":
        return GrpcTransport(grpc_url, timeout)
    raise ValueError(f"Unsupported protocol: {protocol}")


class RepositoryController:
    """Use Triton's HTTP-only model repository control API."""

    def __init__(self, http_url: str, timeout: float = 60.0) -> None:
        self._base_url = f"http://{http_url}"
        self._timeout = timeout

    def _request(
        self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise TransportError(f"Triton HTTP {error.code} for {path}: {detail}") from error
        except urllib.error.URLError as error:
            raise TransportError(f"Cannot reach Triton repository API: {error.reason}") from error
        return json.loads(body) if body else None

    def ready_set(self) -> set[tuple[str, str]]:
        rows = self._request("/v2/repository/index", method="POST", payload={})
        if not isinstance(rows, list):
            raise TransportError("Triton repository index is not a JSON array")
        return {
            (str(row["name"]), str(row["version"]))
            for row in rows
            if row.get("state") == "READY" and row.get("version") is not None
        }

    def load(self, model: str) -> None:
        self._request(f"/v2/repository/models/{model}/load", method="POST", payload={})

    def unload(self, model: str) -> None:
        self._request(f"/v2/repository/models/{model}/unload", method="POST", payload={})

    def ensure_ready(
        self,
        transport: Any,
        model: str,
        version: str | None,
        *,
        auto_load: bool,
    ) -> bool:
        if transport.is_ready(model, version):
            return False
        if not auto_load:
            version_label = version or "default"
            raise TransportError(f"Model {model}:{version_label} is not READY")
        self.load(model)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if transport.is_ready(model, version):
                return True
            time.sleep(0.25)
        version_label = version or "default"
        raise TransportError(f"Timed out waiting for {model}:{version_label} readiness")
