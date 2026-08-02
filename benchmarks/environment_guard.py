#!/usr/bin/env python3
"""Host-side NVIDIA telemetry and cross-boundary benchmark trial handshakes."""

from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class GuardError(RuntimeError):
    """Host telemetry cannot prove that a benchmark trial is clean."""


WINFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", ctypes.c_uint32),
    ]


class PDH_VALUE_UNION(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("wideStringValue", ctypes.c_wchar_p),
        ("ansiStringValue", ctypes.c_char_p),
    ]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("CStatus", ctypes.c_uint32), ("value", PDH_VALUE_UNION)]


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("value", PDH_FMT_COUNTERVALUE)]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _guid(value: str) -> GUID:
    return GUID.from_buffer_copy(uuid.UUID(value).bytes_le)


def _com_method(pointer: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])


def enumerate_dxgi_adapters() -> list[dict[str, Any]]:
    if os.name != "nt":
        raise GuardError("DXGI adapter enumeration requires Windows")
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    iid = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long
    if create_factory(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        raise GuardError("CreateDXGIFactory1 failed")
    adapters: list[dict[str, Any]] = []
    try:
        enum_adapters = _com_method(
            factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            status = enum_adapters(factory, index, ctypes.byref(adapter))
            if status & 0xFFFFFFFF == 0x887A0002:
                break
            if status != 0:
                raise GuardError(f"IDXGIFactory1.EnumAdapters1 failed: {status:#x}")
            try:
                description = DXGI_ADAPTER_DESC1()
                get_description = _com_method(
                    adapter,
                    10,
                    ctypes.c_long,
                    ctypes.POINTER(DXGI_ADAPTER_DESC1),
                )
                if get_description(adapter, ctypes.byref(description)) != 0:
                    raise GuardError("IDXGIAdapter1.GetDesc1 failed")
                luid = (
                    f"0x{description.AdapterLuid.HighPart & 0xFFFFFFFF:08X}_"
                    f"0x{description.AdapterLuid.LowPart:08X}"
                )
                adapters.append(
                    {
                        "description": description.Description,
                        "vendor_id": int(description.VendorId),
                        "device_id": int(description.DeviceId),
                        "luid": luid,
                    }
                )
            finally:
                _com_method(adapter, 2, ctypes.c_ulong)(adapter)
            index += 1
    finally:
        _com_method(factory, 2, ctypes.c_ulong)(factory)
    return adapters


def windows_process_names() -> dict[int, str]:
    if os.name != "nt":
        raise GuardError("process enumeration requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise GuardError("CreateToolhelp32Snapshot failed")
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    names: dict[int, str] = {}
    try:
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            names[int(entry.th32ProcessID)] = str(entry.szExeFile)
            has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return names


GPU_ENGINE_INSTANCE = re.compile(
    r"^pid_(?P<pid>[0-9]+)_luid_(?P<luid>0x[0-9A-Fa-f]{8}_0x[0-9A-Fa-f]{8})_"
    r"phys_(?P<physical>[0-9]+)_eng_(?P<engine>[0-9]+)_engtype_(?P<type>.+)$"
)


class WindowsGpuEngineSampler:
    """A persistent PDH wildcard query that refreshes GPU Engine instances."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise GuardError("Windows GPU Engine counters require Windows")
        adapters = enumerate_dxgi_adapters()
        nvidia = [item for item in adapters if item["vendor_id"] == 0x10DE]
        if not nvidia:
            raise GuardError("DXGI did not expose an NVIDIA adapter")
        self.adapters = nvidia
        self._target_luids = {str(item["luid"]).lower() for item in nvidia}
        self._pdh = ctypes.WinDLL("pdh")
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        self._pdh.PdhOpenQueryW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self._pdh.PdhCollectQueryData.restype = ctypes.c_long
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self._pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
        self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise GuardError("PdhOpenQueryW failed")
        if (
            self._pdh.PdhAddEnglishCounterW(
                self._query,
                r"\GPU Engine(*)\Utilization Percentage",
                0,
                ctypes.byref(self._counter),
            )
            != 0
        ):
            self.close()
            raise GuardError("Windows GPU Engine counters are unavailable")
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            self.close()
            raise GuardError("initial Windows GPU Engine counter collection failed")

    def close(self) -> None:
        if getattr(self, "_query", None) and self._query.value:
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()

    def sample(self) -> list[dict[str, Any]]:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            raise GuardError("Windows GPU Engine counter collection failed")
        size = ctypes.c_uint32(0)
        count = ctypes.c_uint32(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter,
            0x00000200,
            ctypes.byref(size),
            ctypes.byref(count),
            None,
        )
        if status & 0xFFFFFFFF != 0x800007D2 or size.value == 0:
            raise GuardError(f"GPU Engine wildcard sizing failed: {status:#x}")
        buffer = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter,
            0x00000200,
            ctypes.byref(size),
            ctypes.byref(count),
            buffer,
        )
        if status != 0:
            raise GuardError(f"GPU Engine wildcard read failed: {status:#x}")
        items = ctypes.cast(buffer, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W))
        process_names = windows_process_names()
        engines: list[dict[str, Any]] = []
        for index in range(count.value):
            item = items[index]
            match = GPU_ENGINE_INSTANCE.fullmatch(item.name or "")
            if match is None or match.group("luid").lower() not in self._target_luids:
                continue
            pid = int(match.group("pid"))
            utilization = None
            if item.value.CStatus == 0:
                candidate = float(item.value.doubleValue)
                if math.isfinite(candidate) and candidate >= 0:
                    utilization = candidate
            engines.append(
                {
                    "pid": pid,
                    "process_name": process_names.get(pid, "unknown"),
                    "adapter_luid": match.group("luid").upper().replace("X", "x"),
                    "physical_adapter_index": int(match.group("physical")),
                    "engine_index": int(match.group("engine")),
                    "engine_type": match.group("type"),
                    "utilization_percent": utilization,
                }
            )
        if not engines:
            raise GuardError("GPU Engine counters returned no NVIDIA adapter instances")
        return engines


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _optional_number(value: str, *, integer: bool = False) -> int | float | None:
    normalized = value.strip().replace("%", "")
    if normalized in {"", "-", "N/A", "[N/A]", "[Not Supported]"}:
        return None
    try:
        return int(float(normalized)) if integer else float(normalized)
    except ValueError:
        return None


def _process_name(value: str) -> str:
    normalized = value.strip().strip('"')
    if not normalized or normalized.startswith("["):
        return "unknown"
    return normalized.replace("\\", "/").rsplit("/", 1)[-1]


def parse_gpu_query(output: str) -> dict[str, Any]:
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 10:
        raise GuardError("nvidia-smi GPU query did not return exactly one GPU")
    uuid, utilization, used, total, sm_clock, memory_clock, pstate, power, temperature, pci_bus_id = (
        item.strip() for item in rows[0]
    )
    parsed = {
        "uuid": uuid,
        "utilization_percent": _optional_number(utilization),
        "memory_used_mib": _optional_number(used, integer=True),
        "memory_total_mib": _optional_number(total, integer=True),
        "sm_clock_mhz": _optional_number(sm_clock, integer=True),
        "memory_clock_mhz": _optional_number(memory_clock, integer=True),
        "pstate": None if pstate in {"", "N/A", "[N/A]"} else pstate,
        "power_draw_watts": _optional_number(power),
        "temperature_celsius": _optional_number(temperature),
        "pci_bus_id": None if pci_bus_id in {"", "N/A", "[N/A]"} else pci_bus_id,
    }
    if not uuid.startswith("GPU-") or any(
        parsed[field] is None
        for field in ("utilization_percent", "memory_used_mib", "memory_total_mib")
    ):
        raise GuardError("nvidia-smi GPU query returned incomplete telemetry")
    return parsed


def parse_compute_apps(output: str) -> dict[int, dict[str, Any]]:
    processes: dict[int, dict[str, Any]] = {}
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        processes[pid] = {
            "pid": pid,
            "process_name": _process_name(row[1]),
            "gpu_memory_mib": _optional_number(row[2], integer=True),
            "gpu_utilization_percent": None,
            "gpu_memory_utilization_percent": None,
        }
    return processes


PMON_ROW = re.compile(
    r"^\s*(?P<gpu>\d+)\s+(?P<pid>\d+)\s+\S+\s+"
    r"(?P<sm>\S+)\s+(?P<mem>\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<fb>\S+)\s+\S+\s+(?P<name>\S+)"
)


def merge_pmon(output: str, processes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    for line in output.splitlines():
        match = PMON_ROW.match(line)
        if match is None:
            continue
        pid = int(match.group("pid"))
        process = processes.setdefault(
            pid,
            {
                "pid": pid,
                "process_name": _process_name(match.group("name")),
                "gpu_memory_mib": None,
                "gpu_utilization_percent": None,
                "gpu_memory_utilization_percent": None,
            },
        )
        if process["process_name"] == "unknown":
            process["process_name"] = _process_name(match.group("name"))
        process["gpu_utilization_percent"] = _optional_number(match.group("sm"))
        process["gpu_memory_utilization_percent"] = _optional_number(
            match.group("mem")
        )
        pmon_memory = _optional_number(match.group("fb"), integer=True)
        if process["gpu_memory_mib"] is None and pmon_memory is not None:
            process["gpu_memory_mib"] = pmon_memory
    return sorted(processes.values(), key=lambda item: (item["pid"], item["process_name"]))


def _normalize_gpu_engines(
    gpu_engines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_by_process: dict[tuple[int, str], dict[str, Any]] = {}
    active_or_unknown_engines: list[dict[str, Any]] = []
    for engine in gpu_engines:
        identity = (int(engine["pid"]), str(engine["process_name"]))
        inventory = inventory_by_process.setdefault(
            identity,
            {
                "pid": identity[0],
                "process_name": identity[1],
                "adapter_luids": set(),
                "engine_types": set(),
                "instance_count": 0,
            },
        )
        inventory["adapter_luids"].add(engine["adapter_luid"])
        inventory["engine_types"].add(engine["engine_type"])
        inventory["instance_count"] += 1
        utilization = engine["utilization_percent"]
        if utilization is None or float(utilization) > 0:
            active_or_unknown_engines.append(engine)
    inventory_rows = [
        {
            **item,
            "adapter_luids": sorted(item["adapter_luids"]),
            "engine_types": sorted(item["engine_types"]),
        }
        for item in sorted(
            inventory_by_process.values(),
            key=lambda value: (value["pid"], value["process_name"]),
        )
    ]
    return inventory_rows, active_or_unknown_engines


def _command(args: list[str], timeout_seconds: float = 10.0) -> str:
    process = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if process.returncode != 0:
        message = (process.stderr or process.stdout).strip()
        raise GuardError(f"{' '.join(args[:2])} failed: {message}")
    return process.stdout


def collect_nvidia_sample(
    sequence: int,
    gpu_engine_sampler: WindowsGpuEngineSampler | None = None,
) -> dict[str, Any]:
    observed_at_utc = _utc_now()
    monotonic_ns = time.monotonic_ns()
    try:
        gpu = parse_gpu_query(
            _command(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        processes = parse_compute_apps(
            _command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        if gpu_engine_sampler is None:
            gpu_engine_sampler = WindowsGpuEngineSampler()
            try:
                time.sleep(1.0)
                gpu_engines = gpu_engine_sampler.sample()
            finally:
                gpu_engine_sampler.close()
        else:
            gpu_engines = gpu_engine_sampler.sample()
        gpu_engine_inventory, active_or_unknown_engines = _normalize_gpu_engines(
            gpu_engines
        )
        return {
            "schema_version": 1,
            "sequence": sequence,
            "observed_at_utc": observed_at_utc,
            "host_monotonic_ns": monotonic_ns,
            "sample_kind": "periodic",
            "device_metrics_source_sequence": sequence,
            "collection_ok": True,
            "gpu": gpu,
            "gpu_engine_inventory": gpu_engine_inventory,
            "gpu_engines": active_or_unknown_engines,
            "wddm_adapters": gpu_engine_sampler.adapters,
            "nvidia_processes": sorted(
                processes.values(),
                key=lambda item: (item["pid"], item["process_name"]),
            ),
            "error": None,
        }
    except (OSError, subprocess.SubprocessError, GuardError) as error:
        return {
            "schema_version": 1,
            "sequence": sequence,
            "observed_at_utc": observed_at_utc,
            "host_monotonic_ns": monotonic_ns,
            "sample_kind": "periodic",
            "device_metrics_source_sequence": sequence,
            "collection_ok": False,
            "gpu": None,
            "gpu_engine_inventory": [],
            "gpu_engines": [],
            "wddm_adapters": [],
            "nvidia_processes": [],
            "error": str(error),
        }


def collect_boundary_sample(
    sequence: int,
    gpu_engine_sampler: WindowsGpuEngineSampler,
    latest_periodic_sample: dict[str, Any],
) -> dict[str, Any]:
    """Capture fresh WDDM attribution without blocking on diagnostic nvidia-smi."""
    observed_at_utc = _utc_now()
    monotonic_ns = time.monotonic_ns()
    try:
        if latest_periodic_sample.get("collection_ok") is not True:
            raise GuardError("latest periodic device sample is unavailable")
        gpu_engines = gpu_engine_sampler.sample()
        inventory, active_or_unknown = _normalize_gpu_engines(gpu_engines)
        return {
            "schema_version": 1,
            "sequence": sequence,
            "observed_at_utc": observed_at_utc,
            "host_monotonic_ns": monotonic_ns,
            "sample_kind": "boundary",
            "device_metrics_source_sequence": int(
                latest_periodic_sample["device_metrics_source_sequence"]
            ),
            "collection_ok": True,
            "gpu": latest_periodic_sample["gpu"],
            "gpu_engine_inventory": inventory,
            "gpu_engines": active_or_unknown,
            "wddm_adapters": gpu_engine_sampler.adapters,
            "nvidia_processes": latest_periodic_sample["nvidia_processes"],
            "error": None,
        }
    except (OSError, subprocess.SubprocessError, GuardError, KeyError, TypeError) as error:
        return {
            "schema_version": 1,
            "sequence": sequence,
            "observed_at_utc": observed_at_utc,
            "host_monotonic_ns": monotonic_ns,
            "sample_kind": "boundary",
            "device_metrics_source_sequence": int(
                latest_periodic_sample.get("device_metrics_source_sequence", sequence)
            ),
            "collection_ok": False,
            "gpu": None,
            "gpu_engine_inventory": [],
            "gpu_engines": [],
            "wddm_adapters": [],
            "nvidia_processes": [],
            "error": str(error),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GuardError(f"telemetry line {line_number} is not an object")
            rows.append(value)
    return rows


def _normalized_process_name(value: str) -> str:
    return value.strip().lower().removesuffix(".exe")


def _engine_is_active(engine: dict[str, Any], guard: dict[str, Any]) -> bool:
    utilization = engine.get("utilization_percent")
    return utilization is not None and float(utilization) > float(
        guard["gpu_engine_activity_threshold_percent"]
    )


def recompute_guard(
    telemetry: list[dict[str, Any]],
    boundary: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    """Recompute telemetry continuity and attributed interference from raw samples."""
    by_sequence = {row.get("sequence"): row for row in telemetry}
    baseline_start = int(boundary["baseline_start_seq"])
    baseline_end = int(boundary["baseline_end_seq"])
    guard_start = int(boundary["guard_start_seq"])
    guard_end = int(boundary["guard_end_seq"])
    if not (baseline_start <= baseline_end <= guard_start <= guard_end):
        return {"classification": "ERROR", "reasons": ["invalid_boundary_order"]}
    rows: list[dict[str, Any]] = []
    for sequence in range(baseline_start, guard_end + 1):
        row = by_sequence.get(sequence)
        if row is None:
            return {"classification": "ERROR", "reasons": ["telemetry_sequence_gap"]}
        rows.append(row)
    if any(row.get("collection_ok") is not True for row in rows):
        return {"classification": "ERROR", "reasons": ["telemetry_collection_failure"]}
    maximum_gap_ns = int(guard["maximum_sample_gap_ms"]) * 1_000_000
    for previous, current in zip(rows, rows[1:]):
        if (
            not isinstance(previous.get("host_monotonic_ns"), int)
            or not isinstance(current.get("host_monotonic_ns"), int)
            or current["host_monotonic_ns"] <= previous["host_monotonic_ns"]
            or current["host_monotonic_ns"] - previous["host_monotonic_ns"]
            > maximum_gap_ns
        ):
            return {"classification": "ERROR", "reasons": ["telemetry_time_gap"]}
    baseline_rows = [
        by_sequence[sequence] for sequence in range(baseline_start, baseline_end + 1)
    ]
    trial_rows = [
        by_sequence[sequence] for sequence in range(guard_start, guard_end + 1)
    ]
    if len(baseline_rows) < int(guard["minimum_consecutive_baseline_samples"]):
        return {"classification": "ERROR", "reasons": ["insufficient_valid_baseline"]}
    if any(not row.get("gpu_engine_inventory") for row in rows):
        return {"classification": "ERROR", "reasons": ["windows_gpu_engine_unavailable"]}
    baseline_process_set = {
        (
            int(process["pid"]),
            _normalized_process_name(str(process["process_name"])),
        )
        for row in baseline_rows
        for process in row.get("gpu_engine_inventory", [])
    }
    baseline_engine_set = {
        (
            int(process["pid"]),
            _normalized_process_name(str(process["process_name"])),
            str(engine_type),
        )
        for row in baseline_rows
        for process in row.get("gpu_engine_inventory", [])
        for engine_type in process.get("engine_types", [])
    }
    baseline_active_processes = {
        (int(engine["pid"]), _normalized_process_name(str(engine["process_name"])))
        for row in baseline_rows
        for engine in row.get("gpu_engines", [])
        if _engine_is_active(engine, guard)
    }
    forbidden = {
        _normalized_process_name(name) for name in guard["forbidden_processes"]
    }
    owned = {
        _normalized_process_name(name) for name in guard["benchmark_owned_processes"]
    }
    attributed: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in trial_rows:
        for engine in row.get("gpu_engines", []):
            if not _engine_is_active(engine, guard):
                continue
            identity = (
                int(engine["pid"]),
                _normalized_process_name(str(engine["process_name"])),
            )
            if identity[1] in owned:
                continue
            reason: str | None = None
            if identity[1] in forbidden:
                reason = "forbidden_process_activity"
            elif identity not in baseline_process_set:
                reason = "new_gpu_process_activity"
            elif identity not in baseline_active_processes:
                reason = "baseline_idle_process_became_active"
            if reason is not None:
                key = (identity[0], identity[1], str(engine["engine_type"]), reason)
                attributed.setdefault(key, {
                    "pid": identity[0],
                    "process_name": engine["process_name"],
                    "engine_type": engine["engine_type"],
                    "utilization_percent": float(engine["utilization_percent"]),
                    "reason": reason,
                    "first_sequence": int(row["sequence"]),
                })
    reasons = sorted(
        attributed.values(),
        key=lambda item: (item["first_sequence"], item["pid"], item["reason"]),
    )
    return {
        "classification": "CONTAMINATED" if reasons else "CLEAN",
        "reasons": reasons,
        "baseline_processes": [
            {"pid": pid, "process_name": name}
            for pid, name in sorted(baseline_process_set)
        ],
        "baseline_active_processes": [
            {"pid": pid, "process_name": name}
            for pid, name in sorted(baseline_active_processes)
        ],
        "baseline_engines": [
            {
                "pid": pid,
                "process_name": name,
                "engine_type": engine_type,
            }
            for pid, name, engine_type in sorted(baseline_engine_set)
        ],
    }


def classify_trial(
    *,
    scenario_status: str,
    runtime_error: bool,
    guard_classification: str,
    measurement_valid: bool,
) -> str:
    """Classify validity without interpreting PA stability or performance values."""
    if runtime_error or guard_classification == "ERROR":
        return "ERROR"
    if guard_classification == "CONTAMINATED":
        return "CONTAMINATED"
    if scenario_status != "formal":
        return "ERROR"
    return "VALID" if measurement_valid else "ERROR"


class BoundaryClient:
    """Container-side marker/ack client using the shared benchmark cache."""

    def __init__(self, run_root: Path, guard: dict[str, Any]) -> None:
        self.root = run_root / "guard"
        self.guard = guard
        self.markers = self.root / "markers"
        self.acks = self.root / "acks"

    def _wait(self, path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + float(self.guard["ack_timeout_seconds"])
        while time.monotonic() < deadline:
            error_path = self.root / "observer-error.json"
            if error_path.is_file():
                raise GuardError(json.loads(error_path.read_text(encoding="utf-8"))["error"])
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise GuardError(f"invalid guard acknowledgement: {path.name}")
                return value
            time.sleep(0.05)
        raise GuardError(f"host observer acknowledgement timed out: {path.name}")

    def start(self, trial_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        _atomic_json(
            self.markers / f"{trial_id}.start.json",
            {
                "schema_version": 1,
                "kind": "trial-start",
                "trial_id": trial_id,
                "written_at_utc": _utc_now(),
                **metadata,
            },
        )
        return self._wait(self.acks / f"{trial_id}.start.json")

    def end(self, trial_id: str) -> dict[str, Any]:
        _atomic_json(
            self.markers / f"{trial_id}.end.json",
            {
                "schema_version": 1,
                "kind": "trial-end",
                "trial_id": trial_id,
                "written_at_utc": _utc_now(),
            },
        )
        return self._wait(self.acks / f"{trial_id}.end.json")

    def checkpoint(self, trial_id: str, pass_number: int) -> dict[str, Any]:
        suffix = f"pass-{pass_number:02d}"
        _atomic_json(
            self.markers / f"{trial_id}.{suffix}.json",
            {
                "schema_version": 1,
                "kind": "trial-pass",
                "trial_id": trial_id,
                "pass_number": pass_number,
                "written_at_utc": _utc_now(),
            },
        )
        return self._wait(self.acks / f"{trial_id}.{suffix}.json")

    def stop(self) -> dict[str, Any]:
        _atomic_json(
            self.root / "stop-request.json",
            {"schema_version": 1, "written_at_utc": _utc_now()},
        )
        return self._wait(self.root / "observer-complete.json")


class HostObserver:
    """Windows host observer that owns nvidia-smi sampling and acknowledgements."""

    def __init__(self, run_root: Path, guard: dict[str, Any]) -> None:
        self.root = run_root / "guard"
        self.guard = guard
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.actions_path = self.root / "actions.jsonl"
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None

    def start(self) -> None:
        if os.name != "nt":
            raise GuardError("environment guard host observer requires Windows")
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="benchmark-gpu-observer")
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise GuardError("host observer did not stop")
        if self._failure is not None:
            raise GuardError(str(self._failure))

    def force_stop(self) -> None:
        self._stop_event.set()

    def _terminate_forbidden(self, sample: dict[str, Any]) -> None:
        forbidden = {
            _normalized_process_name(name)
            for name in self.guard["forbidden_processes"]
        }
        handled: set[int] = set()
        for engine in sample.get("gpu_engines", []):
            pid = int(engine["pid"])
            if (
                _normalized_process_name(str(engine.get("process_name", "")))
                not in forbidden
                or not _engine_is_active(engine, self.guard)
                or pid in handled
            ):
                continue
            handled.add(pid)
            if pid in {os.getpid(), os.getppid()}:
                continue
            command = ["taskkill", "/PID", str(pid), "/T", "/F"]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            _append_jsonl(
                self.actions_path,
                {
                    "schema_version": 1,
                    "observed_at_utc": _utc_now(),
                    "host_monotonic_ns": time.monotonic_ns(),
                    "pid": pid,
                    "process_name": engine["process_name"],
                    "engine_type": engine["engine_type"],
                    "utilization_percent": engine["utilization_percent"],
                    "action": "terminate_forbidden_gpu_process",
                    "succeeded": result.returncode == 0,
                },
            )

    def _run(self) -> None:
        sequence = 0
        pending: dict[str, dict[str, Any]] = {}
        handled_ends: set[str] = set()
        handled_passes: set[str] = set()
        interval = float(self.guard["sample_interval_ms"]) / 1000.0
        marker_poll_seconds = min(0.02, interval / 10.0)
        sampler: WindowsGpuEngineSampler | None = None
        latest_periodic_sample: dict[str, Any] | None = None
        latest_sample: dict[str, Any] | None = None
        try:
            sampler = WindowsGpuEngineSampler()
            next_periodic = time.monotonic() + interval

            def record(sample: dict[str, Any]) -> None:
                nonlocal latest_sample
                _append_jsonl(self.telemetry_path, sample)
                latest_sample = sample

            def record_boundary() -> dict[str, Any]:
                nonlocal sequence
                if latest_periodic_sample is None:
                    raise GuardError("boundary observed before first periodic sample")
                sequence += 1
                value = collect_boundary_sample(
                    sequence, sampler, latest_periodic_sample
                )
                record(value)
                return value

            while not self._stop_event.is_set():
                now = time.monotonic()
                if now >= next_periodic:
                    sequence += 1
                    latest_periodic_sample = collect_nvidia_sample(sequence, sampler)
                    record(latest_periodic_sample)
                    next_periodic += interval
                    while next_periodic <= time.monotonic():
                        next_periodic += interval
                for marker in sorted((self.root / "markers").glob("*.start.json")):
                    trial_id = marker.name.removesuffix(".start.json")
                    ack = self.root / "acks" / f"{trial_id}.start.json"
                    if ack.exists() or trial_id in pending:
                        continue
                    if latest_periodic_sample is None:
                        continue
                    baseline_start = record_boundary()
                    pending[trial_id] = {
                        "detected_monotonic": time.monotonic(),
                        "baseline_start_seq": baseline_start["sequence"],
                    }
                for trial_id, state in list(pending.items()):
                    if time.monotonic() - state["detected_monotonic"] < float(
                        self.guard["baseline_seconds"]
                    ):
                        continue
                    sample = record_boundary()
                    _atomic_json(
                        self.root / "acks" / f"{trial_id}.start.json",
                        {
                            "schema_version": 1,
                            "trial_id": trial_id,
                            "baseline_start_seq": state["baseline_start_seq"],
                            "baseline_end_seq": sequence,
                            "guard_start_seq": sequence,
                            "observed_at_utc": sample["observed_at_utc"],
                            "host_monotonic_ns": sample["host_monotonic_ns"],
                        },
                    )
                    del pending[trial_id]
                for marker in sorted((self.root / "markers").glob("*.pass-*.json")):
                    marker_key = marker.name.removesuffix(".json")
                    if marker_key in handled_passes:
                        continue
                    value = json.loads(marker.read_text(encoding="utf-8"))
                    trial_id = str(value["trial_id"])
                    pass_number = int(value["pass_number"])
                    start_ack_path = self.root / "acks" / f"{trial_id}.start.json"
                    if not start_ack_path.is_file():
                        raise GuardError(
                            f"trial pass observed before start ack: {trial_id}/{pass_number}"
                        )
                    sample = record_boundary()
                    suffix = f"pass-{pass_number:02d}"
                    _atomic_json(
                        self.root / "acks" / f"{trial_id}.{suffix}.json",
                        {
                            "schema_version": 1,
                            "trial_id": trial_id,
                            "pass_number": pass_number,
                            "sequence": sequence,
                            "observed_at_utc": sample["observed_at_utc"],
                            "host_monotonic_ns": sample["host_monotonic_ns"],
                        },
                    )
                    handled_passes.add(marker_key)
                for marker in sorted((self.root / "markers").glob("*.end.json")):
                    trial_id = marker.name.removesuffix(".end.json")
                    if trial_id in handled_ends:
                        continue
                    start_ack_path = self.root / "acks" / f"{trial_id}.start.json"
                    if not start_ack_path.is_file():
                        raise GuardError(f"trial end observed before start ack: {trial_id}")
                    start_ack = json.loads(start_ack_path.read_text(encoding="utf-8"))
                    sample = record_boundary()
                    _atomic_json(
                        self.root / "acks" / f"{trial_id}.end.json",
                        {
                            "schema_version": 1,
                            "trial_id": trial_id,
                            "baseline_start_seq": start_ack["baseline_start_seq"],
                            "baseline_end_seq": start_ack["baseline_end_seq"],
                            "guard_start_seq": start_ack["guard_start_seq"],
                            "guard_end_seq": sequence,
                            "observed_at_utc": sample["observed_at_utc"],
                            "host_monotonic_ns": sample["host_monotonic_ns"],
                        },
                    )
                    handled_ends.add(trial_id)
                if (self.root / "stop-request.json").is_file():
                    if latest_sample is None:
                        raise GuardError("observer stopped before collecting telemetry")
                    _atomic_json(
                        self.root / "observer-complete.json",
                        {
                            "schema_version": 1,
                            "last_sequence": sequence,
                            "observed_at_utc": latest_sample["observed_at_utc"],
                            "host_monotonic_ns": latest_sample["host_monotonic_ns"],
                        },
                    )
                    return
                delay = min(
                    marker_poll_seconds,
                    max(0.0, next_periodic - time.monotonic()),
                )
                self._stop_event.wait(delay)
        except BaseException as error:
            self._failure = error
            _atomic_json(
                self.root / "observer-error.json",
                {"schema_version": 1, "error": str(error), "observed_at_utc": _utc_now()},
            )
        finally:
            if sampler is not None:
                sampler.close()


def signal_name(value: int) -> str:
    try:
        return signal.Signals(value).name
    except ValueError:
        return str(value)
