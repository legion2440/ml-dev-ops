from __future__ import annotations

import json
import unittest

from benchmarks.environment_guard import (
    classify_trial,
    merge_pmon,
    parse_compute_apps,
    parse_gpu_query,
    recompute_guard,
)
from benchmarks.run_benchmark import CONFIG_PATH
from scripts.validate_benchmark_evidence import _slot_numbering_errors


def sample(
    sequence: int,
    engines: list[dict[str, object]],
    total_utilization: float = 10.0,
) -> dict[str, object]:
    inventory: dict[tuple[int, str], set[str]] = {}
    for item in engines:
        key = (int(item["pid"]), str(item["process_name"]))
        inventory.setdefault(key, set()).add(str(item["engine_type"]))
    return {
        "schema_version": 1,
        "sequence": sequence,
        "observed_at_utc": f"2026-08-02T00:00:0{sequence}Z",
        "host_monotonic_ns": sequence * 1_000_000_000,
        "sample_kind": "periodic",
        "device_metrics_source_sequence": sequence,
        "collection_ok": True,
        "gpu": {
            "uuid": "GPU-example",
            "utilization_percent": total_utilization,
            "memory_used_mib": 100,
            "memory_total_mib": 12000,
            "sm_clock_mhz": 1800,
        },
        "gpu_engine_inventory": [
            {
                "pid": pid,
                "process_name": name,
                "adapter_luids": ["0x00000000_0x00000001"],
                "engine_types": sorted(engine_types),
                "instance_count": len(engine_types),
            }
            for (pid, name), engine_types in sorted(inventory.items())
        ],
        "gpu_engines": engines,
        "nvidia_processes": [],
        "error": None,
    }


def engine(
    pid: int,
    name: str,
    utilization: float | None,
    engine_type: str = "3D",
    engine_index: int = 0,
) -> dict[str, object]:
    return {
        "pid": pid,
        "process_name": name,
        "adapter_luid": "0x00000000_0x00000001",
        "physical_adapter_index": 0,
        "engine_index": engine_index,
        "engine_type": engine_type,
        "utilization_percent": utilization,
    }


class EnvironmentGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.guard = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))[
            "environment_guard"
        ]
        cls.boundary = {
            "baseline_start_seq": 1,
            "baseline_end_seq": 5,
            "guard_start_seq": 5,
            "guard_end_seq": 7,
        }

    def test_nvidia_outputs_keep_unknown_process_fields_as_null(self) -> None:
        gpu = parse_gpu_query(
            "GPU-example, 33, 1389, 12282, 255, 7001, P2, 45.5, 62, 00000000:01:00.0\n"
        )
        self.assertEqual(gpu["utilization_percent"], 33.0)
        apps = parse_compute_apps("123, C:\\Tools\\browser.exe, [N/A]\n")
        merged = merge_pmon(
            "    0        123   C+G      -      -      -      -      -      -      -      0    browser.exe\n",
            apps,
        )
        self.assertIsNone(merged[0]["gpu_utilization_percent"])
        self.assertIsNone(merged[0]["gpu_memory_utilization_percent"])
        self.assertIsNone(merged[0]["gpu_memory_mib"])

    def test_new_attributed_gpu_process_is_contaminated(self) -> None:
        baseline = engine(10, "vmmemWSL.exe", 20)
        foreign = engine(20, "browser.exe", 5)
        telemetry = [sample(index, [baseline]) for index in range(1, 6)]
        telemetry.extend([sample(6, [baseline, foreign]), sample(7, [baseline])])
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result["classification"], "CONTAMINATED")
        self.assertEqual(result["reasons"][0]["reason"], "new_gpu_process_activity")

    def test_unknown_process_values_are_not_treated_as_zero_or_activity(self) -> None:
        owned = engine(10, "vmmemWSL.exe", 10)
        unknown = engine(20, "renderer.exe", None)
        telemetry = [sample(index, [owned]) for index in range(1, 6)]
        telemetry.extend([sample(6, [owned, unknown]), sample(7, [owned])])
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result["classification"], "CLEAN")

    def test_forbidden_baseline_process_activity_is_contaminated(self) -> None:
        browser_idle = engine(20, "chrome.exe", 0.0)
        browser_active = engine(20, "chrome.exe", 4)
        telemetry = [sample(index, [browser_idle]) for index in range(1, 6)]
        telemetry.extend(
            [sample(6, [browser_active]), sample(7, [browser_idle])]
        )
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result["classification"], "CONTAMINATED")
        self.assertEqual(result["reasons"][0]["reason"], "forbidden_process_activity")

    def test_telemetry_gap_is_error(self) -> None:
        owned = engine(10, "vmmemWSL.exe", 10)
        telemetry = [sample(index, [owned]) for index in (1, 2, 3, 5, 6, 7)]
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result, {"classification": "ERROR", "reasons": ["telemetry_sequence_gap"]})

    def test_windows_gpu_counter_failure_is_error(self) -> None:
        owned = engine(10, "vmmemWSL.exe", 10)
        telemetry = [sample(index, [owned]) for index in range(1, 8)]
        telemetry[5]["gpu_engine_inventory"] = []
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(
            result,
            {"classification": "ERROR", "reasons": ["windows_gpu_engine_unavailable"]},
        )

    def test_baseline_idle_external_process_becoming_active_is_contaminated(self) -> None:
        owned = engine(10, "vmmemWSL.exe", 10)
        external_idle = engine(30, "helper.exe", 0.0)
        external_active = engine(30, "helper.exe", 0.2)
        telemetry = [sample(index, [owned, external_idle]) for index in range(1, 6)]
        telemetry.extend(
            [sample(6, [owned, external_active]), sample(7, [owned, external_idle])]
        )
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result["classification"], "CONTAMINATED")
        self.assertEqual(
            result["reasons"][0]["reason"],
            "baseline_idle_process_became_active",
        )

    def test_high_total_utilization_without_attribution_is_clean(self) -> None:
        owned = engine(10, "vmmemWSL.exe", 25)
        telemetry = [sample(index, [owned], 99.0) for index in range(1, 8)]
        result = recompute_guard(telemetry, self.boundary, self.guard)
        self.assertEqual(result["classification"], "CLEAN")

    def test_pa_instability_is_not_a_validity_classification(self) -> None:
        self.assertEqual(
            classify_trial(
                scenario_status="formal",
                runtime_error=False,
                guard_classification="CLEAN",
                measurement_valid=True,
            ),
            "VALID",
        )
        self.assertEqual(
            classify_trial(
                scenario_status="formal",
                runtime_error=False,
                guard_classification="CONTAMINATED",
                measurement_valid=True,
            ),
            "CONTAMINATED",
        )

    def test_invalid_measurement_is_an_error(self) -> None:
        self.assertEqual(
            classify_trial(
                scenario_status="formal",
                runtime_error=False,
                guard_classification="CLEAN",
                measurement_valid=False,
            ),
            "ERROR",
        )

    def test_slot_attempts_cannot_be_cherry_picked(self) -> None:
        accepted = [
            {
                "slot_id": f"slot-{index}",
                "slot_attempt": 1,
                "scenario": "latency" if index < 8 else "throughput",
            }
            for index in range(16)
        ]
        accepted[0]["slot_attempt"] = 3
        contaminated = [
            {"slot_id": "slot-0", "slot_attempt": 1, "scenario": "latency"}
        ]
        errors = _slot_numbering_errors(accepted, contaminated, 3)
        self.assertTrue(any("cherry-picks" in error for error in errors))

    def test_non_formal_scenario_cannot_consume_replacement(self) -> None:
        accepted = [
            {
                "slot_id": f"slot-{index}",
                "slot_attempt": 1,
                "scenario": "latency" if index < 8 else "throughput",
            }
            for index in range(16)
        ]
        errors = _slot_numbering_errors(
            accepted,
            [{"slot_id": "slot-0", "slot_attempt": 1, "scenario": "dynamic_batching"}],
            3,
        )
        self.assertTrue(any("non-formal scenario" in error for error in errors))

    def test_contaminated_attempt_cannot_enter_aggregate_slots(self) -> None:
        accepted = [
            {
                "slot_id": f"slot-{index}",
                "slot_attempt": 1,
                "scenario": "latency" if index < 8 else "throughput",
                "classification": "VALID",
            }
            for index in range(16)
        ]
        accepted[4]["classification"] = "CONTAMINATED"
        errors = _slot_numbering_errors(accepted, [], 3)
        self.assertTrue(any("entered formal aggregates" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
