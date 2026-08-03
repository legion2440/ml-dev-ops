from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.model_preparation import prepare_models
from scripts.validate_model_repository import (
    validate_artifact_inventory,
    validate_spec_semantics,
)


class ArtifactStalenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prepare_models.load_spec()
        self.manifest = json.loads(prepare_models.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_missing_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            errors: list[str] = []
            validate_artifact_inventory(self.manifest, errors, Path(directory))
        self.assertEqual(
            sum("missing" in error for error in errors),
            len(prepare_models.serving_artifact_paths(self.spec)),
        )

    def test_gpu_query_accepts_selected_89_and_86_devices(self) -> None:
        for capability in ("8.9", "8.6"):
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    f"GPU-{capability}, NVIDIA Test GPU, 610.88, {capability}\n"
                ),
                stderr="",
            )
            with patch.object(prepare_models, "_run", return_value=completed) as run:
                gpu = prepare_models._gpu_query(self.spec, "0")
            self.assertEqual(gpu["compute_capability"], capability)
            self.assertIn("device=0", run.call_args.args[0])

    def test_gpu_query_ignores_container_entrypoint_banner(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                "\n=====================\n== NVIDIA TensorRT ==\n"
                "Copyright (c) NVIDIA Corporation\n"
                "GPU-selected, NVIDIA Test GPU, 610.88, 8.6\n"
            ),
            stderr="",
        )
        with patch.object(prepare_models, "_run", return_value=completed):
            gpu = prepare_models._gpu_query(self.spec, "0")
        self.assertEqual(gpu["uuid"], "GPU-selected")
        self.assertEqual(gpu["compute_capability"], "8.6")

    def test_build_uses_one_selected_gpu_for_every_container_call(self) -> None:
        gpu = {
            "uuid": "GPU-selected",
            "name": "NVIDIA Selected GPU",
            "driver_version": "610.88",
            "compute_capability": "8.6",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / ".cache/model-preparation/tensorrt-build.json"
            calls: list[list[str]] = []

            def run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(arguments)
                if "nvidia-smi" in arguments:
                    output = ", ".join(gpu.values()) + "\n"
                elif "--version" in arguments:
                    output = "TensorRT v11.1\n"
                elif "python" in arguments and "-c" in arguments:
                    output = '{"cuda_version":"13.0","tensorrt_version":"11.1.0"}\n'
                else:
                    output = ""
                save_argument = next(
                    (item for item in arguments if item.startswith("--saveEngine=")),
                    None,
                )
                if save_argument:
                    engine = root / save_argument.removeprefix("--saveEngine=/workspace/")
                    engine.parent.mkdir(parents=True, exist_ok=True)
                    engine.write_bytes(b"portable-plan")
                return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

            with (
                patch.object(prepare_models, "REPOSITORY_ROOT", root),
                patch.object(prepare_models, "BUILD_RECORD_PATH", record_path),
                patch.object(prepare_models, "_run", side_effect=run),
                patch.object(prepare_models, "build_exporter_image"),
                patch.object(prepare_models, "_run_exporter"),
                patch.object(prepare_models, "_bind_parity_contract"),
            ):
                prepare_models.build_tensorrt(self.spec, " GPU-selected ")

            docker_runs = [call for call in calls if call[:2] == ["docker", "run"]]
            self.assertGreaterEqual(len(docker_runs), 4)
            self.assertTrue(
                all("device=GPU-selected" in call for call in docker_runs), docker_runs
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["gpu_selector"], "GPU-selected")
            self.assertEqual(record["gpu"], gpu)
            self.assertEqual(
                record["engine"]["path"],
                "models/resnet50_tensorrt/1/model.plan",
            )

    def test_host_provenance_does_not_change_shared_contracts(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["generated_at_utc"] = "2099-01-01T00:00:00Z"
        changed["build"]["gpu"] = {
            "uuid": "GPU-portable",
            "name": "NVIDIA Alternate GPU",
            "driver_version": "999.0",
            "compute_capability": "8.6",
        }
        changed["models"]["resnet50_tensorrt"]["compute_capability"] = "8.6"
        changed["models"]["resnet50_tensorrt"]["versions"]["1"]["artifact"][
            "sha256"
        ] = "0" * 64
        parity = changed["artifact_validation"]["tensorrt"]["parity"]
        parity["max_abs_error"] = 0.25
        parity["mean_abs_error"] = 0.025
        parity["cosine_similarity"] = 0.9995

        self.assertEqual(
            prepare_models.render_client_contract(changed),
            prepare_models.render_client_contract(self.manifest),
        )
        self.assertEqual(
            prepare_models.render_benchmark_pair_contract(changed),
            prepare_models.render_benchmark_pair_contract(self.manifest),
        )

    def test_contract_change_makes_generated_config_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["models"]["resnet50"]["output"]["shape"][-1] += 1
        changed["models"]["resnet50"]["labels"]["count"] += 1
        self.assertEqual(validate_spec_semantics(changed), [])
        generated = prepare_models.render_config(changed, "resnet50_onnx")
        tracked = (prepare_models.REPOSITORY_ROOT / "models/resnet50_onnx/config.pbtxt").read_text(
            encoding="utf-8"
        )
        self.assertNotEqual(generated, tracked)
        manifest_errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any(".output" in error for error in manifest_errors))

    def test_profile_change_makes_generated_data_stale(self) -> None:
        changed = copy.deepcopy(self.spec)
        resnet = changed["models"]["resnet50"]
        serving = resnet["serving"]
        new_max_batch = serving["tensorrt"]["profile"]["max"][0] + 1
        serving["tensorrt"]["profile"]["max"][0] = new_max_batch
        serving["tensorrt"]["max_batch_size"] = new_max_batch
        serving["onnx"]["max_batch_size"] = new_max_batch

        self.assertEqual(validate_spec_semantics(changed), [])
        generated = prepare_models.render_config(changed, serving["tensorrt"]["name"])
        tracked = (
            prepare_models.REPOSITORY_ROOT / serving["tensorrt"]["config_path"]
        ).read_text(encoding="utf-8")
        self.assertNotEqual(generated, tracked)
        manifest_errors = prepare_models.manifest_staleness(changed, self.manifest)
        self.assertTrue(any("profile" in error for error in manifest_errors))


if __name__ == "__main__":
    unittest.main()
