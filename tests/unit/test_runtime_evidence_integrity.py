"""Step 2 historical-integrity and compatibility regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deployment import runtime_evidence
from deployment.scripts import capture_runtime_evidence
from scripts import validate_runtime_evidence


class RuntimeEvidenceIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = validate_runtime_evidence.REPOSITORY_ROOT
        self.evidence_directory = (
            self.root / validate_runtime_evidence.EVIDENCE_RELATIVE
        )
        self.integrity = json.loads(
            (self.evidence_directory / "runtime-integrity.json").read_text(
                encoding="utf-8"
            )
        )

    def _copy_evidence(self, target: Path) -> None:
        shutil.copytree(
            self.evidence_directory,
            target / validate_runtime_evidence.EVIDENCE_RELATIVE,
        )

    def test_historical_artifact_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_evidence(root)
            smoke = root / validate_runtime_evidence.EVIDENCE_RELATIVE / "smoke.json"
            smoke.write_text(
                smoke.read_text(encoding="utf-8").replace('"ok": true', '"ok": false', 1),
                encoding="utf-8",
                newline="\n",
            )
            errors = validate_runtime_evidence.validate_historical(root)
        self.assertTrue(any("artifact hashes" in error for error in errors))

    def test_historical_source_manifest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_evidence(root)
            path = root / validate_runtime_evidence.EVIDENCE_RELATIVE / "runtime-integrity.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["runtime_source_hashes"]["docker-compose.yml"] = "0" * 64
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            errors = validate_runtime_evidence.validate_historical(root)
        self.assertTrue(any("source manifest hash" in error for error in errors))

    def test_unrelated_monitoring_evolution_remains_compatible(self) -> None:
        self.assertEqual(
            self.integrity["runtime_source_revision"],
            "de079e1af1653236cb6cae6dd38708f0bfde494d",
        )
        self.assertEqual(
            self.integrity["runtime_source_fingerprint_sha256"],
            "1829acf0227771faaf4e96be64048bc655ed739e31f09b10ff26e34d64acbade",
        )
        self.assertEqual(
            runtime_evidence.compatibility_projection(),
            self.integrity["runtime_compatibility_projection"],
        )

    def test_critical_contract_change_does_not_invalidate_history(self) -> None:
        self.assertEqual(validate_runtime_evidence.validate_historical(), [])
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            for relative in (
                ".env.example",
                "docker-compose.yml",
                "monitoring/prometheus/prometheus.yml",
                "monitoring/grafana/provisioning/datasources/prometheus.yml",
            ):
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / relative, target)
            env_path = source_root / ".env.example"
            env_path.write_text(
                env_path.read_text(encoding="utf-8").replace(
                    "TRITON_HTTP_PORT=8000", "TRITON_HTTP_PORT=8999"
                ),
                encoding="utf-8",
                newline="\n",
            )
            errors = validate_runtime_evidence.validate_current_compatibility(
                self.integrity, source_root
            )
        self.assertTrue(any("incompatible" in error for error in errors))

    def test_check_modes_are_read_only(self) -> None:
        paths = sorted(self.evidence_directory.iterdir())
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(validate_runtime_evidence.validate(), [])
        self.assertEqual(
            validate_runtime_evidence.validate(historical_only=True), []
        )
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(after, before)

    def test_capture_check_never_contacts_runtime_or_writes(self) -> None:
        with (
            patch.object(sys, "argv", ["capture_runtime_evidence.py", "--check"]),
            patch.object(capture_runtime_evidence, "_validate_evidence", return_value=0),
            patch.object(capture_runtime_evidence, "_capture_smoke") as capture_smoke,
            patch.object(capture_runtime_evidence, "_write_atomic") as write_atomic,
        ):
            self.assertEqual(capture_runtime_evidence.main(), 0)
        capture_smoke.assert_not_called()
        write_atomic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
