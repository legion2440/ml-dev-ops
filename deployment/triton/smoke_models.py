"""Validate immutable step 3 runtime evidence without contacting Triton."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the immutable step 3 evidence snapshot",
    )
    arguments = parser.parse_args()
    if not arguments.check:
        parser.error(
            "step 3 runtime evidence is immutable; use --check or run "
            "deployment/triton/verify_serving.py for current runtime verification"
        )
    process = subprocess.run(
        [sys.executable, "scripts/validate_model_evidence.py"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
