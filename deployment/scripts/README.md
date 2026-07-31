# Deployment scripts

The host-side lifecycle entrypoints are:

- `run_environment.sh`: validate prerequisites and start all services;
- `run_triton.sh`: start only Triton;
- `check_environment.sh`: show Compose state and run infrastructure smoke checks;
- `capture_runtime_evidence.py`: save sanitized runtime evidence after a successful
  smoke run;
- `stop_environment.sh`: stop services while preserving named volumes;
- `stop_environment.sh --purge`: explicitly remove Compose named volumes.

All scripts resolve the repository root from their own location and share the
Compose command in `compose_common.sh`. They pass `.env` or `.env.example` to
Compose with `--env-file`; environment files are never executed as shell code.

Committed evidence is validated without a running daemon:

```text
python scripts/validate_runtime_evidence.py
```
