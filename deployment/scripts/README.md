# Deployment scripts

The host-side lifecycle entrypoints are:

- `run_environment.sh`: validate prerequisites and start all services;
- `run_triton.sh`: start only Triton;
- `check_environment.sh`: show Compose state and run infrastructure smoke checks;
- `stop_environment.sh`: stop services while preserving named volumes;
- `stop_environment.sh --purge`: explicitly remove Compose named volumes.

All scripts resolve the repository root from their own location and share the
Compose command in `compose_common.sh`. They pass `.env` or `.env.example` to
Compose with `--env-file`; environment files are never executed as shell code.
