# Triton serving

Triton startup policy is implemented in `docker-compose.yml` and exposed through
`deployment/scripts/run_triton.sh`.

The server uses explicit model control, publishes HTTP, gRPC, and metrics endpoints,
and mounts `models` read-only. Step 2 expects an empty repository and does not claim
working inference. Model files and `config.pbtxt` ownership remain under `models`.
