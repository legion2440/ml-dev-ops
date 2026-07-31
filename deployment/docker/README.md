# Container build definitions

`Dockerfile` is a minimal wrapper around the official Triton image selected through
the required `TRITON_IMAGE` build argument. Triton server arguments belong only to
`docker-compose.yml`, and `tritonserver` remains PID 1.
