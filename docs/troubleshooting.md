# Troubleshooting

## Architecture validation

If a generated graph is stale, run:

```text
python scripts/generate_dependency_graph.py
```

If Make is unavailable, use the direct Python commands documented in `README.md`.

An `implemented` path must exist. A `generated` path must exist and its generator
must pass `--check`. A `planned` path may be absent.

## Runtime troubleshooting

GPU, Triton, Docker, model export, inference, and monitoring troubleshooting will be
added with their implementation scopes. No runtime behavior is claimed in step 1.
