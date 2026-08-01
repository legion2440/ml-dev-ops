# Shared contracts

This directory is reserved for schemas and data-transfer objects that cross at
least two module boundaries. It must not become a collection of unrelated helper
functions.

`client-model-contracts.json` is a tracked generated projection of the current model
manifest and labels. It contains only client-facing tensor, preprocessing, version,
task, and label data; model artifact paths and build details remain private to the
model repository.
