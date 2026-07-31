# Client and inference logging

## Current status

The inference client and structured history writer are planned.

## Client contract

The client will accept one file or a directory, preprocess by model, call Triton
over REST or gRPC, handle batches and timeouts, select explicit model versions, and
display classes, confidence values, or bounding boxes.

Planned commands include classification, detection, health, metadata, version
listing, model loading, and model unloading.

## Logging contract

Each inference event will include a timestamp, request ID, model and version,
protocol, input name, batch size, preprocessing time, server request time, total
latency, status, prediction summary, and error text.

JSONL is the primary history format. CSV is a derived export for offline analysis.
