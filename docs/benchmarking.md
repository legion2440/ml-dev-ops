# Benchmarking

## Current status

No performance measurements have been collected.

## Comparison contract

The benchmark will compare ResNet50 ONNX with ResNet50 TensorRT under identical
images, input shapes, batch sizes, concurrency, warmup, request counts, GPU
hardware, and container versions.

Metrics will include average, p50, p95, and p99 latency; throughput; error count;
GPU utilization and memory; server queue time; and compute time.

Raw measurements, aggregate CSV files, commands, environment details, and the final
comparison belong in `benchmarks`. The report must state honestly when optimization
does not improve a tested configuration.
