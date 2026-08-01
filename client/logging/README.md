# Inference logging

`writer.py` validates and appends one JSON object per Triton request using
`schemas/inference-event.schema.json`. Append mode preserves prior history and error
events are retained for debugging. Logged names are basenames; raw tensors, secrets,
and absolute host paths are excluded.

`csv_export.py` creates a deterministic derived CSV and never acts as an editable
source of truth. Export operational history with:

```text
python client/inference_client.py export-logs --input-log logs/inference.jsonl --output-csv logs/inference.csv
```
