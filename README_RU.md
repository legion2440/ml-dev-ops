# ML DevOps с NVIDIA Triton

Production-style система компьютерного зрения на NVIDIA Triton Inference Server: ONNX и TensorRT модели, воспроизводимый Docker deployment, benchmark, Prometheus metrics, Grafana dashboard, GPU monitoring, постоянные inference-логи и явное управление версиями моделей.

Репозиторий обслуживает классификацию ResNet50 и детекцию YOLO11n, сравнивает ONNX baseline с FP16 TensorRT optimization, поддерживает inference по HTTP и gRPC и хранит committed runtime evidence по основным требованиям задания.

· [English version](README.md)

## 📋 Содержание

- [🚀 Быстрый старт](#-быстрый-старт)
- [📝 О проекте](#-о-проекте)
- [✨ Возможности](#-возможности)
- [🔄 Архитектура](#-архитектура)
- [🧠 Модели](#-модели)
- [⚡ Triton serving](#-triton-serving)
- [📈 Benchmark](#-benchmark)
- [📊 Monitoring](#-monitoring)
- [🖼️ Python client и samples](#️-python-client-и-samples)
- [🧾 Логирование и CSV export](#-логирование-и-csv-export)
- [🔎 Runtime evidence](#-runtime-evidence)
- [🧰 Технологии](#-технологии)
- [🧪 Тесты и проверки](#-тесты-и-проверки)
- [📁 Структура проекта](#-структура-проекта)
- [⚠️ Примечания](#️-примечания)
- [🧑‍💻 Автор](#-автор)

## 🚀 Быстрый старт

### Требования

- Python `3.10+`
- Docker Engine с Docker Compose v2
- NVIDIA GPU с совместимым драйвером
- NVIDIA Container Toolkit
- Bash; на Windows поддерживается Git Bash
- Docker Desktop с WSL2 backend для GPU-контейнеров на Windows

Образы Triton, TensorRT и model exporter занимают много места. Для Docker желательно иметь несколько десятков гигабайт свободного диска.

### Клонирование и установка

```bash
git clone https://github.com/legion2440/ml-dev-ops.git
cd ml-dev-ops

python -m pip install -r requirements.txt
```

`.env.example` — каноническая конфигурация. Локальный `.env` необязателен и игнорируется Git.

Проверка конфигурации чистого checkout:

```bash
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

### Подготовка моделей

```bash
python scripts/model_preparation/prepare_models.py prepare
```

Команда создаёт локальные model artifacts для Triton. Бинарники моделей воспроизводимы и намеренно не коммитятся в репозиторий.

### Запуск стека

```bash
bash deployment/scripts/run_environment.sh
```

Проверка окружения:

```bash
bash deployment/scripts/check_environment.sh
python deployment/scripts/smoke_environment.py
```

### Inference

Health:

```bash
python client/inference_client.py health
```

Классификация ResNet50:

```bash
python client/inference_client.py classify client/samples/01_dog.jpg
```

TensorRT classification по gRPC:

```bash
python client/inference_client.py classify client/samples/ \
    --model resnet50_tensorrt \
    --protocol grpc \
    --batch-size 4
```

YOLO detection:

```bash
python client/inference_client.py detect client/samples/ \
    --protocol http \
    --batch-size 2
```

Остановка:

```bash
bash deployment/scripts/stop_environment.sh
```

Если установлен GNU Make, те же сценарии доступны через `make prepare-models`, `make up`, `make verify-serving`, `make verify-monitoring`, `make benchmark` и `make validate`.

## 📝 О проекте

Это не отдельный model script, а полный ML inference pipeline вокруг NVIDIA Triton.

Model repository содержит versioned serving contracts для ResNet50 и YOLO11n. Переиспользуемый Python client выполняет preprocessing, отправляет запросы по Triton HTTP или gRPC, декодирует predictions и записывает одно структурированное событие на каждый request. Метрики Triton и GPU собираются Prometheus и отображаются на provisioned Grafana dashboard.

Формальный benchmark сравнивает одну и ту же ResNet50 workload в ONNX Runtime и TensorRT. Historical runtime evidence отделён от проверки current semantic compatibility, поэтому последующие изменения документации или monitoring не переписывают исходный benchmark run.

Код, конфигурация, generated contracts, runtime evidence и audit documentation проверяются независимо.

## ✨ Возможности

### Model serving

- NVIDIA Triton Inference Server в Docker;
- explicit model control;
- read-only mount model repository;
- ResNet50 classification;
- YOLO11n object detection;
- ONNX Runtime и TensorRT backends;
- HTTP и gRPC inference;
- явный выбор версии модели;
- load, unload, reload и default-version behavior;
- dynamic batching с runtime evidence.

### Model optimization

- ResNet50 ONNX FP32 baseline;
- TensorRT FP16 compute с FP32 public I/O;
- одинаковые source weights у baseline и optimized вариантов;
- parity checks перед публикацией benchmark;
- deterministic benchmark input;
- committed baseline, optimized, comparison, raw и report artifacts.

### Observability

- Triton Prometheus metrics;
- DCGM GPU metrics;
- provisioned Grafana datasource;
- provisioned dashboard `ML DevOps Inference`;
- inference throughput;
- request rate;
- average request latency;
- GPU utilization;
- failed request count;
- Prometheus rules для высокой latency и inference failures.

### Client и evidence

- переиспользуемый Python inference client;
- contract-driven preprocessing и postprocessing;
- 10 tracked real sample images с provenance;
- JSONL inference history;
- deterministic CSV export;
- sanitized committed runtime evidence;
- tamper/staleness validation;
- repository hygiene checks.

## 🔄 Архитектура

```text
                         +----------------------+
                         |  JPG / PNG samples   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Python client     |
                         | preprocess / decode  |
                         +----+------------+----+
                              |            |
                         HTTP |            | gRPC
                              v            v
                    +-----------------------------+
                    | NVIDIA Triton Inference     |
                    | Server                      |
                    |                             |
                    | ResNet50 ONNX v1 / v2       |
                    | ResNet50 TensorRT v1        |
                    | YOLO11n ONNX v1             |
                    +------+----------------------+
                           |
              +------------+-------------+
              |                          |
              v                          v
    +-------------------+       +-------------------+
    | Triton /metrics   |       | JSONL request log |
    +---------+---------+       +---------+---------+
              |                           |
              v                           v
    +-------------------+       +-------------------+
    | Prometheus        |       | CSV export        |
    +---------+---------+       +-------------------+
              |
        +-----+--------------------+
        |                          |
        v                          v
+---------------+          +---------------+
| Grafana       |          | Alert rules   |
| dashboard     |          | latency/error |
+---------------+          +---------------+

DCGM Exporter --------------------> Prometheus --------------------> Grafana
```

Triton работает в explicit model control mode: lifecycle моделей наблюдаем и тестируем, а не скрываем за автоматическим repository scan.

Подробности: `ARCHITECTURE.md` и `docs/generated/dependency-graph.md`.

## 🧠 Модели

| Модель              | Backend      | Версии   | Precision              | Задача                     |
| ------------------- | ------------ | ---:     | ---------------------- | -------------------------- |
| `resnet50_onnx`     | ONNX Runtime | `1`, `2` | FP32                   | ImageNet-1K classification |
| `resnet50_tensorrt` | TensorRT     | `1`      | FP16 compute, FP32 I/O | ImageNet-1K classification |
| `yolo11n_onnx`      | ONNX Runtime | `1`      | FP32                   | COCO object detection      |

ResNet50 ONNX и TensorRT построены из одних source weights. Preparation выбирает одну GPU, собирает и валидирует canonical `model.plan` на ней, а host provenance хранит отдельно от portable model semantics.

В Git хранятся model specifications, Triton configs, labels, source hashes, licenses, generated manifests и runtime evidence. Крупные model binaries остаются локальными.

Подготовка artifacts:

```bash
python scripts/model_preparation/prepare_models.py prepare
```

Structure-only validation без model binaries:

```bash
python scripts/validate_model_repository.py --structure-only
```

Подробнее: `docs/model-preparation.md` и `docs/model-versioning.md`.

## ⚡ Triton serving

Serving layer проверяет:

- server liveness и readiness;
- model metadata и configuration;
- HTTP inference;
- gRPC inference;
- numerical protocol parity;
- explicit model versions;
- default version selection;
- load и unload;
- in-place reload;
- dynamic batching;
- финальную очистку READY state.

Metadata для ResNet50 ONNX v2:

```bash
python client/inference_client.py metadata \
    --model resnet50_onnx \
    --version 2
```

Если модель недоступна, client может загрузить её через Triton repository-control API перед inference.

Immutable Step 4 serving evidence и current GPU-portability proof хранятся раздельно:

```text
docs/evidence/step-4/
docs/evidence/portability/
```

## 📈 Benchmark

Формальный benchmark сравнивает:

```text
baseline:  resnet50_onnx:v1
optimized: resnet50_tensorrt:v1
```

Оба варианта используют одинаковые ResNet50 weights и один public FP32 tensor contract.

### Формальные сценарии

| Сценарий   | Batch | Concurrency | Primary metric      |
| ---------- | ----: | ----------: | ------------------- |
| Latency    | `1`   | `1`         | mean client latency |
| Throughput | `8`   | `4`         | inferences / second |

Опубликованный run использует 4 paired repetitions в сбалансированном порядке:

```text
ONNX -> TensorRT
TensorRT -> ONNX
ONNX -> TensorRT
TensorRT -> ONNX
```

### Результат

| Метрика                              | Результат   |
| ------------------------------------ | ----------: |
| Median paired latency improvement    | **19.32%**  |
| Улучшившихся latency pairs           | **4 / 4**   |
| Median paired throughput improvement | **114.11%** |
| Улучшившихся throughput pairs        | **4 / 4**   |
| Valid formal slots                   | **16 / 16** |

Две попытки с host-activity contamination сохранены в raw evidence вместе с same-slot replacements. Они исключены по заранее заданному environment guard и не нужны для вывода о выигрыше TensorRT.

Запуск benchmark:

```bash
python benchmarks/run_benchmark.py run --env-file .env.example
```

Проверка committed evidence без нового inference:

```bash
python scripts/validate_benchmark_evidence.py --check
```

Проверка только immutable historical run:

```bash
python scripts/validate_benchmark_evidence.py --historical-only
```

Compatibility gate основан не на byte equality всего дерева, а на semantics: из production code выводятся Perf Analyzer command behavior, aggregation behavior, environment-guard classification, replacement behavior, model pair, methodology и benchmark-relevant deployment projection.

Подробнее: `docs/benchmarking.md` и `benchmarks/report.md`.

## 📊 Monitoring

В stack входят:

- Prometheus;
- Grafana;
- NVIDIA DCGM Exporter;
- native metrics Triton.

Локальные endpoints по умолчанию:

| Сервис         | Адрес                           |
| -------------- | ------------------------------- |
| Triton HTTP    | `http://127.0.0.1:8000`         |
| Triton gRPC    | `127.0.0.1:8001`                |
| Triton metrics | `http://127.0.0.1:8002/metrics` |
| Prometheus     | `http://127.0.0.1:9090`         |
| Grafana        | `http://127.0.0.1:3000`         |
| DCGM metrics   | `http://127.0.0.1:9400/metrics` |

UID provisioned Grafana dashboard:

```text
ml-dev-ops-inference
```

Dashboard:

```text
http://127.0.0.1:3000/d/ml-dev-ops-inference/ml-dev-ops-inference
```

Пять основных панелей:

1. inference throughput;
2. request rate;
3. average request latency;
4. GPU utilization;
5. failed requests.

Prometheus загружает два rules:

- `HighInferenceLatency`;
- `InferenceRequestFailures`.

Проверка полного пути Triton -> Prometheus -> Grafana:

```bash
python monitoring/verify_runtime.py --env-file .env.example
```

Verifier создаёт короткую controlled inference workload, проходит минимум два Prometheus scrape intervals, выполняет dashboard queries через Grafana Prometheus datasource proxy, сверяет GPU identity, проверяет alert definitions и восстанавливает исходный READY set.

Подробнее: `docs/monitoring.md`.

## 🖼️ Python client и samples

Client принимает отдельные изображения и директории.

Classification:

```bash
python client/inference_client.py classify client/samples/
```

Явная ONNX версия:

```bash
python client/inference_client.py classify client/samples/01_dog.jpg \
    --model resnet50_onnx \
    --version 2
```

TensorRT по gRPC:

```bash
python client/inference_client.py classify client/samples/ \
    --model resnet50_tensorrt \
    --protocol grpc \
    --batch-size 4
```

YOLO по HTTP:

```bash
python client/inference_client.py detect client/samples/ \
    --protocol http \
    --batch-size 2
```

В `client/samples/` лежат 10 tracked JPG images. Source, license/provenance, dimensions и SHA-256 записаны в `client/samples/manifest.json`.

Runtime client evidence охватывает все 10 samples и оба serving protocols.

Подробнее: `docs/client.md`.

## 🧾 Логирование и CSV export

Каждый Triton request добавляет одно структурированное JSONL событие.

Операционные логи находятся вне committed evidence и игнорируются Git.

Экспорт JSONL history в CSV:

```bash
python client/inference_client.py export-logs \
    --input-log logs/inference.jsonl \
    --output-csv logs/inference.csv
```

Committed Step 5 evidence:

```text
docs/evidence/step-5/inference-log.jsonl
docs/evidence/step-5/inference-log.csv
docs/evidence/step-5/predictions.txt
docs/evidence/step-5/client-runtime.json
```

Reference run содержит успешные classification/detection requests без raw tensors, secrets и host-specific paths.

## 🔎 Runtime evidence

README сам по себе не считается runtime proof. Для основных live-этапов в репозитории лежит machine-checkable evidence.

| Step | Evidence                | Что доказывает                                                                       |
| ---- | ----------------------- | ------------------------------------------------------------------------------------ |
| 2    | `docs/evidence/step-2/` | Docker stack, GPU visibility, service health, Prometheus targets, Grafana datasource |
| 3    | `docs/evidence/step-3/` | model contracts и runtime model smoke                                                |
| 4    | `docs/evidence/step-4/` | HTTP/gRPC serving, batching, versions, lifecycle, cleanup                            |
| 5    | `docs/evidence/step-5/` | real-image client, predictions, JSONL/CSV logging, READY restoration                 |
| 6    | `docs/evidence/step-6/` + `benchmarks/results/` | ONNX vs TensorRT benchmark и raw measurement evidence        |
| 7    | `docs/evidence/step-7/` | Prometheus/Grafana/DCGM data path и загруженные alert rules                          |
| GPU portability | `docs/evidence/portability/` | provenance выбранной GPU/TensorRT сборки, parity-gated manifest и current serving proof |

Step 2 и Step 6 разделяют:

```text
historical integrity
current semantic compatibility
```

Historical runtime snapshot не переписывается только потому, что изменились не связанные с измерением файлы репозитория.

Полная матрица requirement -> evidence:

```text
docs/audit-evidence.md
```

## 🧰 Технологии

| Область               | Технология                              |
| --------------------- | --------------------------------------- |
| Inference server      | NVIDIA Triton Inference Server `2.71.0` |
| Container runtime     | Docker + Docker Compose                 |
| Classification        | ResNet50                                |
| Detection             | YOLO11n                                 |
| Baseline runtime      | ONNX Runtime                            |
| Optimized runtime     | TensorRT                                |
| Client                | Python                                  |
| Protocols             | Triton HTTP и gRPC                      |
| Metrics               | Triton Prometheus metrics               |
| GPU telemetry         | NVIDIA DCGM Exporter                    |
| Metrics storage/query | Prometheus                              |
| Dashboard             | Grafana                                 |
| Benchmark tool        | Triton Perf Analyzer                    |
| Contracts             | JSON / JSON Schema / YAML               |
| Tests                 | Python `unittest`                       |

Версии контейнеров и model/export dependencies закреплены в repository configuration; floating `latest` tags не используются.

## 🧪 Тесты и проверки

Установка dependencies:

```bash
python -m pip install -r requirements.txt
```

С GNU Make:

```bash
make validate
```

Прямой набор проверок:

```bash
python scripts/validate_structure.py
python scripts/validate_module_map.py
python scripts/validate_deployment.py
python scripts/validate_runtime_evidence.py
python scripts/validate_model_repository.py --structure-only
python scripts/validate_model_evidence.py
python scripts/validate_serving.py --structure-only
python scripts/validate_serving_evidence.py
python scripts/validate_client.py
python scripts/validate_client_evidence.py
python scripts/validate_benchmark.py
python scripts/validate_benchmark_evidence.py
python scripts/validate_monitoring.py
python scripts/validate_repository_hygiene.py
python scripts/generate_dependency_graph.py --check
python -m unittest discover -s tests/unit -t . -p "test_*.py"
docker compose --project-directory . --file docker-compose.yml --env-file .env.example config --quiet
```

Validation layer проверяет:

- repository structure;
- module и dependency metadata;
- Docker configuration;
- runtime-evidence integrity;
- model repository structure;
- model и serving evidence;
- client contracts и evidence;
- benchmark arithmetic и raw-data recomputation;
- behavioral compatibility probes;
- monitoring configuration и evidence;
- deterministic generation;
- read-only check modes;
- tracked-file hygiene;
- утечки host paths и secret-like значений в evidence.

`promtool` используется при наличии. Его отсутствие не отменяет обязательные YAML и semantic checks репозитория.

## 📁 Структура проекта

```text
ml-dev-ops/
├── benchmarks/
│   ├── configs/
│   ├── results/
│   ├── aggregate_results.py
│   ├── environment_guard.py
│   ├── report.md
│   └── run_benchmark.py
├── client/
│   ├── logging/
│   ├── samples/
│   ├── inference_client.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── transport.py
├── deployment/
│   ├── docker/
│   ├── scripts/
│   ├── triton/
│   └── runtime_evidence.py
├── docs/
│   ├── evidence/
│   ├── generated/
│   ├── audit-evidence.md
│   ├── benchmarking.md
│   ├── client.md
│   ├── deployment.md
│   ├── model-preparation.md
│   ├── model-versioning.md
│   └── monitoring.md
├── models/
│   ├── resnet50_onnx/
│   ├── resnet50_tensorrt/
│   ├── yolo11n_onnx/
│   ├── model-manifest.json
│   └── model-spec.yaml
├── monitoring/
│   ├── grafana/
│   ├── prometheus/
│   └── verify_runtime.py
├── schemas/
├── scripts/
├── shared/
├── tests/
├── ARCHITECTURE.md
├── dependency-graph.json
├── docker-compose.yml
├── Makefile
├── module-map.json
├── README.md
└── README_RU.md
```

`module-map.json` описывает ownership модулей. `dependency-graph.json` задаёт allowed/forbidden dependencies, а `docs/generated/dependency-graph.md` генерируется из него.

## ⚠️ Примечания

- Model binaries намеренно игнорируются Git и должны быть подготовлены локально.
- TensorRT engines hardware-specific; portable workflow пересобирает `model.plan` на выбранной GPU, а build record и manifest фиксируют host provenance.
- Reference runtime evidence получен на NVIDIA GeForce RTX 4080 Laptop GPU с compute capability `8.9`.
- Runtime evidence доказывает конкретные reference runs, а не гарантирует идентичные performance numbers на любом будущем host.
- Внутренний текст Perf Analyzer о stability — diagnostic, а не benchmark acceptance criterion.
- GPU utilization может корректно быть `0%`; monitoring validity требует числовую серию с правильной GPU identity, а не искусственно ненулевое значение.
- Persistent data Grafana и Prometheus хранится в Docker volumes и не коммитится.
- `.env`, local cache, model binaries, operational logs и runtime junk игнорируются Git.
- Репозиторий использует лицензию `AGPL-3.0-only`, поскольку model-preparation workflow использует open-source Ultralytics YOLO toolchain. Сторонние компоненты сохраняют собственные лицензии; см. `THIRD_PARTY_NOTICES.md`.

## 🧑‍💻 Автор

Nazar Yestayev (@nyestaye)
