# How to Build a MELD-Compatible Inference Runtime

## Prerequisites

Before you start, make sure you have:

- A trained model artifact
- Docker installed
- Access to a container registry

## Project Layout

Structure your project as follows:

```
project/
├── artifact/
│   └── model.keras
├── inference/
│   ├── inference.py
│   └── requirements.txt
├── resources/
│   ├── contract.yaml
│   └── query.sql
└── Dockerfile
```

## Step 1: Write the Model Contract

Create `resources/contract.yaml`. This file tells the orchestrator how to run your runtime and what data to provide. See [the reference](https://github.com/aktin/MELD/blob/main/docs/contract_reference.md) for details.

```yaml
contract:
  name: "Emergency Department Patient Volume Predictor"
  description: >
    Predicts the expected number of emergency department admissions
    during the next 24 hours based on historical patient arrivals and
    weather observations.
  version: "1.0.0"

runtime:
  framework: "TensorFlow Decision Forests"

  image:
    name: "ghcr.io/aktin/meld-ed-volume-predictor"
    tag: "1.0.0"
    digest: "sha256:cb98cb9d0cf99d5fd2c86c3d6b5b6d71d34db8d8d7281d9a1e4e0fd3c5d9d1c4"

  environment_variables:
    TF_CPP_MIN_LOG_LEVEL: "3"
    CUDA_VISIBLE_DEVICES: ""

input_schema:
  temporal_scope:
    type: "relative"
    value: "P30D"
    anchor: "2026-04-23T13:56:55Z"

  features:
    - name: "temperature"
      datatype: "Float64"
      required: true

    - name: "precipitation"
      datatype: "Float64"
      required: true

    - name: "weekday"
      datatype: "string"
      required: true

    - name: "holiday"
      datatype: "boolean"
      required: false

    - name: "patient_count"
      datatype: "Int64"
      required: true

  query:
    type: "sql"
    statement: |
      SELECT
          weather.temperature,
          weather.precipitation,
          calendar.weekday,
          calendar.holiday,
          visits.patient_count
      FROM meld.training_dataset
      WHERE timestamp BETWEEN :start AND :end
      ORDER BY timestamp;

output_schema:
  type: "csv"

  predictor:
    - name: "predicted_patient_count"
      datatype: "Float64"
```

The `temporal_scope` defines the time window for data retrieval. With `type: relative`, the orchestrator calculates:
- `start = anchor + value` (e.g. anchor minus 1 month)
- `end = anchor`

The feature names in `input_schema.features` must match the column names returned by your SQL query.

## Step 2: Write the SQL Query

Write an SQL query for the Aktin DWH. The result of the query is the input to your inference logic.

Requirements:
- Use PostgreSQL syntax.
- Include `:start` and `:end` parameters — the orchestrator substitutes the values from `temporal_scope`.
- Return column names that match the feature names defined in `input_schema.features`.

Save the query in the contract file under `input_schema.query.statement`.

```sql
SELECT field AS my-feature
FROM mytable
WHERE timestamp BETWEEN :start AND :end;
```

## Step 3: Implement the Inference Logic

When the orchestrator starts your container, it provides:

- `/input/input.csv` — the SQL query result set
  - the first row is the header
  - the remaining rows are the data
- `/input/contract.yaml` — your contract file

Your inference script must:

1. Load `/input/contract.yaml`
2. Load `/input/input.csv`
3. Run inference
4. Write all results to `/output/output.csv`
5. Exit

**Exit codes:**
- `0` — execution succeeded
- Non-zero — execution failed; the orchestrator records this as a failed run

Do not write to `/input/`. Do not rely on any state from a previous execution — the container is destroyed after each run.

```python
import yaml
import pandas as pd

if __name__ == "__main__":
    with open("/input/contract.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv("/input/input.csv")

    # ... run your model ...

    result.to_csv("/output/output.csv", index=False)
```

## Step 4: Write the Dockerfile

Your container must start the inference script automatically when launched. The orchestrator does not call anything inside the container — it only starts it.

Requirements:
- Define `ENTRYPOINT` or `CMD` so inference starts on container launch.
- Set `PYTHONUNBUFFERED=1` (or equivalent) to ensure logs are flushed in real time.
- Do not create `/input/` or `/output/` in the Dockerfile — the orchestrator creates these before the container starts.

```Dockerfile
FROM python:3.12-slim

LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.title="MELD runtime image example decision forest"
LABEL org.opencontainers.image.description="Example runtime inference container for MELD."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=""

# Model artifact
COPY ./artifact /artifact

# Inference logic and dependencies
COPY ./inference /inference

WORKDIR /inference

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "inference.py"]
```

## Step 5: Build and Push the Image

```bash
docker build -t my-registry/my-org/my-image:0.1.0 .
docker push my-registry/my-org/my-image:0.1.0
```

The `image.name` and `image.tag` in your contract must match the pushed image exactly.

## Logging

Write informational messages to `stdout` and errors or warnings to `stderr`. The orchestrator captures both streams continuously and includes them in the execution archive.

The orchestrator does not interpret log messages semantically — they are diagnostic only.

```python
import sys

print("Loading model...")                     # stdout
print("Error: missing feature", file=sys.stderr)  # stderr
```

> **@TODO** Structured log format — not yet defined.
> **@TODO** Progress reporting — not yet defined.

## Model Card

The model card is a markdown file that describes the model and its performance. It has no formal structure, but it should contain the following sections:

* Model description
* Training data
* Model performance
* Model architecture
* Model parameters
* Model evaluation
* Model limitations
* Model references
* Model citations
* Model acknowledgements

The card should be stored in the Git repository alongside the model artifact. If you are using GHCR.io and Github, you can connect the published image and the repository using the `org.opencontainers.image.source` label. Then, the card should be displayed on the runtime image's homepage.

For more information, see [Google Model Cards](https://deepmind.google/models/model-cards/).

## Reference: Orchestrator Lifecycle

For each inference request, the orchestrator:

1. Executes the SQL query against the AKTIN DWH.
2. Creates a new container from your image.
3. Creates `/input/` and `/output/` inside the container.
4. Copies `input.csv` and `contract.yaml` to `/input/`.
5. Starts the container.
6. Monitors the container until it terminates.
7. Reads `/output/output.csv` and archives results, logs, and metadata.
8. Destroys the container.

If the container exits with a non-zero code, the orchestrator records the execution as failed. The contents of `/output/` are still archived for debugging.

> **@TODO** Timeout and cancellation behavior — not yet defined.
