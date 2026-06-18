# Overview

The MELD orchestrator is designed to be programming-language- and machine-learning-framework-agnostic. Communication between the orchestrator and the inference runtime is performed exclusively through the directories `/input/` and `/output/`, which form the interface between both components.

The orchestrator is responsible for

- preparing input data,
- managing the execution lifecycle,
- collecting results and logs, and
- archiving execution artifacts.

The inference runtime is responsible solely for executing inference. It is contained and its access is limited to the container itself.

* * *

# Define the Contract

The contract describes the inference runtime and its input and output schemas.

```
inference:
  name: "my-inference-runtime"    # descriptive short name
  description: ""                 # human-readable description
  version: ""                     # contract version

runtime:
  framework: "tensorflow"         # machine learning framework used

  image:
    # image reference: my-registry/my-org/my-image:0.1.0
    name: "my-registry/my-org/my-image"
    tag: "0.1.0"
    digest: "sha256:..."          # currently unused

  environment_variables:
    TF_CPP_MIN_LOG_LEVEL: "3"

input_schema:
  temporal_scope:
    type: "relative"              # currently only relative scopes are supported
    value: "-P1M"                 # ISO 8601 duration
    anchor: "2026-04-23T13:56:55.525240"

    # resulting time window:
    # start = 2026-03-23T13:56:55.525240
    # end   = 2026-04-23T13:56:55.525240

  features:
    - name: "my-feature"
      datatype: "string"          # reserved for future schema validation

  query:
    type: "sql"
    digest: "sha256:..."          # currently unused

output_schema:
  format: "csv"                   # currently only CSV is supported

  predictor:
    - name: "admitted"
      datatype: "integer"         # reserved for future schema validation
```

* * *

# Write the Query

A SQL query is required to retrieve data from the AKTIN DWH.

The query must contain the parameters `:start` and `:end`, which define the temporal window used for data extraction.

Example:

```
SELECT field AS my-feature
FROM mytable
WHERE timestamp BETWEEN :start AND :end;
```

The values of `:start` and `:end` are derived from the `temporal_scope` section of the contract.

The query must return a result set whose column names correspond to the feature names defined in `input_schema.features`.

* * *

# Implement the Inference Logic

Before starting the container, the MELD orchestrator copies

- the query result set to `/input/input.csv`, and
- the contract to `/input/contract.yaml`.

The inference logic should

1.  load `contract.yaml`,
2.  load `input.csv`,
3.  perform inference, and
4.  write predictions to `/output/`.

* * *

# Create the Container Image

## Folder Structure

The inference runtime image **must** provide the directories

- `/input/`
- `/output/`

These directories form the interface between the orchestrator and the runtime.

## Container Entrypoint

Inference execution is initiated by the MELD orchestrator. Therefore, the inference logic must start automatically when the container is started.

This can be achieved by defining an appropriate `ENTRYPOINT` or `CMD` instruction.

### Example Dockerfile

```
FROM python:3.12-slim

LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.title="MELD runtime image example decision forest"
LABEL org.opencontainers.image.description="Example runtime inference container for MELD. The included model was trained using artificially synthesized data."
LABEL org.opencontainers.image.source="https://github.com/aktin/MELD"

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=""

# Required by the orchestrator
RUN mkdir -p /input /output

# Model artifact
COPY ./artifact /artifact

# Inference logic and dependencies
COPY ./inference /inference

WORKDIR /inference

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "inference.py"]
```

* * *

# Runtime Lifecycle

For each inference request, the orchestrator performs the following steps:

1.  Create a new container from the inference runtime image.
2.  Copy
    - the query result set to `/input/input.csv`, and
    - the contract to `/input/contract.yaml`.
3.  Start the container.
4.  Wait for the inference logic to complete.
5.  Copy the contents of `/output/`.
6.  Archive results, metadata, and logs.
7.  Destroy the container.

The runtime container is ephemeral and should not assume any persistent state between executions.

* * *

# Logging

The inference runtime should write informational messages to stdout and warnings or errors to stderr.

The MELD orchestrator continuously reads both streams and stores them as execution logs. These logs are included in the execution archive and may be used for debugging, monitoring, and auditing purposes.

Runtime output is treated as diagnostic information only. The runtime should not assume that stdout or stderr messages are interpreted semantically by the orchestrator.

* * *

# Example Project Layout

```
project/
├── artifact/
│   └── model.keras
├── inference/
│   ├── inference.py
│   └── requirements.txt
├── resources/
│	├── contract.yaml
│	└── query.sql
└── build/
    └── Dockerfile
```

You can find example projects in [./examples](https://github.com/aktin/MELD/tree/main/examples)