# MELD Contract Reference

## Top-Level Fields

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `contract` | ✅ | object | `{ ... }` | Metadata describing the inference runtime. |
| `runtime` | ✅ | object | `{ ... }` | Configuration of the inference runtime and container image. |
| `input_schema` | ✅ | object | `{ ... }` | Specification of the required input data. |
| `output_schema` | ✅ | object | `{ ... }` | Specification of the inference output. |

---

# `contract`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `contract.name` | ✅ | string | `"ED Patient Volume Prediction"` | Human-readable name of the inference runtime. |
| `contract.description` | ✅ | string | `"Predicts patient arrivals during the next hour."` | Short description of the inference runtime. |
| `contract.version` | ✅ | string | `"1.0.0"` | Version of the runtime contract. |

---

# `runtime`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `runtime.framework` | ✅ | string | `"TensorFlow Decision Forests"` | Machine learning framework used by the runtime. |
| `runtime.image` | ✅ | object | `{ ... }` | OCI/Docker image reference. |
| `runtime.environment_variables` | ❌ | object | `{ TF_CPP_MIN_LOG_LEVEL: "3" }` | Environment variables passed to the runtime container. |

---

# `runtime.image`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `runtime.image.name` | ✅ | string | `"ghcr.io/aktin/meld-runtime"` | OCI/Docker image repository. |
| `runtime.image.tag` | ✅ | string | `"1.0.0"` | OCI image tag. |
| `runtime.image.digest` | ✅ | string | `"sha256:8b2f1d6f..."` | OCI image digest uniquely identifying the image. |

---

# `input_schema`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `input_schema.temporal_scope` | ✅ | object | `{ ... }` | Defines the temporal window used for data extraction. |
| `input_schema.features` | ✅ | array | `[ ... ]` | List of input features expected by the inference runtime. |
| `input_schema.query` | ✅ | object | `{ ... }` | SQL query executed by the orchestrator. |

---

# `input_schema.temporal_scope`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `type` | ✅ | string | `"relative"` | Defines whether the temporal scope is relative or absolute. |

## Relative Temporal Scope (`type = relative`)

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `value` | ✅ | string (ISO-8601 duration) | `"P30D"` | Relative duration subtracted from the anchor timestamp. |
| `anchor` | ✅ | string (ISO-8601 date-time) | `"2026-05-01T12:00:00Z"` | End of the observation window. |

## Absolute Temporal Scope (`type = absolute`)

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `start` | ✅ | string (ISO-8601 date-time) | `"2026-04-01T00:00:00Z"` | Beginning of the observation window. |
| `end` | ✅ | string (ISO-8601 date-time) | `"2026-05-01T00:00:00Z"` | End of the observation window. |

---

# `input_schema.features`

Each feature entry has the following structure.

| Field | Required | Type | Example | Description                                                                         |
|------|:--------:|------|---------|-------------------------------------------------------------------------------------|
| `name` | ✅ | string | `"temperature"` | Name of the feature. Must match the corresponding column returned by the SQL query. |
| `datatype` | ✅ | string | `"Float64"` | Expected datatype of the feature. Based on pandas datatypes.                        |
| `required` | ❌ | boolean | `true` | Indicates whether the feature is mandatory.                       |

### Supported Datatypes

| Value | Description |
|------|-------------|
| `Int64` | 64-bit signed integer |
| `Float64` | 64-bit floating-point number |
| `boolean` | Boolean value |
| `string` | UTF-8 encoded string |
| `datetime64[ns]` | Date and time |
| `timedelta64[ns]` | Time interval |
| `category` | Categorical value |
| `object` | Generic object |

---

# `input_schema.query`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `type` | ✅ | string | `"sql"` | Query language. Currently only SQL is supported. |
| `statement` | ✅ | string | `"SELECT ... WHERE timestamp BETWEEN :start AND :end"` | SQL statement executed against the AKTIN DWH. The placeholders `:start` and `:end` are substituted automatically by the orchestrator. |

---

# `output_schema`

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `type` | ✅ | string | `"csv"` | Output file format. Currently only CSV is supported. |
| `predictor` | ✅ | array | `[ ... ]` | List of prediction columns written by the inference runtime. |


---

# `output_schema.predictor`

Each predictor entry has the following structure.

| Field | Required | Type | Example | Description |
|------|:--------:|------|---------|-------------|
| `name` | ✅ | string | `"predicted_patient_count"` | Name of the prediction column written to the output file. |
| `datatype` | ✅ | string | `"Float64"` | Datatype of the prediction. Reserved for future schema validation. |
| `required` | ❌ | boolean | `false` | Indicates whether the predictor is mandatory. Currently unused. |

---

# Enumerations

## `input_schema.temporal_scope.type`

| Value | Description |
|------|-------------|
| `relative` | The observation window is computed from an anchor timestamp and an ISO-8601 duration. |
| `absolute` | The observation window is explicitly defined by start and end timestamps. |

---

## `input_schema.query.type`

| Value | Description |
|------|-------------|
| `sql` | SQL statement executed by the orchestrator. |

---

## `output_schema.type`

| Value | Description |
|------|-------------|
| `csv` | Prediction results are written as a CSV file. |

---

## `feature.datatype`

| Value | Description |
|------|-------------|
| `Int64` | 64-bit signed integer (`pandas.Int64Dtype`) |
| `Float64` | 64-bit floating-point number |
| `boolean` | Boolean value |
| `string` | UTF-8 encoded string |
| `datetime64[ns]` | Date and time value |