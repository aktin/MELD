# MELD unit testing

The MELD unit suite covers the implemented contract, execution, runtime, API,
and support code without requiring Docker, PostgreSQL, a registry, or a live
Flask server. External boundaries are mocked, while filesystem behavior uses
temporary directories and real YAML, JSON, tar, and ZIP files where those
formats are part of the behavior being tested.

## Running the suite

From the repository root, run:

```bash
./.venv/bin/python -m unittest discover -s MELD/tests -p 'test*.py' -v
```

The equivalent command from inside `MELD` is:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

`MELD/tests/test_support.py` creates temporary log and database-password
resources before importing application modules. It also supplies harmless
database connection settings. Each test patches contract paths, execution
paths, Docker clients, or database connections as needed, so the commands do
not connect to external services.

## Contract schema and generated models

`MELD/resources/contract.schema.json` is the source of truth for contract
validation. The Python files under `MELD/ModelManager/generated/` are generated
dataclasses that describe the schema's fields and nesting; constructing one of
these dataclasses does not enforce JSON Schema rules.

This distinction matters for conditional rules such as `if`/`then`, `anyOf`,
and `unevaluatedProperties`. Validate contracts through the schema-backed
contract loader when testing acceptance or rejection. Do not add handwritten
changes to generated files.

After changing the schema, regenerate the documentation and models from the
repository root with:

```bash
sh scripts/update_docs.sh
```

The generated human-readable reference is written to
`docs/generated/contract_schema_reference.md`.

## Test case matrix

| Area | Cases covered | Isolation and assertion method |
| --- | --- | --- |
| Contract models | Schema acceptance and rejection; nested typed models; YAML streams; preserved extension fields; name-version IDs and identity equality; digest and tag image references | Real schema validation and round-trip dictionary comparisons |
| Contract service | Empty payloads; persistence and retrieval; ready and installing states; name-version duplicate contracts; failed-installation retry; Docker pull arguments; layer progress aggregation; retry backoff; malformed progress files; digest configuration | Temporary contract directories, mocked image availability and pull workers, progress JSON inspection |
| HTTP contract API | Health and version; route registration; YAML media validation; create, list, retrieve, validate, delete; name-version IDs and `Location` headers; installation progress; conflict and storage errors; Swagger schema | Flask test client with temporary contract storage and mocked services |
| HTTP execution API | Start, list, retrieve, cancel, and logs; running status; successful ZIP archive; final-state log reads; streaming logs; not-found and storage errors | Flask test client, mocked execution service, in-memory ZIP streams and generators |
| Execution context | Folder creation; initial `PENDING`; persisted status writes and reloads; image references; cancellation events and callbacks; context provider access | Temporary roots and mocked contracts/loggers |
| Execution service | Background execution submission; future cleanup; matching and nonmatching cancellation; execution listing IDs; result archives; completed log reads; streaming log lines | Temporary execution trees, futures, mocked inference and context creation |
| Manager | Absolute and relative time windows; invalid windows; query success and failure states; row metrics; feature validation and normalization; ZIP metrics; inference orchestration; runtime image pull/remove | DataFrames, mocked query/runtime functions, temporary archives |
| Inference runner | Interface directory tar creation; input and contract archive upload; result extraction and missing output; CSV conversion; output verification metrics; successful runtime flow; cancellation before container start; metadata-only failure archive | Mocked containers and runtime functions, real tar/ZIP buffers, temporary job folders |
| Docker runtime | Lazy import; authenticated and unauthenticated pulls; streamed progress and pull errors; image existence; image deletion; container create/start/stop/remove; image size; stdout/stderr routing; success, failure, and timeout waits | Mock Docker client and Docker exceptions; no daemon connection |
| Monitor and data loader | Contract and runtime metric metadata; timer start/stop timestamps; SQL text, parameters, and dtype mapping | Mock context/provider, mocked SQLAlchemy connection and pandas query |
| Utilities and validation | YAML stream/path loading; invalid paths; YAML serialization; dictionary and model feature definitions; required, unexpected, and incompatible columns; contract file loading | Temporary files and pandas DataFrames |
| Logger | Traceback suppression while preserving records; idempotent handlers; file output; job-specific log paths | Temporary log directories and isolated logger handlers |
| REST server entry point and exports | Flask server startup; contract loader delegation; lazy exports and unknown-attribute errors | Mocked Flask server and package modules |

## Regression cases

The suite protects four implementation corrections found during the coverage
review:

1. Importing Docker runtime code does not contact the Docker daemon. The client
   is created only when a runtime operation is called.
2. Execution listings use the execution directory as the execution ID instead
   of returning the literal `status` directory name.
3. Completed log reads expand `*.log` in the same way as streaming reads.
4. Failed inference runs can package input and log metadata without attempting
   to write a missing result file.
5. Contract datatypes such as `Int64` and `Float64` are matched
   case-insensitively during DataFrame validation.

`ExecutionScheduler.Scheduler` is intentionally listed as a placeholder;
implementing scheduling is outside this unit-test expansion.
