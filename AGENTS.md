# Project Instructions

## Project overview

MELD (Machine Learning Execution and Deployment) is a Python orchestrator for running machine-learning inference jobs. It loads model contracts, queries the Aktin data warehouse, prepares input data, runs an inference runtime in Docker, records execution state and logs, and writes a summarized ZIP archive.

## Repository layout

- `MELD/` contains the orchestrator application.
- `MELD/ModelEnvironment/` contains execution contexts, background execution services, inference runtime integration, and Docker operations.
- `MELD/ModelManager/` contains contract models, contract persistence, configuration loading, and inference orchestration.
- `MELD/Server/` contains the Flask API and HTTP endpoints.
- `MELD/tests/` contains the unit test suite.
- `examples/` contains example inference runtimes.
- `docs/` contains project and testing documentation.
- `scripts/` contains development and integration helpers.

## Development commands

Run the unit test suite from the repository root:

```bash
./.venv/bin/python -m unittest discover -s MELD/tests -p 'test*.py' -v
```

The unit tests mock Docker, PostgreSQL, registries, and live servers. They should not require external services.

For container-based development, use the configuration under `MELD/.devcontainer/`. The application dependencies are listed in `MELD/requirements.txt`.

## Implementation guidance

- Match the existing Python style and keep changes targeted to the requested behavior.
- Preserve the public API, contract file layout, and existing filesystem paths unless the task explicitly changes them.
- Treat contract and execution directories as persistent application data. Avoid changing their names or structure without updating all readers and writers.
- Execution state is persisted in `status/status.json`; logs are stored under the execution's `logs/` directory; completed results are stored as `output/summarized_execution.zip`.
- `ExecutionService` submits inference work to its single-worker executor, tracks the active execution, supports cancellation, lists persisted executions, reads logs, streams log lines, and returns result archives. Changes to this behavior should update the corresponding tests in `MELD/tests/test_execution_service.py`.
- Keep Docker and database interactions behind the existing runtime and data-loader boundaries so unit tests remain isolated.
- Do not add dependencies, tools, services, or broad refactors unless the task requires them.

## Validation

- Run the smallest relevant existing tests for the change.
- Run the full unit-test command when modifying shared execution, contract, API, runtime, or persistence behavior.
- Review `git diff` and run `git diff --check` before handing off changes.
- Do not modify unrelated existing worktree changes.

## Assumptions

- This is a proof of concept
- Currently only one execution should run at a time