# Model Card: MELD Wait Runtime

**Contract:** `wait-runtime` v0.1.0
**Framework:** test runtime
**Runtime Image:** `ghcr.io/simhue/meld/runtime_examples/wait:0.1.0`

## Purpose

Test-only runtime for checking execution duration, timeout handling, and empty result processing.

The runtime accepts `/input/input.csv`, ignores its contents, waits for the configured duration, and writes a result with zero data rows to `/output/output.csv`.

## Configuration

Set `WAIT_SECONDS` through the contract’s `runtime.environment_variables` section. The default is 10 seconds when the variable is not set.

## Build

```bash
cd examples/wait/build
./build.sh
```

The contract contains a placeholder digest until the image is published. For a local image test, set `MELD_PULL_WITH_DIGEST=False` or replace the digest with the published image digest.
