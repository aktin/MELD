# MELD - Machine Learning Execution and Deployment

MELD is a framework to execute machine learning models on the Aktin DWH.

It separates data preparation from model execution by running each inference runtime inside an isolated Docker container. The orchestrator is responsible for querying the AKTIN DWH, preparing the input data, executing the runtime, collecting logs, and archiving the results.

MELD consists of two components:

- **Orchestrator** – executes SQL queries, prepares the runtime environment, manages the execution lifecycle, and archives the results.
- **Inference Runtime** – a Docker image containing a trained model and the inference logic.

The interaction between both components is defined by a **model contract**, which specifies the runtime image, input schema, output schema, and execution parameters.


## Quickstart

1. Install [Docker](https://docs.docker.com/engine/install/)
2. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/compose.yaml)
3. Create `.env` and `db_password.txt` as described in the prerequisites below.
4. Start MELD. The orchestrator runs as a non-root user and needs the Docker socket's group ID:

   ```bash
   export DOCKER_SOCKET_GID="$(stat -c '%g' /var/run/docker.sock)"
   docker compose up -d
   ```

5. Copy the contract you want to use into `./contracts` and name it `contract.yaml`.
6. Pull the runtime image for the contract:

   ```bash
   docker compose exec meld orchestrator pull contract.yaml
   ```

   If the image is hosted on a private registry, log in first as described in [Load inference runtime image](#load-inference-runtime-image).
7. Run an inference job:

   ```bash
   docker compose exec meld orchestrator run contract.yaml
   ```

8. Retrieve the execution archive from `./jobs/<job_id>/output/summarized_execution.zip`.

## Prerequisites

1. Install [Docker](https://docs.docker.com/engine/install/)
2. Make sure the Aktin DWH is installed and running.
3. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/compose.yaml)
   ```bash
   curl -o compose.yaml https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/compose.yaml
   ```
4. Create `.env` in the same folder as `compose.yaml` and configure the database connection and MELD settings. Docker Compose automatically reads this file.

   ```dotenv
   # Aktin DWH connection info
   #DB_HOST=my-host # do not set if db is hosted on localhost
   DB_PORT=5432
   DB_USER=i2b2crcdata
   DB_SCHEMA=i2b2
   # Optional timezone used by the container.
   TZ=Europe/Berlin
   ```

5. In the same folder, create `db_password.txt` and fill it with the database password. You can either create the file directly:

   **db_password.txt:**

   ```text
   my-db-password
   ```

   Or open a terminal, navigate to the folder containing `compose.yaml`, and run:

   ```bash
   echo "my-db-password" > db_password.txt
   ```

6. Start MELD. The orchestrator runs as a non-root user and needs the Docker socket's group ID. Set it in the shell before starting Docker Compose:

   ```bash
   export DOCKER_SOCKET_GID="$(stat -c '%g' /var/run/docker.sock)"
   docker compose up -d
   ```

   The following folders are created after startup:

   | Folder      | Purpose                                                                            |
   |-------------|------------------------------------------------------------------------------------|
   | `./jobs`      | Contains all related files for every job, e.g. logs, input data, and inference results |
   | `./contracts` | Stores the contract                                                   |
   | `./logs`      | Contains MELD logs   |
7. Copy the contract you want to use into `./contracts` and name it `contract.yaml`.

## Load inference runtime image

Before the first inference execution, pull the inference runtime image.

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. If the image is hosted on a private registry, log in first with [`docker login`](https://docs.docker.com/reference/cli/docker/login/) using either classic username/password authentication:

   ```bash
   docker login <registry>
   ```

   Or, if you have a [security token](https://docs.docker.com/security/access-tokens/):

   ```bash
   export DOCKER_TOKEN=<token>
   echo $DOCKER_TOKEN | docker login <registry> --username <user> --password-stdin
   ```

3. Start the worker and pull the runtime for the selected contract:

   ```bash
   docker compose up -d
   docker compose exec meld orchestrator pull contract.yaml
   ```

## Run inference job

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Run an inference job once:

   ```bash
   docker compose exec meld orchestrator run contract.yaml
   ```

   The orchestrator creates a new folder for each job in `./jobs/`. The job ID consists of the ID in the contract and the job start timestamp. For example:

   ```text
   ./jobs/stationary-admission-test-tfdf_20260708153747/
   ```

## Scheduling

Scheduling is currently a placeholder. The `ExecutionScheduler` package is
kept as a dependency-free shell for a future implementation.

## Delete runtime image

When the runtime image is no longer needed, you can delete it.

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Remove the runtime image:

   ```bash
   docker compose exec meld orchestrator remove contract.yaml
   ```

## Get results

After a job finishes, find the output archive at
`./jobs/<job_id>/output/summarized_execution.zip`. The job ID is printed to the console.

The archive contains:

* `input/`: input files such as `input.csv` and `contract.yaml`
* `logs/`: job logs
* `output/`: the `output.csv` result file from the runtime container
