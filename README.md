# MELD - Machine Learning Execution and Deployment

MELD is a framework to execute machine learning models on the Aktin DWH.

It separates data preparation from model execution by running each inference runtime inside an isolated Docker container. The orchestrator is responsible for querying the AKTIN DWH, preparing the input data, executing the runtime, collecting logs, and archiving the results.

MELD consists of two components:

- **Orchestrator** – executes SQL queries, prepares the runtime environment, manages the execution lifecycle, and archives the results.
- **Inference Runtime** – a Docker image containing a trained model and the inference logic.

The interaction between both components is defined by a **model contract**, which specifies the runtime image, input schema, output schema, and execution parameters.


## Quickstart

1. Install [Docker](https://docs.docker.com/engine/install/)
2. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/tags/v0.2.0-alpha/scripts/compose.yaml)
3. Create .env and db_password.txt (see Prerequisites below).
3. Run `docker compose up`
5. Retrieve the execution archive from `./jobs/<job_id>/output/execution_summary.zip`

## Prerequisites

1. Install [Docker](https://docs.docker.com/engine/install/)
1. Aktin DWH needs to be installed and running
2. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/tags/v0.2.0-alpha/scripts/compose.yaml)
3. Make sure that the contract is stored on the same host, you can find
   examples [here](https://github.com/aktin/MELD/tree/main/examples/nn/resources)
4. Create `.env` in the same folder as `compose.yaml` andconfigure the database connection and MELD settings, e.g.
    * Docker compose automatically reads the contents of .env

   ```dotenv
   # Aktin DWH connection info
   #DB_HOST=my-host # do not set if db is hosted on localhost
   DB_PORT=5432
   DB_USER=i2b2crcdata
   DB_SCHEMA=i2b2
   # orchestrator information
   # Optional.
   # Defaults to ./contract.yaml relative to compose.yaml.
   MELD_CONTRACT_FILE=$HOME/MELD/examples/nn/resources/contract.yaml
   ```

4. In the same folder also
    * create a file `db_password.txt` and fill it with the db password, or

      **db_password.txt:**

       ```text
       my-db-password
       ```
    * open a terminal and navigate to the folder containing the `compose.yaml` and run
      ```bash
      echo "my-db-password" > db_password.txt
      ```
5. Pull MELD image `docker compose pull`

## Load inference runtime image

Before the first inference execution, the inference runtime image must be pulled.

1. Open a terminal and navigate to the folder containing the `compose.yaml`
   * @TODO: make this step optional
    * If the image is hosted on a private registry, log in to the registry first with [
      `docker login`](https://docs.docker.com/reference/cli/docker/login/) for classic username/password authentication
   ```bash
   docker login <registry>
   ```
   * or if you have a security token
   ```bash
   export DOCKER_TOKEN=<token>
   echo $DOCKER_TOKEN | docker login <registry> --username <user> --password-stdin
   ```
2. Run `docker compose run meld pull`

## Run inference job

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Run `docker compose run meld run`
    * The orchestrator creates for each job a new folder in `./jobs/`
    * The job id consists of the id in the contract and the timestamp of the job start
   ```text
   stationary-admission-test-tfdf_20260708153747
   ```

## Delete runtime image

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Run `docker compose run meld delete`

## Get results

After a finished job, you can find the output archive in
`./jobs/<job_id>/output/summarized_execution.zip`. The job id is printed to the console.

It contains

* `input/`: all input files like `input.csv` and `contract.yaml`
* `logs/`: logs of the job
* `output/`: result file `output.csv` from the runtime container 