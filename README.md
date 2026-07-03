# MELD - Machine Learning Execution and Deployment

MELD is a framework to execute machine learning models on the Aktin DWH.

## Quickstart

1. Install [Docker](https://docs.docker.com/engine/install/)
2. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/tags/v0.2.0-alpha/scripts/compose.yaml)
3. Create .env and secrets.txt (see Prerequisites below).
3. Run `docker compose pull`
4. Run `docker compose run meld pull`
4. Run `docker compose run meld run`
5. Retrieve the execution archive from `./jobs/<job_id>/output/execution_summary.zip`

## Prerequisites

1. Install [Docker](https://docs.docker.com/engine/install/)
1. Aktin DWH needs to be installed and running
2. Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/tags/v0.2.0-alpha/scripts/compose.yaml)
3. Make sure that the contract is stored on the same host, you can find
   examples [here](https://github.com/aktin/MELD/tree/main/examples/nn/resources)
4. Create `.env` in the same folder as `compose.yaml` andconfigure the database connection and MELD settings, e.g.
    * Docker compose automatically reads the contents of .env

   ```.env
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

4. In the same folder also create a file `secrets.txt` and fill it with the db password
5. Pull MELD image `docker compose pull`

## Load inference runtime image

Before the first inference run, the inference runtime image must be pulled.

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Run `docker compose run meld pull`

## Run inference job

1. Open a terminal and navigate to the folder containing the `compose.yaml`
2. Run `docker compose run meld run`
   * The orchestrator creates for each job a new folder in `./jobs/`
   * The job id consists the timestamp of the job start.

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