# Prerequisites

1.  Install [Docker](https://docs.docker.com/engine/install/)
2.  Download [compose.yaml](https://raw.githubusercontent.com/aktin/MELD/refs/tags/v0.2.0-alpha/scripts/compose.yaml)
3. Make sure that the contract and the sql file with the query to the Aktin DWH are on the same host, you can find examples [here](https://github.com/aktin/MELD/tree/main/examples/nn/resources)
4.  Create `.env` in the same folder as `compose.yaml` and store your setup information, e.g.
	   * Docker compose automatically reads the contents of .env
	   
	```.env
	# Aktin DWH connection info
	#DB_HOST=my-host # do not set if db is hosted on localhost
	DB_PORT=5432
	DB_USER=i2b2crcdata
	DB_SCHEMA=i2b2
	# orchestrator information
	# optional, only needed if file path differs from ./contract.yaml
	MELD_CONTRACT_FILE=$HOME/MELD/examples/nn/resources/contract.yaml
	# optional, only needed if file path differs from ./query.yaml
	MELD_QUERY_FILE=$HOME/MELD/examples/nn/resources/query.sql
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

## Delete runtime image
1.Open a terminal and navigate to the folder containing the `compose.yaml`
2 Run `docker compose run meld delete`

## Get results
After a completed (successful or unsuccessful), you can find the output archive in `./jobs/<job_id>/outout/summarized_execution.zip`.
It contains 
* `input/`: all input files like `input.csv` and `contract.yaml`
* `logs/`: logs of the job
* `output/`: result files from the runtime container, like `output.csv`