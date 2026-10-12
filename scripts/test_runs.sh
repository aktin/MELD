export DB_HOST=localhost
export DB_PORT=5433
export DB_USER=i2b2crcdata
export DB_SCHEMA=i2b2
export DB_PASSWORD_FILE=../MELD/.devcontainer/db_password.txt
export MELD_LOG_DIR=../MELD/logs
export MELD_ROOT_DIR=../MELD

cd ../MELD

./.venv/bin/python main.py &
server_pid=$!
trap 'kill "$server_pid"' EXIT

curl --fail --data-binary @../examples/nn/resources/contract.yaml -H 'Content-Type: application/yaml' http://localhost:5000/contracts
curl --fail --data-binary @../examples/sklearn/resources/contract.yaml -H 'Content-Type: application/yaml' http://localhost:5000/contracts
curl --fail --data-binary @../examples/tfdf/resources/contract.yaml -H 'Content-Type: application/yaml' http://localhost:5000/contracts
