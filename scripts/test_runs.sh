export DB_HOST=localhost
export DB_PORT=5433
export DB_USER=i2b2crcdata
export DB_SCHEMA=i2b2
export DB_PASSWORD_FILE=../MELD/.devcontainer/db_password.txt
export MELD_LOG_DIR=../MELD/logs
export MELD_ROOT_DIR=../MELD

cd ../MELD

MELD_CONTRACT_DIRECTORY=../examples/nn/resources \
    ./.venv/bin/python main.py run contract.yaml

MELD_CONTRACT_DIRECTORY=../examples/sklearn/resources \
    ./.venv/bin/python main.py run contract.yaml

MELD_CONTRACT_DIRECTORY=../examples/tfdf/resources \
    ./.venv/bin/python main.py run contract.yaml
