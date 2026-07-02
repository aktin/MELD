import json
import os.path

from jsonschema import validate

from Logger.logger import get_meld_logger
from utils import load_yaml

logger = get_meld_logger()


def load_contract(path: str) -> dict:
    """
    Loads a contract from the specified YAML file path.

    :param path: The file path to the YAML file containing the contract definition.
    :type path: str
    :return: The contract data loaded from the specified YAML file.
    :rtype: Any
    """
    logger.debug(f"Loading contract from {path}")
    contract = load_yaml(path)

    _validate_contract(contract)

    return contract

def _validate_contract(contract: dict):
    with open(os.path.join(os.path.dirname(__file__), "contract.schema.json"), "r") as f:
        schema = json.load(f)

    validate(contract, schema)