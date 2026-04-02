from meld_logger import setup_logger
from meld_utils import load_yaml

logger = setup_logger("meld")


def load_contract(path: str):
    """
    Loads a contract from the specified YAML file path.

    :param path: The file path to the YAML file containing the contract definition.
    :type path: str
    :return: The contract data loaded from the specified YAML file.
    :rtype: Any
    """
    logger.info(f"Loading contract from {path}")
    return load_yaml(path)
