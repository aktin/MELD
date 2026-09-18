from typing import TextIO

from Logger import get_meld_logger

from .contract_models import Contract

logger = get_meld_logger()


def load_contract(source: str | TextIO) -> Contract:
    """
    Loads a contract from the specified YAML file path.

    :param source: The YAML file path or an open text stream.
    :return: The contract data loaded from the specified YAML file.
    :rtype: Any
    """
    logger.debug("Loading contract from %s", source)
    return Contract.from_yaml(source)
