import yaml

from Logger import logger
from utils import load_yaml


def load_contract(path: str):
    return load_yaml(path)
