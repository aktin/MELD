"""
Utilities for interacting with file paths, YAML files, and downloading files
from URLs.

This module provides helper functions to resolve file paths relative to a
base directory, load and parse YAML files into dictionaries, sanitize and
validate URLs, generate safe filenames, and download files from the web.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

import yaml

from utils.config import ROOT_DIR


def load_yaml(source: str | TextIO) -> dict:
    """
    Loads a YAML file and parses its contents into a dictionary.

    :param source: The file path or an open text stream containing YAML.
    :return: A dictionary representation of the YAML file's contents.
    :rtype: dict
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the specified file is not a valid YAML file.
    """

    if hasattr(source, "read"):
        return yaml.safe_load(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"{path} is not a YAML file.")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def to_yaml(source: dict) -> str:
    return yaml.dump(source, default_flow_style=False)

def read_contract(contract_id: str) -> Contract:
    from ModelManager.contract_models import Contract

    contract_path = os.path.join(ROOT_DIR, "contracts", contract_id, "contract.yaml")
    with open(contract_path, "r", encoding="utf-8") as contract_file:
        return Contract.from_yaml(contract_file)
