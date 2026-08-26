"""
Utilities for interacting with file paths, YAML files, and downloading files
from URLs.

This module provides helper functions to resolve file paths relative to a
base directory, load and parse YAML files into dictionaries, sanitize and
validate URLs, generate safe filenames, and download files from the web.
"""
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    """
    Loads a YAML file and parses its contents into a dictionary.

    :param path: The file path to a YAML file to be loaded.
    :type path: str
    :return: A dictionary representation of the YAML file's contents.
    :rtype: dict
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the specified file is not a valid YAML file.
    """

    if not Path(path).exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    if not path.endswith(".yaml") and not path.endswith(".yml"):
        raise ValueError(f"The file {path} is not a YAML file.")

    with open(path, "r") as file:
        contract = yaml.safe_load(file)

    return contract


def construct_image_ref(contract: dict) -> str:
    """
    Constructs a formatted image reference string based on the provided contract
    dictionary.

    Parameters:
    contract (dict): Dictionary containing the 'runtime.image' section with
    the keys 'name' and 'name'.

    Returns:
    str: A formatted image ref string in the format "<name>:<tag>@<digest>".
    """
    return f"{contract['runtime']['image']['name']}:{contract['runtime']['image']['tag']}" # @{contract['runtime']['image']['digest']}"
