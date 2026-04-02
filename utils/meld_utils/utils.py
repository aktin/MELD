from pathlib import Path

import yaml


def resolve_path(path_from_contract: str, base_dir: str | Path | None = None) -> str:
    """
    Resolve a file system path relative to a base directory.

    This function combines a given relative path with a base directory
    or the parent directory of the script. It then resolves the result
    to a full, absolute path.

    :param path_from_contract: A relative path to be resolved.
    :type path_from_contract: str
    :param base_dir: The base directory to resolve the relative path against.
                     If not provided, the parent directory of the current script 
                     will be used.
    :type base_dir: str | Path | None
    :return: The absolute resolved path as a string.
    :rtype: str
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    return str((root / path_from_contract).resolve())


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
