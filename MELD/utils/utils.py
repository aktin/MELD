from pathlib import Path

import yaml


def resolve_path(path_from_contract: str, base_dir: str | Path | None = None) -> str:
    """
    Resolve a path relative to a configurable base directory.

    If base_dir is not provided, uses the current file's parent directory.
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    return str((root / path_from_contract).resolve())


def load_yaml(path: str) -> dict:
    with open(path, "r") as file:
        contract = yaml.safe_load(file)

    return contract
