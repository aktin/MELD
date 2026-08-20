from __future__ import annotations

from .config_loader import load_contract

__all__ = ["load_contract", "pull_runtime", "remove_runtime", "run_inference"]


def run_inference(contract_path: str) -> None:
    from .manager import run_inference as _run_inference

    return _run_inference(contract_path)


def pull_runtime(contract_path: str) -> None:
    from .manager import pull_runtime as _pull_runtime

    return _pull_runtime(contract_path)


def remove_runtime(contract_path: str) -> None:
    from .manager import remove_runtime as _remove_runtime

    return _remove_runtime(contract_path)
