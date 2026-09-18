from __future__ import annotations

from .config_loader import load_contract
from .contract_models import (
    Contract,
    ContractDataType,
    ContractMetadata,
    Feature,
    InputSchema,
    OutputSchema,
    Query,
    RuntimeConfig,
    RuntimeImage,
    TemporalScope,
)
from .contract_service import ContractService

__all__ = [
    "Contract",
    "ContractDataType",
    "ContractMetadata",
    "ContractService",
    "Feature",
    "InputSchema",
    "OutputSchema",
    "Query",
    "RuntimeConfig",
    "RuntimeImage",
    "TemporalScope",
    "load_contract",
    "pull_runtime",
    "remove_runtime",
    "run_inference",
]


def run_inference(contract: Contract, ctx=None) -> None:
    from .manager import run_inference as _run_inference

    return _run_inference(contract, ctx)


def pull_runtime(contract: Contract) -> None:
    from .manager import pull_runtime as _pull_runtime

    return _pull_runtime(contract)


def remove_runtime(contract: Contract) -> None:
    from .manager import remove_runtime as _remove_runtime

    return _remove_runtime(contract)
