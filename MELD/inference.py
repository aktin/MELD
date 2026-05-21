import os

from ModelManager.manager import run_inference
from meld_logger import setup_logger

logger = setup_logger("meld")

if __name__ == "__main__":
    run_inference(contract_path="/resources/contract.yaml")
