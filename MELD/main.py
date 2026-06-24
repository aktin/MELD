import os
import sys

from Logger import get_meld_logger
from ModelManager import run_inference
from ModelManager.manager import pull_runtime, remove_runtime


if __name__ == "__main__":
    logger = get_meld_logger()


    argv = sys.argv[1:]

    if not argv:
        logger.error("No command set")

    cmd = argv[0]

    if cmd == "run":
        run_inference(contract_path="/resources/contract.yaml")
    elif cmd == "pull":
        pull_runtime(contract_path="/resources/contract.yaml")
    elif cmd == "delete":
        remove_runtime(contract_path="/resources/contract.yaml")
    else:
        logger.error(f"Unknown command: {cmd}")
