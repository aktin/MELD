import os
import sys

from Logger import get_meld_logger
from ModelManager import run_inference
from ModelManager.manager import pull_runtime, remove_runtime


def main(argv: list[str] | None = None) -> int:
    logger = get_meld_logger()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        logger.error("No command set")
        return 1

    cmd = argv[0]

    contract_path = os.environ.get("MELD_CONTRACT_FILE", "/resources/contract.yaml")

    if cmd == "run":
        run_inference(contract_path=contract_path)
    elif cmd == "pull":
        pull_runtime(contract_path=contract_path)
    elif cmd == "delete":
        remove_runtime(contract_path=contract_path)
    else:
        logger.error(f"Unknown command: {cmd}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
