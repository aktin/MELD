import os
from datetime import timedelta, datetime

import isodate

from ModelManager import manager, config_loader, run_training

if __name__ == "__main__":
    # Minimal CLI: customize paths via env vars to avoid adding argparse right now.
    contract_path = os.getenv("CONTRACT_PATH", "/home/shuening/code/KlimaNot/artifact/contract.yaml")
    run_training(contract_path)

