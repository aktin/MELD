import os

from meld_logger import setup_logger
from ModelManager.manager import run_inference

logger = setup_logger("meld")

if __name__ == "__main__":
    # Minimal CLI: customize paths via env vars to avoid adding argparse right now.
    contract = os.getenv("CONTRACT_PATH", "/home/shuening/code/KlimaNot/artifact/contract.yaml")
    output_csv = os.getenv("OUTPUT_CSV", "~/code/KlimaNot/output/predictions.csv")
    path = run_inference(contract_path=contract, output_csv_path=output_csv)
    logger.info(f"Wrote predictions to: {path}")
