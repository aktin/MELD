import os

from ModelManager.manager import run_inference

if __name__ == "__main__":
    # Minimal CLI: customize paths via env vars to avoid adding argparse right now.
    contract = os.getenv("CONTRACT_PATH", "/home/shuening/code/KlimaNot/artifact/contract.yaml")
    output_csv = os.getenv("OUTPUT_CSV", "predictions.csv")
    path = run_inference(contract_path=contract, output_csv_path=output_csv)
    print(f"Wrote predictions to: {path}")
