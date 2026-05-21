import os

from ModelManager import run_training

if __name__ == "__main__":
    # Minimal CLI: customize paths via env vars to avoid adding argparse right now.
    contract_path = os.getenv("CONTRACT_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/resources/contract-training.yaml")
    query_path = os.getenv("QUERY_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/resources/long-query-training.sql")
    artifact_path = os.getenv("ARTIFACT_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/artifact_test")
    run_training(contract_path, query_path, artifact_path)

