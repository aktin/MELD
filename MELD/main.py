import os

from ModelManager import run_inference
from ModelManager.manager import pull_runtime, remove_runtime

cmd = os.environ.get("MELD_CMD")

if not cmd:
    raise Exception("MELD_CMD is not set")

# if cmd == "train":
#     pass
#     # Minimal CLI: customize paths via env vars to avoid adding argparse right now.
#     contract_path = os.getenv("CONTRACT_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/resources/contract-training.yaml")
#     query_path = os.getenv("QUERY_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/resources/long-query-training.sql")
#     artifact_path = os.getenv("ARTIFACT_PATH", "/home/shuening/code/KlimaNot/examples/tfdf/artifact_test")
#     run_training(contract_path, query_path, artifact_path)
# elif cmd == "run":
if cmd == "run":
    run_inference(contract_path="/resources/contract.yaml")
elif cmd == "pull":
    pull_runtime(contract_path="/resources/contract.yaml")
elif cmd == "delete":
    remove_runtime(contract_path="/resources/contract.yaml")
