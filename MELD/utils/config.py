import os

CONTRACT_DIRECTORY = os.environ.get("MELD_CONTRACT_DIRECTORY", "/contracts")
API_HOST = os.environ.get("MELD_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("MELD_API_PORT", "5000"))
LOG_DIR = os.environ.get("MELD_LOG_DIR", "/logs")
ROOT_DIR = os.environ.get("MELD_ROOT_DIR", "/")
PULL_WITH_DIGEST = not os.environ.get("MELD_PULL_WITH_DIGEST", "True") == "False"
