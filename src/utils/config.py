import os
from pathlib import Path

CONTRACTS_DIR = Path(os.environ.get("MELD_CONTRACT_DIRECTORY", "/contracts"))
API_HOST = os.environ.get("MELD_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("MELD_API_PORT", "5000"))
LOG_DIR = Path(os.environ.get("MELD_LOG_DIR", "/logs"))
ROOT_DIR = Path(os.environ.get("MELD_ROOT_DIR", "/"))
PULL_WITH_DIGEST = not os.environ.get("MELD_PULL_WITH_DIGEST", "True") == "False"
SCHEDULE_DIR = Path(os.environ.get("MELD_SCHEDULE_DIR", "/schedules"))
