import os
import sys
import tempfile
from pathlib import Path


MELD_ROOT = Path(__file__).resolve().parents[1]
if str(MELD_ROOT) not in sys.path:
    sys.path.insert(0, str(MELD_ROOT))

_TEST_ENVIRONMENT = tempfile.TemporaryDirectory(prefix="meld-tests-")
TEST_ROOT = Path(_TEST_ENVIRONMENT.name)
TEST_LOG_DIR = TEST_ROOT / "logs"
TEST_PASSWORD_FILE = TEST_ROOT / "db_password.txt"
TEST_LOG_DIR.mkdir()
TEST_PASSWORD_FILE.write_text("test-password\n", encoding="utf-8")

# Unit tests own these resources so inherited deployment settings cannot make
# imports write to `/logs` or require an external database secret.
os.environ["MELD_LOG_DIR"] = str(TEST_LOG_DIR)
os.environ["DB_PASSWORD_FILE"] = str(TEST_PASSWORD_FILE)
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5433"
os.environ["DB_USER"] = "test"
os.environ["DB_SCHEMA"] = "test"


def contract_data(**overrides):
    data = {
        "contract": {
            "name": "example-contract",
            "description": "Example",
            "version": "1.0.0",
        },
        "runtime": {
            "framework": "sklearn",
            "image": {
                "name": "example/runtime",
                "tag": "1.0.0",
                "digest": "sha256:example",
            },
        },
        "input_schema": {
            "temporal_scope": {
                "type": "relative",
                "value": "P1D",
                "anchor": "2020-01-01T00:00:00Z",
            },
            "features": [{"name": "age", "datatype": "Int64"}],
            "query": {"type": "sql", "statement": "SELECT age"},
        },
        "output_schema": {
            "type": "csv",
            "predictor": [{"name": "prediction", "datatype": "Float64"}],
        },
    }
    data.update(overrides)
    return data
