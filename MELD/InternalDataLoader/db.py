import logging
import os
from pathlib import Path

from sqlalchemy import create_engine

logger = logging.getLogger("meld")

host = os.getenv("DB_HOST", default="localhost")
port = os.getenv("DB_PORT", default="5432")
user = os.getenv("DB_USER", default="i2b2crcdata")
schema = os.getenv("DB_SCHEMA", default="i2b2")
password_file = os.getenv("DB_PASSWORD_FILE")

try:
    with Path(password_file).open("r") as f:
        password = f.readline().strip()
except FileNotFoundError:
    logger.error(f"Database password file could not be found")
    exit(1)
except TypeError:
    logger.error(f"DB_PASSWORD_FILE is not set")
    exit(1)

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{schema}")
