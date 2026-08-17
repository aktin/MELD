import logging
import os

from sqlalchemy import create_engine

logger = logging.getLogger("meld")

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
schema = os.getenv("DB_SCHEMA")
password_file = os.getenv("DB_PASSWORD_FILE")

try:
    with open(password_file, "r") as f:
        password = f.readline().strip()
except FileNotFoundError:
    logger.error(f"Database password file could not be found")
    exit(1)
except TypeError:
    logger.error(f"DB_PASSWORD_FILE is not set")
    exit(1)

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{schema}")
