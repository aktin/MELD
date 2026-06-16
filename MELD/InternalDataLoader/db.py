import os

from sqlalchemy import create_engine

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
schema = os.getenv("DB_SCHEMA")
password_file = os.getenv("DB_PASSWORD_FILE")

with open(password_file, "r") as f:
    password = f.read()

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{schema}")
engine.connect()
