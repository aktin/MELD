import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
schema = os.getenv("DB_SCHEMA")


engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{schema}")
engine.connect()