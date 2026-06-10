from typing import Iterator

import pandas as pd
# TODO: ergibt sqlalchemy sinn?
from sqlalchemy import text

import pandas as pd
from Logger import setup_logger
from pandas import DataFrame
from .db import engine

logger = setup_logger("meld")

def execute_query(sql: str, params: dict = None) -> pd.DataFrame:
    with (engine.connect() as connection):
        df = pd.read_sql_query(text(sql), connection, params=params)

        return df
