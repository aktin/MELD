from typing import Iterator

# TODO: ergibt sqlalchemy sinn?
from sqlalchemy import text

import pandas as pd
from pandas import DataFrame

from Logger import get_meld_logger
from .db import engine

logger = get_meld_logger()


def execute_query(sql: str, params: dict | None = None) -> DataFrame | Iterator[DataFrame]:
    """
    Executes a SQL query and retrieves the result as a DataFrame or an iterator of DataFrames.

    Parameters:
        sql (str): The SQL query to execute.
        params (dict | None): The parameters to use with the SQL query. Defaults to None.

    Returns:
        DataFrame | Iterator[DataFrame]: The result set of the query as a DataFrame
        or an iterator of DataFrames (if applicable).
    """
    with (engine.connect() as connection):
        df = pd.read_sql_query(text(sql), connection, params=params)

        return df
