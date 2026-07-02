from typing import Iterator

# TODO: ergibt sqlalchemy sinn?
from sqlalchemy import text

import pandas as pd
from ModelEnvironment import JobContext
from pandas import DataFrame

from Logger import get_meld_logger
from .db import engine

logger = get_meld_logger()


def execute_query(job_context: JobContext, params: dict | None = None) -> DataFrame | Iterator[DataFrame]:
    """
    Executes a SQL query and retrieves the result as a DataFrame or an iterator of DataFrames.

    Parameters:
        sql (str): The SQL query to execute.
        params (dict | None): The parameters to use with the SQL query. Defaults to None.

    Returns:
        DataFrame | Iterator[DataFrame]: The result set of the query as a DataFrame
        or an iterator of DataFrames (if applicable).
    """
    statement = job_context.contract["input_schema"]["query"]["statement"]
    features = job_context.contract["input_schema"]["features"]
    with (engine.connect() as connection):
        df = pd.read_sql_query(text(statement),
                               connection,
                               params=params,
                               dtype={ col["name"]: col["datatype"] for col in features})

        return df
