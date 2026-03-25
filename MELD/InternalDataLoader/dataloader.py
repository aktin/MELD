from datetime import date

import pandas as pd
# TODO: ergibt sqlalchemy sinn?
from sqlalchemy import text

from .db import engine

def execute_query(sql: str, params: dict = None) -> pd.DataFrame:

    with (engine.connect() as connection):
        df = pd.read_sql_query(text(sql), connection, params=params)

        return df

# def get_results_multi(sql: str, start: date, end: date, features) -> dict[str, pd.DataFrame]:
#     with engine.connect() as connection:
#         connection.execute(text(sql), {"start": start, "end": end})
#
#         # For PostgreSQL - query pg_tables for temporary tables
#         result = connection.execute(text("""
#                                          SELECT tablename
#                                          FROM pg_tables
#                                          WHERE schemaname LIKE 'pg_temp%'
#                                          """))
#         temp_tables = [row[0] for row in result]
#
#         dataframes = {}
#         for temp_table in temp_tables:
#             df = pd.read_sql(text(f"select * from {temp_table}"), connection)
#             dataframes[temp_table] = df
#         return dataframes