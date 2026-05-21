import os
from datetime import datetime, timedelta

import isodate
import pandas as pd
from ModelEnvironment import JobContext

from ModelEnvironment.job_context import create_job_context, JobStatus
from meld_logger import setup_logger

import ModelEnvironment
from InternalDataLoader import execute_query
from ModelManager import load_contract
from meld_utils import resolve_path

logger = setup_logger("meld")


def load_query(job_context: JobContext) -> str:
    """
    Loads a SQL query string from the specified file path.

    :param path: The file path to the query file to be loaded.
    :type path: str
    :return: The SQL query string loaded from the file.
    :rtype: str
    """
    job_context.logger.info(f"Loading query from {job_context.query_path}")

    if not os.path.exists(job_context.query_path):
        raise FileNotFoundError(f"The file {job_context.query_path} does not exist.")
    if not job_context.query_path.endswith(".sql"):
        raise ValueError(f"The file {job_context.query_path} is not a SQL file. ")

    with open(job_context.query_path, "r") as file:
        query = file.read()
        return query


def query_data(query: str, params: dict, job_context: JobContext) -> pd.DataFrame:
    """
    Executes a given SQL query with parameters and returns the result as a pandas DataFrame.

    :param query: SQL query string to be executed.
    :param params: Dictionary of parameters to be used in the SQL query.
    :return: A pandas DataFrame containing the results of the executed query.
    :rtype: pd.DataFrame
    """
    job_context.logger.info(f"Executing query")
    start = datetime.now()
    data = execute_query(query, params)
    end = datetime.now()
    job_context.logger.info(f"Query returned {len(data)} rows and took {end - start} seconds.")
    return data


def run_inference(contract_path: str = "contract.yaml") -> None:
    job_context = create_job_context(contract_path)
    try:
        job_context.log_event("Preparing inference", JobStatus.PREPARING)
        start, end = _compute_time_window(job_context)
        params = {"start": start.isoformat(), "end": end.isoformat()}

        query = load_query(job_context)

        df = query_data(query, params, job_context)

        feature_cols = _validate_features(df, job_context)
        x = _normalize_features(df, feature_cols)

        ModelEnvironment.run_inference(x, job_context)
    except Exception as e:
        logger.exception(f"An exception occurred during inference: {e}")


def run_training(contract_path: str, query_path: str, artifact_path: str) -> str:
    """
    !!! Reine Testfunktion !!!

    Runs the training pipeline, loading configuration and processing data
    before invoking the model training.

    :param contract_path: Path to the directory containing the contract
        and related configuration files. Defaults to "artifact_df".
    :type contract_path: str
    :return: Path to the directory containing the artifacts of the training
        process.
    :rtype: str
    """
    config = load_contract(contract_path)

    start, end = _compute_time_window(job_context)
    params = {"start": start.isoformat(), "end": end.isoformat()}

    query = load_query(query_path)

    df = query_data(query, params)

    feature_cols = _validate_features(df, job_context)
    x = _normalize_features(df, feature_cols)

    ModelEnvironment.run_training(x, artifact_path=artifact_path)


def _compute_time_window(job_context: JobContext) -> tuple[datetime, datetime]:
    scope = job_context.contract["input_schema"]["temporal_scope"]
    anchor = scope.get("anchor")
    duration = scope.get("value")

    end = datetime.fromisoformat(anchor) if anchor else datetime.now()
    td: timedelta = isodate.parse_duration(duration, as_timedelta_if_possible=False).totimedelta(end=end)
    start = end + td
    job_context.logger.info(f"Temporal window start: {start.isoformat()}, end: {end.isoformat()}")
    return start, end


def _validate_features(df: pd.DataFrame, job_context: JobContext) -> list[dict]:
    job_context.logger.info(f"Validating features")
    features = job_context.contract["input_schema"]["features"]
    required = [f["name"] for f in features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Query result is missing required feature columns: {', '.join(missing)}. "
            f"Ensure your SQL produces columns matching contract.yaml input_schema.features."
        )
    return required


def _normalize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()

    for col in x.columns:
        if pd.api.types.is_integer_dtype(x[col].dtype):
            x[col] = x[col].fillna(0).astype("int64")
        elif pd.api.types.is_float_dtype(x[col].dtype):
            x[col] = x[col].fillna(0.0).astype("float32")
        elif pd.api.types.is_datetime64_any_dtype(x[col].dtype):
            x[col] = pd.to_datetime(x[col], errors="coerce")
            x[col] = (x[col].astype("int64") / 10 ** 9).astype("float32")
        else:
            x[col] = x[col].fillna("").astype(str)

    return x


def _normalize_column(df: pd.DataFrame, column: str, datatype: str) -> pd.Series:
    dt = datatype.strip().lower()

    if dt == "integer":
        return pd.to_numeric(df[column], errors="coerce").astype("Int64")
    elif dt == "float":
        return pd.to_numeric(df[column], errors="coerce").astype("Float64")
    elif dt == "boolean":
        return df[column].astype("boolean")
    elif dt in {"datetime", "date"}:
        return pd.to_datetime(df[column], errors="coerce")
    elif dt == "duration":
        return pd.to_timedelta(df[column], errors="coerce")
    elif dt == "categorical":
        return df[column].astype("category")
    elif dt == "string":
        return df[column].astype("string")
    else:
        return df[column].astype("object")

