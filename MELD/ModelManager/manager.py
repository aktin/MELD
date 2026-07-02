import os
import shutil
from datetime import datetime, timedelta

import isodate

import ModelEnvironment
import pandas as pd
from InternalDataLoader import execute_query
from Logger.logger import get_meld_logger
from ModelEnvironment import JobContext
from ModelEnvironment.docker_runtime import pull_image, delete_image, ensure_image_exists
from ModelEnvironment.job_context import JobStatus
from ModelManager import load_contract
from utils import construct_image_ref, safe_filename_from_url, download_file

logger = get_meld_logger()


def load_query(file_path: str, job_context: JobContext) -> str:
    """
    Loads the SQL query from the specified file.

    Parameters:
    file_path: str
        The path to the SQL file to be loaded.

    job_context: JobContext
        The context object that includes the logger to be used for
        logging file loading operations.

    Raises:
    FileNotFoundError
        If the specified file path does not exist.

    ValueError
        If the provided file path is not a SQL file.

    Returns:
    str
        The content of the SQL file as a string.
    """
    job_context.logger.info(f"Loading query from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.endswith(".sql"):
        raise ValueError(f"The file {file_path} is not a SQL file. ")

    with open(file_path, "r") as file:
        query = file.read()
        return query


def query_data(query: str, params: dict, job_context: JobContext) -> pd.DataFrame:
    """
    Executes a SQL query and returns the resulting data as a DataFrame.

    Arguments:
    query: A string representing the SQL query to execute.
    params: A dictionary of parameters to use in the query.
    job_context: An instance of JobContext used for logging and contextual information
    relevant to the job execution.

    Returns:
    A pandas DataFrame containing the query results.
    """
    job_context.logger.info(f"Executing query")
    start = datetime.now()
    data = execute_query(query, params)
    timespan = datetime.now() - start
    job_context.logger.info(f"Query returned {len(data)} rows and took {timespan.total_seconds():.3f} seconds.")
    return data


def run_inference(contract_path: str) -> None:
    """
    Run the inference workflow using the given contract file.

    Parameters:
    contract_path: str
        Path to the contract file, default is "contract.yaml".

    Raises:
    Exception
        Logs the exception if an error occurs during the inference process.

    Returns:
    None
    """
    job_context = JobContext.create_job_context(contract_path)
    try:
        job_context.log_event("Preparing inference", JobStatus.PREPARING)

        ensure_image_exists(job_context)

        start, end = _compute_time_window(job_context)
        params = {"start": start.isoformat(), "end": end.isoformat()}

        df = query_data(job_context, params, job_context)

        feature_cols = _validate_features(df, job_context)
        x = _normalize_features(df, feature_cols)

        ModelEnvironment.run_inference(x, job_context)
    except Exception as e:
        job_context.logger.exception(f"An exception occurred during inference: {e}")


def _compute_time_window(job_context: JobContext) -> tuple[datetime, datetime]:
    """
    Compute the temporal window based on the input schema's temporal scope.

    Args:
        job_context (JobContext): The context of the job containing the contract
        metadata, which includes the temporal scope specifications.

    Returns:
        tuple[datetime, datetime]: A tuple containing the start and end datetime
        objects representing the temporal window.
    """
    scope = job_context.contract["input_schema"]["temporal_scope"]
    if scope["type"] == "absolute":
        start = datetime.fromisoformat(scope["start"])
        end = datetime.fromisoformat(scope["end"])
    else:
        anchor = scope.get("anchor")
        duration = scope.get("value")

        # force absolute value duration gets subtracted from anchor, negative values would add to anchor and cause that start > end
        duration = duration[1:] if duration.startswith("-") else duration

        end = datetime.fromisoformat(anchor) if anchor else datetime.now()
        td: timedelta = isodate.parse_duration(duration, as_timedelta_if_possible=False).totimedelta(end=end)

        start = end - td

    if start >= end:
        raise ValueError("Start time must be before end time")

    job_context.logger.info(f"Temporal window start: {start.isoformat()}, end: {end.isoformat()}")
    return start, end


def pull_runtime(contract_path):
    """
    Pulls a runtime image based on the specified contract file.

    Parameters:
    contract_path: str
        The file path to the contract that specifies the runtime information.

    Raises:
    Exception
        Raised if an error occurs during the image tag construction or image
        pulling process.
    """
    try:
        image = construct_image_ref(load_contract(contract_path))
        pull_image(image)
    except Exception as e:
        logger.exception(f"An exception occurred during runtime pull: {e}")


def remove_runtime(contract_path):
    """
    Removes the runtime associated with a given contract.

    Args:
        contract_path (str): The file path to the contract.

    Raises:
        Exception: If an error occurs during image construction or
        deletion, it is caught and logged.
    """
    try:
        image = construct_image_ref(load_contract(contract_path))
        delete_image(image)
    except Exception as e:
        logger.exception(f"An exception occurred during runtime removal: {e}")

def _validate_feature_datatypes(df: pd.DataFrame, features: list[dict]) -> None:
    """
    Validate that DataFrame columns conform to the datatypes defined in the
    feature contract.

    Each feature definition is expected to contain a ``name`` and ``datatype``
    field. For every DataFrame column that has a matching feature definition,
    the column dtype is validated against the expected Pandas datatype category.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    features : list[dict]
        A list of feature definitions. Each feature must contain at least
        ``name`` and ``datatype`` keys.

    Raises
    ------
    ValueError
        If a column's dtype does not match the expected datatype defined in the
        feature contract.
    """
    for col in df.columns:
        feature = next((f for f in features if f["name"] == col), None)
        if not feature:
            continue

        if feature["datatype"].startswith("string") and not pd.api.types.is_string_dtype(df[col].dtype):
            raise ValueError(f"Column {col} is expected to be of type string, but is of type {df[col].dtype}")
        elif feature["datatype"].startswith("int") and not pd.api.types.is_integer_dtype(df[col].dtype):
            raise ValueError(f"Column {col} is expected to be of type integer, but is of type {df[col].dtype}")
        elif feature["datatype"].startswith("float") and not pd.api.types.is_float_dtype(df[col].dtype):
            raise ValueError(f"Column {col} is expected to be of type float, but is of type {df[col].dtype}")
        elif feature["datatype"].startswith("datetime") and not pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            raise ValueError(f"Column {col} is expected to be of type datetime, but is of type {df[col].dtype}")
        elif feature["datatype"].startswith("boolean") and not pd.api.types.is_bool_dtype(df[col].dtype):
            raise ValueError(f"Column {col} is expected to be of type boolean, but is of type {df[col].dtype}")

def _validate_required_features(df: pd.DataFrame, features: list[dict]) -> None:
    """
    Validate that all required features defined in the feature contract are
    present in the DataFrame.

    Features are considered required when their definition contains
    ``required=True``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    features : list[dict]
        A list of feature definitions. Each feature may contain a ``required``
        flag indicating whether the corresponding column must exist.

    Raises
    ------
    ValueError
        If one or more required columns are missing from the DataFrame.
    """
    required_cols = [f["name"] for f in features if f.get("required")]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

def _validate_features(df: pd.DataFrame, job_context: JobContext) -> list[dict]:
    """
    Validates the presence of required feature columns in a given dataframe against the input schema.

    Parameters:
    df : pd.DataFrame
        The dataframe to validate.
    job_context : JobContext
        The context that includes the contract and logger configuration.

    Returns:
    list[str]
        A list of required feature column names.

    Raises:
    ValueError
        If the required feature columns are missing from the dataframe.
    """
    job_context.logger.info(f"Validating features")
    features = job_context.contract["input_schema"]["features"]

    _validate_required_features(df, features)

    _validate_feature_datatypes(df, features)

    return features


def _normalize_features(df: pd.DataFrame, feature_cols: list[dict]) -> pd.DataFrame:
    """
    Normalizes the specified feature columns in the provided DataFrame.

    Parameters:
    df : pd.DataFrame
        Input DataFrame containing the data to be normalized.
    feature_cols : list[str]
        List of feature column names to be normalized within the DataFrame.

    Returns:
    pd.DataFrame
        A new DataFrame where the specified feature columns are normalized
        according to their data types.
    """
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
