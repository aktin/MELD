import os
import shutil
from datetime import datetime, timedelta

import isodate

import ModelEnvironment
import pandas as pd
from InternalDataLoader import execute_query
from ModelEnvironment import JobContext
from ModelEnvironment.docker_runtime import pull_image, delete_image
from ModelEnvironment.job_context import create_job_context, JobStatus
from ModelManager import load_contract
from Logger import setup_logger
from utils import construct_image_tag, safe_filename_from_url, download_file

logger = setup_logger("meld")


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
    end = datetime.now()
    job_context.logger.info(f"Query returned {len(data)} rows and took {end - start} seconds.")
    return data


def run_inference(contract_path: str = "contract.yaml") -> None:
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
    job_context = create_job_context(contract_path)
    default_path = "/resources/query.sql"
    try:
        job_context.log_event("Preparing inference", JobStatus.PREPARING)

        target_folder = job_context.input_data_path

        # check if query file was already mounted into docker container,
        if os.path.exists(default_path):
            job_context.logger.info(f"Query file was already mounted into docker container, overwriting query url from contract")
            query_file_name = "query.sql"
            shutil.copy(default_path, os.path.join(target_folder, query_file_name))
            job_context.logger.info(f"Moved query file to {target_folder}")
        else:
            query_file_name = safe_filename_from_url(job_context.query_url)
            job_context.logger.info(f"Downloading query file from {job_context.query_url}")
            download_file(job_context.query_url, target_folder)
            job_context.logger.info(f"Downloaded query file to {target_folder}")

        target_file = os.path.join(target_folder, query_file_name)

        start, end = _compute_time_window(job_context)
        params = {"start": start.isoformat(), "end": end.isoformat()}

        query = load_query(target_file, job_context)

        df = query_data(query, params, job_context)

        feature_cols = _validate_features(df, job_context)
        x = _normalize_features(df, feature_cols)

        ModelEnvironment.run_inference(x, job_context)
    except Exception as e:
        logger.exception(f"An exception occurred during inference: {e}")


# def run_training(contract_path: str, job_context: JobContext) -> str:
#     """
#     !!! Reine Testfunktion !!!
#
#     Runs the training pipeline, loading configuration and processing data
#     before invoking the model training.
#
#     :param contract_path: Path to the directory containing the contract
#         and related configuration files. Defaults to "artifact_df".
#     :type contract_path: str
#     :return: Path to the directory containing the artifacts of the training
#         process.
#     :rtype: str
#     """
#     config = load_contract(contract_path)
#
#     start, end = _compute_time_window(job_context)
#     params = {"start": start.isoformat(), "end": end.isoformat()}
#
#     query = load_query(query_path)
#
#     df = query_data(query, params)
#
#     feature_cols = _validate_features(df, job_context)
#     x = _normalize_features(df, feature_cols)
#
#     ModelEnvironment.run_training(x, artifact_path=artifact_path)


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
    anchor = scope.get("anchor")
    duration = scope.get("value")

    end = datetime.fromisoformat(anchor) if anchor else datetime.now()
    td: timedelta = isodate.parse_duration(duration, as_timedelta_if_possible=False).totimedelta(end=end)
    start = end + td
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
        image = construct_image_tag(load_contract(contract_path))
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
        image = construct_image_tag(load_contract(contract_path))
        delete_image(image)
    except Exception as e:
        logger.exception(f"An exception occurred during runtime removal: {e}")


def _validate_features(df: pd.DataFrame, job_context: JobContext) -> list[str]:
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
    required = [f["name"] for f in features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Query result is missing required feature columns: {', '.join(missing)}. "
            f"Ensure your SQL produces columns matching contract.yaml input_schema.features."
        )
    return required


def _normalize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
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
