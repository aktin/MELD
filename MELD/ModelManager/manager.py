import json
import os
import zipfile
from datetime import datetime, timedelta

import isodate

import ModelEnvironment
import pandas as pd
from ExecutionJournal.journal import ExecutionJournal
from ExecutionMonitor.metrics import Metrics
from ExecutionMonitor.monitor import ExecutionMonitor
from InternalDataLoader import execute_query
from Logger.logger import get_meld_logger
from ModelEnvironment import JobContext
from ModelEnvironment.docker_runtime import pull_image, delete_image, ensure_image_exists
from ModelEnvironment.job_context import ContextProvider
from ModelManager import load_contract
from utils import construct_image_ref, validate_required_features, validate_feature_datatypes, get_unexpected_features

logger = get_meld_logger()


def query_data(
        job_context: JobContext,
        params: dict,
        monitor: ExecutionMonitor,
        journal: ExecutionJournal,
) -> pd.DataFrame:
    """
    Executes a SQL query and returns the resulting data as a DataFrame.

    Arguments:
    params: A dictionary of parameters to use in the query.
    job_context: An instance of JobContext used for logging and contextual information
    relevant to the job execution.
    monitor: Execution monitor used for query timing and metrics.

    Returns:
    A pandas DataFrame containing the query results.
    """
    job_context.logger.info("Executing query")
    journal.append_entry("query_started", "Executing the DWH query")
    monitor.start_query_execution_time()
    try:
        data = execute_query(job_context, params)
        timespan = monitor.stop_query_execution_time()
    except Exception as error:
        journal.append_entry("query_failed", str(error), error_type=type(error).__name__)
        raise

    monitor.update_metric_value(Metrics.QUERY_RESULT_ROW_COUNT, len(data))
    job_context.logger.info(f"Query returned {len(data)} rows and took {timespan.total_seconds():.3f} seconds.")
    journal.append_entry(
        "query_finished",
        "DWH query completed",
        row_count=len(data),
        duration_seconds=timespan.total_seconds(),
    )
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
    provider = ContextProvider(job_context)
    journal = ExecutionJournal(provider)
    monitor = ExecutionMonitor(provider)

    journal.append_entry(
        "job_created",
        "Inference job created",
        contract_path=contract_path,
        contract_id=job_context.contract["contract"]["id"],
        contract_version=job_context.contract["contract"]["version"],
    )
    journal.append_entry("contract_validated", "Contract loaded and validated")

    monitor.start_total_execution_time()
    try:
        journal.append_entry("preparing_inference", "Preparing inference")

        ensure_image_exists(job_context, journal)

        start, end = _compute_time_window(job_context)
        params = {"start": start.isoformat(), "end": end.isoformat()}

        df = query_data(job_context, params, monitor, journal)

        feature_cols = _validate_features(df, job_context, journal)
        x = _normalize_features(df, feature_cols)
        journal.append_entry("input_prepared", "Inference input prepared", row_count=len(x), column_count=len(x.columns))

        ModelEnvironment.run_inference(x, job_context, monitor, journal)
    except Exception as error:
        journal.append_entry("job_failed", str(error), error_type=type(error).__name__)
        job_context.logger.exception(f"An exception occurred during inference: {error}")
    finally:
        monitor.stop_total_execution_time()

        zip_path = os.path.join(job_context.output_data_path, "summarized_execution.zip")
        pack_records(zip_path, job_context, monitor, journal)


def pack_records(path: str, job_context: JobContext, monitor: ExecutionMonitor, journal: ExecutionJournal) -> None:
    """
    Packs result files from an input archive into a gzipped tar file.
    """
    job_context.logger.info("Packing record files")

    mode = "a" if os.path.exists(path) else "w"
    with zipfile.ZipFile(path, mode=mode, compression=zipfile.ZIP_DEFLATED) as out_zip:
        out_zip.writestr("records/metrics.json", json.dumps(monitor.collect_metrics(), indent=2))
        out_zip.write(journal.journal_path, "records/journal.jsonl")


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



def _validate_features(
        df: pd.DataFrame,
        job_context: JobContext,
        journal: ExecutionJournal,
) -> list[dict]:
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
    journal.append_entry("input_validation_started", "Validating query result against input schema")
    features = job_context.contract["input_schema"]["features"]

    try:
        validate_required_features(df, features)
        validate_feature_datatypes(df, features)
    except Exception as error:
        journal.append_entry("input_validation_failed", str(error), error_type=type(error).__name__)
        raise

    unexpected_features = get_unexpected_features(df, features)
    if unexpected_features:
        job_context.logger.warning(f"Unexpected features: {', '.join(unexpected_features)}")
        journal.append_entry(
            "input_validation_warning",
            "Unexpected input columns were returned by the query",
            columns=unexpected_features,
        )

    journal.append_entry("input_validated", "Query result conforms to the input schema")

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
    feature_names = [f["name"] for f in feature_cols]
    x = df[feature_names].copy()

    for col in x.columns:
        if pd.api.types.is_integer_dtype(x[col].dtype):
            x[col] = x[col].fillna(0).astype("int64")
        elif pd.api.types.is_float_dtype(x[col].dtype):
            x[col] = x[col].fillna(0.0).astype("float32")
        elif pd.api.types.is_datetime64_any_dtype(x[col].dtype):
            x[col] = pd.to_datetime(x[col], errors="coerce")
            x[col] = (x[col].astype("int64") / 10 ** 9).astype("float32")
        elif pd.api.types.is_bool_dtype(x[col].dtype):
            x[col] = x[col].fillna(False).astype("int64")
        else:
            x[col] = x[col].fillna("").astype(str)

    return x
