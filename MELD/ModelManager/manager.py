import os
from datetime import datetime, timedelta

import isodate
import pandas as pd

import ModelEnvironment
from InternalDataLoader import execute_query
from ModelManager import load_contract
from utils import resolve_path


def load_query(path: str) -> str:
    with open(path, "r") as file:
        query = file.read()
        return query


def get_data(query: str, params: dict) -> pd.DataFrame:
    return execute_query(query, params)


def run_inference(contract_path: str = "contract.yaml", output_csv_path: str = "predictions.csv") -> str:
    artifact_path = os.path.dirname(contract_path)
    config = load_contract(contract_path)

    start, end = _compute_time_window(config)
    params = {"start": start.isoformat(), "end": end.isoformat()}

    query_contract_path = os.path.join(artifact_path, config["input_schema"]["query"]["path"])
    query_path = resolve_path(query_contract_path)
    query = load_query(query_path)

    df = get_data(query, params)

    feature_cols = _validate_features(df, config)
    x = _normalize_features(df, feature_cols)

    result_df = ModelEnvironment.run_inference(x, artifact_path=artifact_path)

    output_path = resolve_path(output_csv_path) if not os.path.isabs(output_csv_path) else output_csv_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_df.to_csv(output_path, index=False)

    return output_path

def run_training(artifact_path: str = "artifact") -> str:
    artifact_path = os.path.dirname(artifact_path)
    config = load_contract(os.path.join(artifact_path, "contract-training.yaml"))

    start, end = _compute_time_window(config)
    params = {"start": start.isoformat(), "end": end.isoformat()}

    query_contract_path = os.path.join(artifact_path, config["input_schema"]["query"]["path"])
    query_path = resolve_path(query_contract_path)
    query = load_query(query_path)

    df = get_data(query, params)

    feature_cols = _validate_features(df, config)
    x = _normalize_features(df, feature_cols)

    ModelEnvironment.run_training(x, artifact_path=artifact_path)


def _compute_time_window(config: dict) -> tuple[datetime, datetime]:
    scope = config["input_schema"]["temporal_scope"]
    anchor = scope.get("anchor")
    duration = scope.get("value")

    end = datetime.fromisoformat(anchor) if anchor else datetime.now()
    td: timedelta = isodate.parse_duration(duration).totimedelta(end=end)
    start = end + td
    return start, end


def _validate_features(df: pd.DataFrame, config: dict) -> list[str]:
    features = config["input_schema"]["features"]
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