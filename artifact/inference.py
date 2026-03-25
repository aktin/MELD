import io
import sys
from contextlib import redirect_stdout

import keras
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from utils import load_yaml


def _cast_series_for_model(series: pd.Series, spec: dict) -> tf.Tensor:
    expected_type = spec["datatype"].lower()

    if expected_type in ("float", "datetime"):
        # Preserve numeric meaning, but cast to the model's expected floating type
        return tf.convert_to_tensor(pd.to_numeric(series, errors="coerce").to_numpy(), dtype=tf.float32)

    if expected_type == "integer":
        # Use pandas nullable integers first to preserve missing values semantics
        values = pd.to_numeric(series, errors="coerce").astype("Int64")
        return tf.convert_to_tensor(values.to_numpy(dtype="int64", na_value=0), dtype=tf.int64)

    if expected_type == "bool":
        return tf.convert_to_tensor(series.astype("boolean").fillna(False).to_numpy(), dtype=tf.bool)

    # Default: treat as string/categorical-like
    return tf.convert_to_tensor(series.astype("string").fillna("").to_numpy(), dtype=tf.string)


def _to_tensor_inputs(df: pd.DataFrame, config: dict) -> dict[str, tf.Tensor]:
    inputs = {}
    features = config["input_schema"]["features"]

    for col in df.columns:
        spec = [f for f in features if f["name"] == col][0]
        inputs[col] = _cast_series_for_model(df[col], spec)

    return inputs


def _extract_prediction_series(raw_predictions, expected_len: int) -> pd.Series:
    if isinstance(raw_predictions, dict):
        first_value = next(iter(raw_predictions.values()))
        values = first_value.numpy().reshape(-1)
    elif hasattr(raw_predictions, "numpy"):
        values = raw_predictions.numpy().reshape(-1)
    else:
        values = pd.Series(raw_predictions).to_numpy().reshape(-1)

    if len(values) != expected_len:
        raise ValueError(
            f"Prediction row count mismatch: expected {expected_len}, got {len(values)}."
        )

    return pd.Series(values)


def _load_decision_tree(path: str) -> keras.layers.TFSMLayer:
    import tensorflow_decision_forests
    model = tf.saved_model.load(path)
    return model.signatures["serving_default"]


def run_inference(df: pd.DataFrame, config: dict) -> DataFrame:
    print(df.to_csv(index=False), file=sys.stderr)
    inputs = _to_tensor_inputs(df, config)

    model_contract_path = config["model"]["artifact"]["path"]
    model_layer = _load_decision_tree(model_contract_path)

    raw_predictions = model_layer(**inputs)
    prediction_series = _extract_prediction_series(raw_predictions, len(df))

    predictor_name = config["output_schema"]["predictor"][0]["name"]
    result_df = df.copy()

    result_df[predictor_name] = prediction_series

    return result_df


if __name__ == "__main__":
    raw_input = sys.stdin.read()
    data = pd.read_csv(io.StringIO(raw_input))
    config = load_yaml("contract.yaml")
    with redirect_stdout(sys.stderr):
        result = run_inference(data, config)
    result.to_csv(sys.stdout, index=False)
