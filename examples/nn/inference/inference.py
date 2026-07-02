import keras
import pandas as pd
import tensorflow as tf
import yaml
from pandas import DataFrame


def _cast_series_for_model(series: pd.Series, spec: dict) -> tf.Tensor:
    expected_type = spec["datatype"].lower()

    if expected_type.startswith(("float", "datetime")):
        # Preserve numeric meaning, but cast to the model's expected floating type
        return tf.convert_to_tensor(pd.to_numeric(series, errors="coerce").to_numpy(), dtype=tf.float32)

    if expected_type.startswith("int"):
        # Use pandas nullable integers first to preserve missing values semantics
        values = pd.to_numeric(series, errors="coerce").astype("Int64")
        return tf.convert_to_tensor(values.to_numpy(dtype="int64", na_value=0), dtype=tf.int64)

    if expected_type == "boolean":
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


def _load_neural_net(path: str) -> keras.layers.TFSMLayer:
    print(f"Loading neural net")
    model = tf.keras.models.load_model(path)
    print(f"Model loaded")
    return model


def run_inference(df: pd.DataFrame, config: dict) -> DataFrame:
    print(f"Inference with {len(df)} rows.")
    inputs = _to_tensor_inputs(df, config)

    model_path = "/artifact/model.keras"
    model_layer = _load_neural_net(model_path)

    print(f"Run inference")
    raw_predictions = model_layer(inputs)
    prediction_series = _extract_prediction_series(raw_predictions, len(df))

    predictor_name = config["output_schema"]["predictor"][0]["name"]
    result_df = df.copy()

    result_df[predictor_name] = prediction_series

    print(f"Inference done")

    return result_df


if __name__ == "__main__":
    with open("/input/contract.yaml", "r") as file:
        config = yaml.safe_load(file)

    with open("/input/input.csv", "r") as f:
        result = run_inference(pd.read_csv(f), config)

    result.to_csv("/output/output.csv", index=False)
