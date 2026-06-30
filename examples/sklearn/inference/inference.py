import joblib
import pandas as pd
import yaml
from pandas import DataFrame


def _preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    features = config["input_schema"]["features"]
    df = df.copy()

    for spec in features:
        col = spec["name"]
        if col not in df.columns:
            continue

        if spec["datatype"] == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10 ** 9

    return df[[f["name"] for f in features if f["name"] in df.columns]]


def run_inference(df: pd.DataFrame, config: dict) -> DataFrame:
    print(f"Inference with {len(df)} rows.")

    X = _preprocess_features(df, config)

    print("Loading model pipeline")
    pipeline = joblib.load("/artifact/model.joblib")
    print("Model loaded")

    print("Run inference")
    predictions = pipeline.predict(X)
    print("Inference done")

    predictor_name = config["output_schema"]["predictor"][0]["name"]
    result_df = df.copy()
    result_df[predictor_name] = predictions

    return result_df


if __name__ == "__main__":
    with open("/input/contract.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("/input/input.csv", "r") as f:
        result = run_inference(pd.read_csv(f), config)

    result.to_csv("/output/output.csv", index=False)
