import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from utils import load_yaml


def _split_dataset(dataset, test_ratio=0.30):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def _train(dataset: pd.DataFrame, predictors: list[str], features: list[str] = None) -> tf.keras.Model:
    predictor = predictors[0]

    # Train with TF-Decision-Forests (TFDF) and export a TF SavedModel.

    train_df, test_df = _split_dataset(dataset, test_ratio=0.2)

    tf_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=predictor)
    tf_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=predictor)

    tfdf_model = tfdf.keras.GradientBoostedTreesModel()
    tfdf_model.fit(tf_train_ds)
    # tfdf_model.evaluate(tf_test_ds)

    return tfdf_model


def _build_artifact(model: tf.keras.Model, path: str) -> None:
    model.save(path)


def run_training(dataset: pd.DataFrame, config: dict):
    if 'admission_time' in dataset.columns:
        dataset['admission_time'] = pd.to_datetime(dataset['admission_time']).astype(
            int) / 10 ** 9  # Convert to seconds since epoch

    predictor = [f["name"] for f in config["output_schema"]["predictor"]]
    m = _train(dataset, features=[f["name"] for f in config["input_schema"]["features"]],
               predictors=predictor)
    _build_artifact(m, config["model"]["artifact"]["path"])

if __name__ == "__main__":
    data = pd.read_csv("input.csv")
    config = load_yaml("contract-training.yaml")
    result = run_training(data, config)
