import os

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_yaml


def _split_dataset(dataset, test_ratio=0.30):
    """Splits a pandas dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def _preprocess_dataframe(dataset: pd.DataFrame, feature_names: list[str], label_name: str):
    dataset = dataset.copy()

    if "admission_time" in dataset.columns:
        dataset["admission_time"] = pd.to_datetime(dataset["admission_time"], errors="coerce")
        dataset["admission_time"] = dataset["admission_time"].astype("int64") // 10 ** 9

    # Keep only selected columns and drop rows with missing values
    dataset = dataset[feature_names + [label_name]].dropna()

    X = dataset[feature_names].copy()
    y = dataset[label_name].astype(str).copy()

    return X, y


def _train(dataset: pd.DataFrame, predictors: list[str], features: list[str]) -> tf.keras.Model:
    label_name = predictors[0]

    train_df, test_df = _split_dataset(dataset, test_ratio=0.2)
    x_train, y_train = _preprocess_dataframe(train_df, features, label_name)
    x_test, y_test = _preprocess_dataframe(test_df, features, label_name)

    # Build label encoding from training labels
    label_lookup = tf.keras.layers.StringLookup(
        output_mode="int",
        num_oov_indices=0
    )
    label_lookup.adapt(y_train)

    num_classes = label_lookup.vocabulary_size()

    inputs = []
    encoded_features = []

    for feature_name in features:
        col = x_train[feature_name]

        # Numeric features
        if pd.api.types.is_numeric_dtype(col):
            inp = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.float32)
            norm = tf.keras.layers.Normalization()
            norm.adapt(np.array(x_train[feature_name]).reshape(-1, 1))
            encoded = norm(inp)

        # String / categorical features
        else:
            inp = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)
            lookup = tf.keras.layers.StringLookup(output_mode="one_hot")
            lookup.adapt(np.array(x_train[feature_name], dtype=str))
            encoded = lookup(inp)

        inputs.append(inp)
        encoded_features.append(encoded)

    x = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    def df_to_dataset(x_df, y_series, shuffle=False, batch_size=32):
        x_dict = {name: np.array(x_df[name]) for name in x_df.columns}
        if num_classes == 2:
            y_arr = (y_series == label_lookup.get_vocabulary()[1]).astype(np.float32).values
        else:
            y_arr = label_lookup(y_series).numpy()

        ds = tf.data.Dataset.from_tensor_slices((x_dict, y_arr))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x_df))
        return ds.batch(batch_size)

    train_ds = df_to_dataset(x_train, y_train, shuffle=True)
    test_ds = df_to_dataset(x_test, y_test, shuffle=False)

    model.fit(train_ds, validation_data=test_ds, epochs=20)
    model.evaluate(test_ds)

    return model


def _build_artifact(model: tf.keras.Model, path: str) -> None:
    # if not os.path.exists(path):
    #     os.makedirs(path)
    model.save(path)


def run_training(dataset: pd.DataFrame, config: dict):
    predictor = [f["name"] for f in config["output_schema"]["predictor"]]
    features = [f["name"] for f in config["input_schema"]["features"] if f["name"] not in predictor]

    m = _train(dataset, features=features, predictors=predictor)
    _build_artifact(m, "../artifact/model.keras")


if __name__ == "__main__":
    data = pd.read_csv("./input.csv")
    config = load_yaml("../resources/contract-training.yaml")
    run_training(data, config)