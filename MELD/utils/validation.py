import pandas as pd


def validate_feature_datatypes(df: pd.DataFrame, features: list[dict]) -> None:
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


def validate_required_features(df: pd.DataFrame, features: list[dict]) -> None:
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


def get_unexpected_features(df: pd.DataFrame, features: list[dict]) -> list[str]:
    """

    """
    feature_names = [f["name"] for f in features]
    unexpected_cols = [col for col in df.columns if col not in feature_names]

    return unexpected_cols