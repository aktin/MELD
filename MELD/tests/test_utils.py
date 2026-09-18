import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import test_support  # noqa: F401

from ModelManager.contract_models import Feature
from utils import (
    get_unexpected_features,
    load_yaml,
    to_yaml,
    validate_feature_datatypes,
    validate_required_features,
)
from utils.utils import read_contract


class UtilsTest(unittest.TestCase):
    def test_load_yaml_accepts_stream_and_yaml_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.yaml"
            path.write_text("value: 1\n", encoding="utf-8")

            self.assertEqual(load_yaml(path), {"value": 1})
        self.assertEqual(load_yaml(StringIO("value: 1\n")), {"value": 1})

    def test_load_yaml_rejects_missing_and_non_yaml_paths(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("missing.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text("value: 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(path)

    def test_to_yaml_serializes_mapping(self):
        self.assertEqual(to_yaml({"value": 1}), "value: 1\n")

    def test_feature_validation_accepts_contract_objects_and_dicts(self):
        df = pd.DataFrame({"age": pd.Series([1], dtype="int64"), "name": ["A"]})
        features = [
            Feature(name="age", datatype="Int64", required=True),
            {"name": "name", "datatype": "string"},
        ]

        validate_required_features(df, features)
        validate_feature_datatypes(df, features)
        self.assertEqual(get_unexpected_features(df, features), [])

    def test_feature_validation_reports_missing_wrong_type_and_unexpected_columns(self):
        df = pd.DataFrame({"age": pd.Series([1.5], dtype="float64"), "extra": [1]})
        features = [{"name": "age", "datatype": "Int64", "required": True}]

        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_required_features(pd.DataFrame(), features)
        with self.assertRaisesRegex(ValueError, "expected to be of type integer"):
            validate_feature_datatypes(df, features)
        self.assertEqual(get_unexpected_features(df, features), ["extra"])

    def test_read_contract_uses_root_contract_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            contract_dir = Path(directory) / "contracts" / "contract-id"
            contract_dir.mkdir(parents=True)
            contract_dir.joinpath("contract.yaml").write_text(
                "contract:\n"
                "  name: example\n"
                "  description: Example\n"
                "  version: '1'\n"
                "runtime:\n"
                "  framework: sklearn\n"
                "  image:\n"
                "    name: image\n"
                "    tag: latest\n"
                "    digest: sha256:x\n"
                "input_schema:\n"
                "  temporal_scope:\n"
                "    type: relative\n"
                "    value: P1D\n"
                "    anchor: '2020-01-01T00:00:00Z'\n"
                "  features:\n"
                "    - name: age\n"
                "      datatype: Int64\n"
                "  query:\n"
                "    type: sql\n"
                "    statement: SELECT age\n"
                "output_schema:\n"
                "  type: csv\n"
                "  predictor:\n"
                "    - name: prediction\n"
                "      datatype: Float64\n",
                encoding="utf-8",
            )
            with patch("utils.utils.ROOT_DIR", directory):
                contract = read_contract("contract-id")

        self.assertEqual(contract.contract.name, "example")


if __name__ == "__main__":
    unittest.main()
