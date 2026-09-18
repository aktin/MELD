import unittest
from io import StringIO

import test_support  # noqa: F401
from jsonschema import ValidationError
import yaml

from ModelManager import Contract


CONTRACT_DATA = {
    "schema_version": "1",
    "contract": {
        "name": "example-contract",
        "description": "Example",
        "version": "1.0.0",
    },
    "runtime": {
        "framework": "sklearn",
        "image": {
            "name": "example/runtime",
            "tag": "1.0.0",
            "digest": "sha256:example",
        },
        "environment_variables": {"MODE": "test"},
    },
    "input_schema": {
        "temporal_scope": {
            "type": "relative",
            "value": "P1D",
            "anchor": "2020-01-01T00:00:00Z",
        },
        "features": [{"name": "age", "datatype": "Int64", "required": True}],
        "query": {"type": "sql", "statement": "SELECT age FROM patients"},
    },
    "output_schema": {
        "type": "csv",
        "predictor": [{"name": "prediction", "datatype": "Float64"}],
    },
}


class ContractModelsTest(unittest.TestCase):
    def test_from_dict_builds_typed_nested_models_and_round_trips(self):
        data = {**CONTRACT_DATA, "custom": {"enabled": True}}

        contract = Contract.from_dict(data)

        self.assertEqual(contract.runtime.image.name, "example/runtime")
        self.assertTrue(contract.input_schema.features[0].required)
        self.assertEqual(contract.custom, {"enabled": True})
        self.assertEqual(contract.to_dict(), data)

    def test_from_yaml_accepts_text_stream(self):
        contract = Contract.from_yaml(StringIO(yaml.safe_dump(CONTRACT_DATA)))

        self.assertIsInstance(contract, Contract)

    def test_schema_validation_rejects_missing_required_sections(self):
        invalid = {key: value for key, value in CONTRACT_DATA.items() if key != "runtime"}

        with self.assertRaises(ValidationError):
            Contract.from_dict(invalid)

    def test_id_is_generated_lazily_and_can_be_forced(self):
        contract_data = {**CONTRACT_DATA, "contract": {**CONTRACT_DATA["contract"], "id": "provided"}}
        contract = Contract.from_dict(contract_data)

        self.assertEqual(contract.id, "provided")
        self.assertNotEqual(contract.assign_id(force=True), "provided")
        self.assertEqual(contract.contract.id, contract.id)

    def test_canonical_content_and_fingerprint_ignore_id(self):
        first = Contract.from_dict(CONTRACT_DATA)
        second = Contract.from_dict(
            {**CONTRACT_DATA, "contract": {**CONTRACT_DATA["contract"], "id": "another"}}
        )

        self.assertEqual(first.canonical_content(), second.canonical_content())
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_image_reference_contains_digest(self):
        contract = Contract.from_dict(CONTRACT_DATA)

        self.assertEqual(
            contract.runtime.image.construct_image_ref(),
            "example/runtime:1.0.0@sha256:example",
        )


if __name__ == "__main__":
    unittest.main()
