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
        "labels": [{"name": "prediction", "datatype": "Float64"}],
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

    def test_schema_validation_rejects_missing_labels(self):
        invalid = {**CONTRACT_DATA, "output_schema": {"type": "csv"}}

        with self.assertRaises(ValidationError):
            Contract.from_dict(invalid)

    def test_schema_validation_allows_contract_without_schedule(self):
        self.assertIsInstance(Contract.from_dict(CONTRACT_DATA), Contract)

    def test_schema_validation_accepts_cron_schedule(self):
        contract = Contract.from_dict(
            {
                **CONTRACT_DATA,
                "schedule": {"expression": "0 0 * * *"},
            }
        )

        self.assertEqual(contract.schedule.expression, "0 0 * * *")

    def test_schema_validation_rejects_cron_without_expression(self):
        invalid = {**CONTRACT_DATA, "schedule": {}}

        with self.assertRaises(ValidationError):
            Contract.from_dict(invalid)

    def test_schema_validation_rejects_schedule_type(self):
        invalid = {
            **CONTRACT_DATA,
            "schedule": {"type": "cron", "expression": "0 0 * * *"},
        }

        with self.assertRaises(ValidationError):
            Contract.from_dict(invalid)

    def test_schema_validation_rejects_interval_fields(self):
        invalid = {**CONTRACT_DATA, "schedule": {"expression": "0 0 * * *", "days": 1}}

        with self.assertRaises(ValidationError):
            Contract.from_dict(invalid)

    def test_id_uses_contract_name_and_version(self):
        contract_data = {**CONTRACT_DATA, "contract": {**CONTRACT_DATA["contract"], "id": "provided"}}
        contract = Contract.from_dict(contract_data)

        self.assertEqual(contract.id, "example-contract-1.0.0")
        self.assertEqual(contract.assign_id(force=True), "example-contract-1.0.0")
        self.assertEqual(contract.contract.id, contract.id)

    def test_contracts_are_equal_when_name_and_version_match(self):
        first = Contract.from_dict(CONTRACT_DATA)
        second = Contract.from_dict(
            {
                **CONTRACT_DATA,
                "contract": {
                    **CONTRACT_DATA["contract"],
                    "description": "Different",
                    "id": "another",
                },
            }
        )
        different_version = Contract.from_dict(
            {**CONTRACT_DATA, "contract": {**CONTRACT_DATA["contract"], "version": "2.0.0"}}
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_version)

    def test_image_reference_contains_digest(self):
        contract = Contract.from_dict(CONTRACT_DATA)

        self.assertEqual(
            contract.runtime.image.construct_image_ref(),
            "example/runtime:1.0.0@sha256:example",
        )


if __name__ == "__main__":
    unittest.main()
