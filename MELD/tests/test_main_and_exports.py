import unittest
from unittest.mock import MagicMock, patch

import test_support  # noqa: F401

import main
from ModelEnvironment import ExecutionStatus
from ModelManager import Contract
from ModelManager.config_loader import load_contract


class MainAndExportsTest(unittest.TestCase):
    @patch("main.app.run")
    def test_main_starts_rest_server(self, run):
        main.main()
        run.assert_called_once_with(host="127.0.0.1", port=main.API_PORT)

    def test_config_loader_delegates_to_contract_parser(self):
        expected = MagicMock()
        with patch.object(Contract, "from_yaml", return_value=expected) as parser:
            actual = load_contract("contract.yaml")

        self.assertIs(actual, expected)
        parser.assert_called_once_with("contract.yaml")

    def test_model_environment_lazy_exports_and_unknown_attribute(self):
        import ModelEnvironment

        with patch("ModelEnvironment.inference.InferenceRunner", autospec=True) as runner:
            # The export is resolved from the inference module on first access.
            exported = ModelEnvironment.InferenceRunner
        self.assertIsNotNone(exported)
        with self.assertRaises(AttributeError):
            getattr(ModelEnvironment, "missing_export")


if __name__ == "__main__":
    unittest.main()
