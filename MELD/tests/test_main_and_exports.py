import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_support  # noqa: F401

import main
from ModelEnvironment import ExecutionStatus
from ModelManager import Contract
from ModelManager.config_loader import load_contract


class MainAndExportsTest(unittest.TestCase):
    def test_contract_path_accepts_relative_paths_and_rejects_escape(self):
        with patch("main.CONTRACTS_DIR", "/contracts"):
            self.assertEqual(main._contract_path("nested/contract.yaml"), "/contracts/nested/contract.yaml")
            with self.assertRaises(ValueError):
                main._contract_path("../outside.yaml")
            with self.assertRaises(ValueError):
                main._contract_path("/absolute.yaml")

    def test_cli_dispatches_pull_run_remove_and_delete_alias(self):
        contract = MagicMock(spec=Contract)
        with patch("main._contract_path", return_value="contract.yaml") as path, patch(
            "ModelManager.load_contract", return_value=contract
        ) as load, patch("ModelManager.pull_runtime") as pull, patch(
            "ModelManager.run_inference"
        ) as run, patch("ModelManager.remove_runtime") as remove:
            self.assertEqual(main.main(["pull", "contract.yaml"]), 0)
            self.assertEqual(main.main(["run", "contract.yaml"]), 0)
            self.assertEqual(main.main(["remove", "contract.yaml"]), 0)
            self.assertEqual(main.main(["delete", "contract.yaml"]), 0)

        self.assertEqual(path.call_count, 4)
        self.assertEqual(load.call_count, 4)
        pull.assert_called_once_with(contract)
        run.assert_called_once_with(contract)
        self.assertEqual(remove.call_count, 2)

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
