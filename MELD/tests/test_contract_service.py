import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import test_support
from ModelManager import Contract, ContractService


class ContractServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = ContractService()

    def tearDown(self):
        self.service._pull_executor.shutdown(wait=True, cancel_futures=True)

    def contract(self, **overrides):
        return Contract.from_dict(test_support.contract_data(**overrides))

    def test_parse_contract_rejects_empty_payload(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.service.parse_contract("  \n")

    def test_pull_runtime_image_passes_progress_callback_and_optional_token(self):
        progress = MagicMock()
        with patch("ModelEnvironment.docker_runtime.pull_image") as pull:
            self.service._pull_runtime_image("image", progress, "token")
            self.service._pull_runtime_image("image", progress)

        pull.assert_any_call("image", progress_callback=progress, registry_api_key="token")
        pull.assert_any_call("image", progress_callback=progress)

    def test_image_reference_can_omit_digest_when_configured(self):
        contract = self.contract()
        with patch("ModelManager.contract_models.PULL_WITH_DIGEST", False):
            self.assertEqual(contract.runtime.image.construct_image_ref(), "example/runtime:1.0.0")

    def test_create_and_retrieve_contract_persists_yaml_and_status(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=True):
            self.assertTrue(self.service.create_contract(contract))
            loaded = self.service.get_contract(contract.id)
            info = self.service.get_contract_info(contract.id)
            contracts_info = self.service.get_contracts_info()

        self.assertEqual(loaded.to_dict(), contract.to_dict())
        self.assertEqual(info["status"], "ready")
        self.assertEqual(contracts_info, [
            {"id": contract.id, "status": "ready"}
        ])

    def test_duplicate_contract_is_rejected_but_fingerprint_collision_is_not(self):
        first = self.contract()
        changed = self.contract(
            contract={
                "name": "example-contract",
                "description": "Different",
                "version": "1.0.0",
            }
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=True):
            self.service.create_contract(first)
            with self.assertRaises(FileExistsError):
                self.service.create_contract(self.contract())
            with patch.object(
                Contract, "fingerprint", new_callable=PropertyMock,
                return_value=first.fingerprint,
            ):
                self.service.create_contract(changed)

        self.assertNotEqual(first.id, changed.id)

    def test_failed_duplicate_restarts_image_pull(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=False), patch.object(
            self.service, "_start_image_pull"
        ) as start_pull:
            self.service.create_contract(contract)
            self.service._track_pull_progress(contract.id, "failed", "pull failed")
            retry = self.contract()
            self.assertFalse(self.service.create_contract(retry))

        self.assertEqual(retry.id, contract.id)
        self.assertEqual(start_pull.call_count, 2)

    def test_progress_aggregates_layers_and_preserves_failure_information(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ):
            self.service._track_pull_progress("id", "pulling")
            self.service._track_pull_progress(
                "id", {"status": "Downloading", "id": "one", "progressDetail": {"current": 25, "total": 100}}
            )
            self.service._track_pull_progress(
                "id", {"status": "Downloading", "id": "two", "progressDetail": {"current": 100, "total": 100}}
            )
            progress = self.service._read_progress("id")
            self.assertEqual(progress["progress"], 62.5)
            self.service._track_pull_progress("id", "failed", "network")
            progress = self.service._read_progress("id")

        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["error"], "network")

    def test_pull_retries_with_capped_backoff_and_marks_available(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"MELD_CONTRACT_DIRECTORY": directory, "MELD_PULL_RETRY_DELAY_SECONDS": "2"},
        ):
            self.assertEqual(self.service._pull_retry_delay(1), 2)
            self.assertEqual(self.service._pull_retry_delay(10), 60)
            with patch.object(
                self.service, "_pull_runtime_image", side_effect=[RuntimeError("retry"), None]
            ) as pull, patch.object(self.service, "_pull_retry_delay", return_value=0):
                self.service._pull_image("id", "image")
            progress = self.service._read_progress("id")

        self.assertEqual(pull.call_count, 2)
        self.assertEqual(progress["status"], "available")

    def test_invalid_progress_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ):
            path = Path(directory) / "id" / "status.json"
            path.parent.mkdir()
            path.write_text("not-json", encoding="utf-8")

            self.assertIsNone(self.service._read_progress("id"))


if __name__ == "__main__":
    unittest.main()
