import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_support
from ModelManager import Contract, ContractService


class ContractServiceTest(unittest.TestCase):
    def setUp(self):
        self.scheduler_service = MagicMock()
        self.execution_service = MagicMock()
        self.service = ContractService(
            scheduler_service=self.scheduler_service,
            execution_service=self.execution_service,
        )

    def tearDown(self):
        self.service._pull_executor.shutdown(wait=True, cancel_futures=True)

    def contract(self, **overrides):
        return Contract.from_dict(test_support.contract_data(**overrides))

    def test_requires_scheduler_and_execution_services(self):
        with self.assertRaises(TypeError):
            ContractService()
        with self.assertRaisesRegex(ValueError, "scheduler_service"):
            ContractService(None, self.execution_service)
        with self.assertRaisesRegex(ValueError, "execution_service"):
            ContractService(self.scheduler_service, None)

    def test_parse_contract_rejects_empty_payload(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.service.parse_contract("  \n")

    def test_create_contract_validates_schedule_before_storage(self):
        contract = self.contract()
        self.scheduler_service.validate_schedule.side_effect = ValueError("invalid schedule")

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available") as image_available:
            with self.assertRaisesRegex(ValueError, "invalid schedule"):
                self.service.register_contract(contract)

        image_available.assert_not_called()
        self.assertFalse(Path(directory, contract.id).exists())

    def test_pull_runtime_image_reads_optional_stored_auth(self):
        contract = self.contract()
        progress = MagicMock()
        auth_config = {"username": "user", "password": "token", "serveraddress": "registry.example"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch("ModelEnvironment.docker_runtime.pull_image") as pull:
            self.service._pull_runtime_image(contract, progress)
            self.service._store_contract(contract)
            self.service._store_docker_image_secret(auth_config, contract)
            self.service._pull_runtime_image(contract, progress)

        pull.assert_any_call(contract.runtime.image.image_ref, progress_callback=progress, auth_config=None)
        pull.assert_any_call(contract.runtime.image.image_ref, progress_callback=progress, auth_config=auth_config)

    def test_registry_credentials_are_persisted_and_used_after_restart(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=False), patch.object(
            self.service, "_start_image_pull"
        ) as start_pull:
            self.service.register_contract(contract, {"username": "user", "password": "token", "serveraddress": "registry.example"})
            start_pull.assert_called_once_with(contract)
            start_pull.reset_mock()
            self.service.restart_image_pull()
            start_pull.assert_called_once_with(contract)

    def test_image_reference_can_omit_digest_when_configured(self):
        contract = self.contract()
        with patch("ModelManager.contract.PULL_WITH_DIGEST", False):
            self.assertEqual(contract.runtime.image.image_ref, "example/runtime:1.0.0")

    def test_create_and_retrieve_contract_persists_yaml_and_status(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=True):
            self.assertTrue(self.service.register_contract(contract))
            self.assertEqual(self.service._read_progress(contract.id)["status"], "ready")
            loaded = self.service.get_contract(contract.id)
            info = self.service.get_contract_info(contract.id)
            contracts_info = self.service.get_contracts_info()

        self.assertEqual(loaded.to_dict(), contract.to_dict())
        self.assertEqual(info["status"], "ready")
        self.assertEqual(contracts_info, [
            {"id": contract.id, "status": "ready"}
        ])

    def test_duplicate_contract_is_rejected_for_same_name_and_version(self):
        first = self.contract()
        changed = self.contract(
            contract={
                "name": "example-contract",
                "description": "Different",
                "version": "1.0.0",
            }
        )
        different_version = self.contract(
            contract={
                "name": "example-contract",
                "description": "Different",
                "version": "2.0.0",
            }
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=True):
            self.service.register_contract(first)
            with self.assertRaises(FileExistsError):
                self.service.register_contract(changed)
            self.assertEqual(self.service.get_contract(first.id).to_dict(), first.to_dict())
            self.service.register_contract(different_version)

        self.assertNotEqual(first.id, different_version.id)

    def test_failed_duplicate_restarts_image_pull(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_image_available", return_value=False), patch.object(
            self.service, "_start_image_pull"
        ) as start_pull:
            self.service.register_contract(contract)
            self.service._track_pull_progress(contract.id, "failed", "pull failed")
            retry = self.contract()
            self.assertFalse(self.service.register_contract(retry))

        self.assertEqual(retry.id, contract.id)
        self.assertEqual(start_pull.call_count, 2)

    def test_failed_retry_preserves_existing_contract_and_results(self):
        contract = self.contract()
        for error in (PermissionError("denied"), RuntimeError("executor unavailable")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
            ):
                self.service._store_contract(contract)
                self.service._track_pull_progress(contract.id, "failed", "pull failed")
                archive = Path(directory, contract.id, "executions", "previous", "output", "summarized_execution.zip")
                archive.parent.mkdir(parents=True)
                archive.write_bytes(b"existing result")

                with patch.object(self.service, "_start_image_pull", side_effect=error):
                    with self.assertRaises(type(error)):
                        self.service.register_contract(self.contract())

                self.assertEqual(self.service.get_contract(contract.id).to_dict(), contract.to_dict())
                self.assertEqual(archive.read_bytes(), b"existing result")

    def test_failed_new_registration_removes_created_contract(self):
        for error in (PermissionError("denied"), RuntimeError("executor unavailable")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
            ), patch.object(self.service, "_image_available", return_value=False), patch.object(
                self.service, "_start_image_pull", side_effect=error
            ):
                contract = self.contract()
                with self.assertRaises(type(error)):
                    self.service.register_contract(contract)

                self.assertFalse(Path(directory, contract.id).exists())

    def test_registration_returns_while_image_pull_worker_is_busy(self):
        release_worker = threading.Event()
        worker_started = threading.Event()

        def occupy_worker():
            worker_started.set()
            release_worker.wait()

        self.service._pull_executor.submit(occupy_worker)
        self.assertTrue(worker_started.wait(timeout=2))
        contract = self.contract()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
            ), patch.object(self.service, "_image_available", return_value=False):
                result = []
                registration = threading.Thread(
                    target=lambda: result.append(self.service.register_contract(contract)),
                    daemon=True,
                )
                registration.start()
                try:
                    registration.join(timeout=3)
                    self.assertFalse(registration.is_alive())
                    self.assertEqual(result, [False])
                    self.assertFalse(self.service._pull_futures[contract.id].running())
                    self.assertFalse(self.service._contract_operation_lock.locked())
                finally:
                    self.service._pull_futures[contract.id].cancel()
                    release_worker.set()
                    registration.join(timeout=3)
        finally:
            release_worker.set()

    def test_available_image_adds_contract_to_scheduler(self):
        contract = self.contract()
        scheduler = MagicMock()
        service = ContractService(
            scheduler_service=scheduler,
            execution_service=MagicMock(),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(service, "_image_available", return_value=True):
            self.assertTrue(service.register_contract(contract))

        scheduler.add_job.assert_called_once()
        scheduled_contract = scheduler.add_job.call_args.args[0]
        self.assertEqual(scheduled_contract.id, contract.id)

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
            self.assertEqual(progress["status"], "installing")
            self.assertEqual(progress["progress"], 62.5)
            self.service._track_pull_progress("id", "failed", "network")
            progress = self.service._read_progress("id")

        self.assertEqual(progress["status"], "installing")
        self.assertEqual(progress["error"], "network")

    def test_pull_retries_with_capped_backoff_and_marks_available(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"MELD_CONTRACT_DIRECTORY": directory, "MELD_PULL_RETRY_DELAY_SECONDS": "2"},
        ):
            self.assertEqual(self.service._pull_retry_delay(1), 2)
            self.assertEqual(self.service._pull_retry_delay(10), 60)
            with patch.object(
                self.service, "_pull_runtime_image", side_effect=[RuntimeError("retry"), None]
            ) as pull, patch.object(self.service, "_pull_retry_delay", return_value=0), patch.object(
                self.service, "_schedule_contract"
            ):
                self.service._pull_image(contract)
            progress = self.service._read_progress(contract.id)

        self.assertEqual(pull.call_count, 2)
        self.assertEqual(progress["status"], "ready")

    def test_successful_image_pull_adds_persisted_contract_to_scheduler(self):
        contract = self.contract()
        scheduler = MagicMock()
        service = ContractService(
            scheduler_service=scheduler,
            execution_service=MagicMock(),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(service, "_pull_runtime_image"):
            service._store_contract(contract)
            service._pull_image(contract)

        scheduler.add_job.assert_called_once()
        scheduled_contract = scheduler.add_job.call_args.args[0]
        self.assertEqual(scheduled_contract.to_dict(), contract.to_dict())

    def test_restore_scheduler_jobs_registers_ready_contracts(self):
        contract = self.contract()
        scheduler = MagicMock()
        service = ContractService(
            scheduler_service=scheduler,
            execution_service=MagicMock(),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ):
            service._store_contract(contract)
            service._track_pull_progress(contract.id, "available")
            service.restore_scheduler_jobs()

        scheduler.add_job.assert_called_once()
        scheduled_contract = scheduler.add_job.call_args.args[0]
        self.assertEqual(scheduled_contract.id, contract.id)

    def test_restore_scheduler_jobs_resumes_interrupted_installation(self):
        contract = self.contract()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_start_image_pull") as start_pull:
            self.service._store_contract(contract)
            self.service._track_pull_progress(contract.id, "pulling")
            self.service.restore_scheduler_jobs()
            start_pull.assert_called_once_with(contract)
        self.scheduler_service.start.assert_called_once()

    def test_delete_contract_waits_for_running_image_pull(self):
        contract = self.contract()
        started = threading.Event()
        release = threading.Event()
        deleted = threading.Event()

        def pull(*args):
            started.set()
            release.wait(timeout=2)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ), patch.object(self.service, "_pull_runtime_image", side_effect=pull), patch(
            "ModelManager.remove_runtime"
        ):
            self.service._store_contract(contract)
            self.service._start_image_pull(contract)
            self.assertTrue(started.wait(timeout=2))
            thread = threading.Thread(target=lambda: (
                self.service.deregister_contract(contract.id), deleted.set()
            ))
            thread.start()
            try:
                self.assertTrue(self.service._pull_cancel_events[contract.id].wait(timeout=2))
                self.assertFalse(deleted.is_set())
                self.assertTrue(self.service._contract_path(contract.id).exists())
            finally:
                release.set()
                thread.join(timeout=2)
            self.assertTrue(deleted.is_set())
            self.assertFalse(self.service._contract_path(contract.id).parent.exists())
            self.scheduler_service.add_job.assert_not_called()

    def test_invalid_progress_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ):
            path = Path(directory) / "id" / "status.json"
            path.parent.mkdir()
            path.write_text("not-json", encoding="utf-8")

            self.assertIsNone(self.service._read_progress("id"))

    def test_delete_contract_cancels_active_executions_before_removing_storage(self):
        contract = self.contract()
        execution = {"execution_id": "execution-id", "status": {"status": "RUNNING"}}
        self.execution_service.get_executions.return_value = [execution]
        calls = []
        self.execution_service.cancel_execution.side_effect = (
            lambda *args, **kwargs: calls.append(("cancel", args, kwargs))
        )
        self.scheduler_service.remove_job.side_effect = (
            lambda value: calls.append(("remove_job", value))
        )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MELD_CONTRACT_DIRECTORY": directory}
        ):
            self.service._store_contract(contract)
            self.service.deregister_contract(contract.id)
            self.assertFalse(Path(directory, contract.id).exists())

        self.assertEqual(calls[0][0], "remove_job")
        self.assertEqual(calls[1][0], "cancel")
        self.execution_service.cancel_execution.assert_called_once_with(
            contract.id, "execution-id", wait=True
        )


if __name__ == "__main__":
    unittest.main()
