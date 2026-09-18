import threading
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_support  # noqa: F401
from ModelEnvironment import ExecutionStatus
from ModelEnvironment.execution_service import ExecutionService, run_inference
from werkzeug.exceptions import BadRequest


class ExecutionServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = ExecutionService()

    def tearDown(self):
        self.service._execution_executor.shutdown(wait=True, cancel_futures=True)

    @patch("ModelEnvironment.execution_service.ExecutionContext.create")
    @patch("ModelEnvironment.execution_service.Contract.from_yaml")
    @patch("ModelEnvironment.execution_service.run_inference")
    async def test_create_execution_schedules_inference_in_background(
        self,
        run_inference,
        from_yaml,
        create_context,
    ):
        context = MagicMock(execution_id="execution-id")
        create_context.return_value = context
        contract = MagicMock()
        from_yaml.return_value = contract
        started = threading.Event()
        release = threading.Event()

        def blocking_inference(*args):
            started.set()
            release.wait(timeout=2)

        run_inference.side_effect = blocking_inference

        with patch.object(self.service, "_check_contract_exists"):
            execution_id = await self.service.create_execution("contract-id")

        self.assertEqual(execution_id, "execution-id")
        self.assertTrue(started.wait(timeout=2))
        run_inference.assert_called_once_with(contract, context)
        release.set()

    def test_cancel_execution_cancels_matching_future_and_context(self):
        context = MagicMock(execution_id="execution-id")
        future = Future()
        self.service.current_inference = future
        self.service.current_execution_id = "execution-id"
        self.service.current_context = context

        with patch.object(self.service, "_check_contract_exists"), patch.object(
            self.service, "_check_execution_exists"
        ):
            self.service.cancel_execution("contract-id", "execution-id")

        context.request_cancel.assert_called_once_with()
        context.set_status.assert_called_once_with(ExecutionStatus.CANCELED)
        self.assertTrue(future.cancelled())

    @patch("ModelEnvironment.execution_service.ExecutionContext.create")
    def test_cancel_execution_does_not_cancel_different_active_execution(
        self,
        create_context,
    ):
        self.service = ExecutionService()
        active_context = MagicMock(execution_id="active-id")
        requested_context = MagicMock(execution_id="requested-id")
        self.service.current_context = active_context
        self.service.current_execution_id = "active-id"
        self.service.current_inference = Future()
        create_context.return_value = requested_context

        with patch.object(self.service, "_check_contract_exists"), patch.object(
            self.service, "_check_execution_exists"
        ):
            self.service.cancel_execution("contract-id", "requested-id")

        active_context.request_cancel.assert_not_called()
        self.assertFalse(self.service.current_inference.cancelled())
        requested_context.set_status.assert_called_once_with(ExecutionStatus.CANCELED)

    def test_cancel_execution_cancels_queued_execution(self):
        context = MagicMock(status=ExecutionStatus.PENDING)
        future = Future()
        self.service._execution_tasks["queued-id"] = (future, context)

        with patch.object(self.service, "_check_contract_exists"), patch.object(
            self.service, "_check_execution_exists"
        ):
            self.service.cancel_execution("contract-id", "queued-id")

        context.request_cancel.assert_called_once_with()
        context.set_status.assert_called_once_with(ExecutionStatus.CANCELED)
        self.assertTrue(future.cancelled())
        self.assertNotIn("queued-id", self.service._execution_tasks)

    def test_cancel_execution_rejects_completed_execution(self):
        for status in (
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELED,
            ExecutionStatus.TIMEOUT,
        ):
            with self.subTest(status=status):
                context = MagicMock(status={"status": status.value})
                with patch(
                    "ModelEnvironment.execution_service.ExecutionContext.create",
                    return_value=context,
                ), patch.object(self.service, "_check_contract_exists"), patch.object(
                    self.service, "_check_execution_exists"
                ):
                    with self.assertRaises(BadRequest):
                        self.service.cancel_execution("contract-id", "execution-id")

                context.request_cancel.assert_not_called()
                context.set_status.assert_not_called()

    def test_get_executions_reports_execution_directory_id_and_status(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ModelEnvironment.execution_service.CONTRACTS_DIR", directory
        ):
            execution_folder = Path(directory) / "contract-id" / "executions" / "execution-id"
            execution_folder.joinpath("status").mkdir(parents=True)
            Path(directory, "contract-id", "contract.yaml").write_text("contract", encoding="utf-8")
            execution_folder.joinpath("status", "status.json").write_text(
                '{"status": "SUCCESS"}', encoding="utf-8"
            )

            executions = list(self.service.get_executions("contract-id"))

        self.assertEqual(
            executions,
            [{"status": {"status": "SUCCESS"}, "execution_id": "execution-id"}],
        )

    def test_get_executions_sorts_by_execution_id_descending(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ModelEnvironment.execution_service.CONTRACTS_DIR", directory
        ), patch(
            "ModelEnvironment.execution_service.glob.glob",
            return_value=[
                f"{directory}/contract-id/executions/execution-001/status/status.json",
                f"{directory}/contract-id/executions/execution-003/status/status.json",
                f"{directory}/contract-id/executions/execution-002/status/status.json",
            ],
        ):
            for execution_id in ("execution-001", "execution-002", "execution-003"):
                status_path = (
                    Path(directory)
                    / "contract-id"
                    / "executions"
                    / execution_id
                    / "status"
                    / "status.json"
                )
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    '{"status": "SUCCESS"}',
                    encoding="utf-8",
                )
            Path(directory, "contract-id", "contract.yaml").write_text(
                "contract",
                encoding="utf-8",
            )

            executions = list(self.service.get_executions("contract-id"))

        self.assertEqual(
            [execution["execution_id"] for execution in executions],
            ["execution-003", "execution-002", "execution-001"],
        )

    def test_get_result_archive_returns_independent_bytes_stream(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ModelEnvironment.execution_service.CONTRACTS_DIR", directory
        ):
            output = Path(directory) / "contract-id" / "executions" / "execution-id" / "output"
            output.mkdir(parents=True)
            archive = output / "summarized_execution.zip"
            archive.write_bytes(b"archive")
            Path(directory, "contract-id", "contract.yaml").write_text("contract", encoding="utf-8")

            result = self.service.get_result_archive("contract-id", "execution-id")

        self.assertEqual(result.read(), b"archive")

    def test_read_log_uses_first_matching_log_file(self):
        context = MagicMock(logs_path="/unused")
        with tempfile.TemporaryDirectory() as directory, patch(
            "ModelEnvironment.execution_service.CONTRACTS_DIR", directory
        ), patch.object(
            self.service, "_get_execution_folder", return_value=str(Path(directory) / "execution")
        ), patch.object(
            self.service, "_check_contract_exists"
        ), patch.object(
            self.service, "_check_execution_exists"
        ), patch(
            "ModelEnvironment.execution_service.ExecutionContext.create", return_value=context
        ):
            logs = Path(directory) / "logs"
            logs.mkdir()
            (logs / "execution.log").write_text("completed\n", encoding="utf-8")
            context.logs_path = str(logs)

            self.assertEqual(self.service.read_log("contract-id", "execution-id"), "completed\n")

    def test_stream_log_yields_existing_lines(self):
        context = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.service, "_check_contract_exists"
        ), patch.object(
            self.service, "_check_execution_exists"
        ), patch(
            "ModelEnvironment.execution_service.ExecutionContext.create", return_value=context
        ):
            log_path = Path(directory) / "execution.log"
            log_path.write_text("first\nsecond\n", encoding="utf-8")
            context.logs_path = directory

            stream = self.service.stream_log("contract-id", "execution-id")
            self.assertEqual(next(stream), "first")
            self.assertEqual(next(stream), "second")

    def test_clear_current_inference_only_clears_matching_future(self):
        future = Future()
        self.service.current_inference = future
        self.service.current_execution_id = "execution-id"
        self.service.current_context = MagicMock()

        self.service._clear_current_inference(Future())
        self.assertIs(self.service.current_inference, future)
        self.service._clear_current_inference(future)

        self.assertIsNone(self.service.current_inference)
        self.assertIsNone(self.service.current_execution_id)
        self.assertIsNone(self.service.current_context)


if __name__ == "__main__":
    unittest.main()
