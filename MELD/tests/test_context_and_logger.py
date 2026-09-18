import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_support  # noqa: F401

from Logger.logger import NoTracebackFormatter, _setup_logger, get_job_logger
from ModelEnvironment.execution_context import (
    ContextProvider,
    ExecutionContext,
    ExecutionStatus,
)


class ExecutionContextTest(unittest.TestCase):
    def setUp(self):
        self.contract = MagicMock()
        self.contract.id = "contract-id"
        self.contract.runtime.image.construct_image_ref.return_value = "image:tag@digest"

    def test_new_context_creates_folders_and_pending_status(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("ModelEnvironment.execution_context.ROOT_DIR", root), patch(
                "ModelEnvironment.execution_context.CONTRACTS_DIR", "contracts"
            ):
                context = ExecutionContext(self.contract)

            self.assertTrue(Path(context.input_data_path).is_dir())
            self.assertTrue(Path(context.output_data_path).is_dir())
            self.assertTrue(Path(context.status_path).is_dir())
            self.assertTrue(Path(context.logs_path).is_dir())
            self.assertEqual(context.image_ref, "image:tag@digest")
            self.assertEqual(context.status, ExecutionStatus.PENDING)

    def test_new_context_execution_id_includes_microseconds(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("ModelEnvironment.execution_context.ROOT_DIR", root), patch(
                "ModelEnvironment.execution_context.CONTRACTS_DIR", "contracts"
            ):
                context = ExecutionContext(self.contract)

        contract_id, timestamp = context.execution_id.rsplit("_", 1)
        self.assertEqual(contract_id, self.contract.id)
        self.assertRegex(timestamp, re.compile(r"^\d{20}$"))

    def test_new_context_writes_pending_status_and_reloads_persisted_status(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("ModelEnvironment.execution_context.ROOT_DIR", root), patch(
                "ModelEnvironment.execution_context.CONTRACTS_DIR", "contracts"
            ):
                context = ExecutionContext(self.contract)
                context.set_status(ExecutionStatus.RUNNING)
                reloaded = ExecutionContext(self.contract, execution_id=context.execution_id)

            status = Path(context.status_path) / "status.json"
            self.assertIn("RUNNING", status.read_text(encoding="utf-8"))
            self.assertEqual(reloaded.status["status"], "RUNNING")

    def test_cancellation_callback_runs_when_registered_before_or_after_request(self):
        with tempfile.TemporaryDirectory() as root, patch(
            "ModelEnvironment.execution_context.ROOT_DIR", root
        ), patch("ModelEnvironment.execution_context.CONTRACTS_DIR", "contracts"):
            context = ExecutionContext(self.contract)
            callback = MagicMock()
            context.request_cancel()
            context.set_cancel_callback(callback)
            context.request_cancel()

        self.assertTrue(context.cancel_requested)
        self.assertEqual(callback.call_count, 2)

    def test_context_provider_returns_context(self):
        context = MagicMock()

        self.assertIs(ContextProvider(context).get(), context)


class LoggerTest(unittest.TestCase):
    def test_formatter_suppresses_traceback_without_mutating_record(self):
        formatter = NoTracebackFormatter("%(message)s")
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failure", (), None)
        try:
            raise ValueError("hidden")
        except ValueError:
            record.exc_info = __import__("sys").exc_info()

        formatted = formatter.format(record)

        self.assertEqual(formatted, "failure")
        self.assertIsNotNone(record.exc_info)

    def test_logger_setup_is_idempotent_and_writes_job_log(self):
        with tempfile.TemporaryDirectory() as directory:
            logger_name = "meld.test.logger"
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            first = _setup_logger(logger_name, log_dir=directory, console=False)
            second = _setup_logger(logger_name, log_dir=directory, console=False)
            first.info("hello")
            for handler in first.handlers:
                handler.flush()

            log_path = Path(directory) / f"{logger_name}.log"
            self.assertIs(first, second)
            self.assertEqual(len(first.handlers), 1)
            self.assertIn("hello", log_path.read_text(encoding="utf-8"))
            for handler in first.handlers[:]:
                handler.close()
                first.removeHandler(handler)

    def test_job_logger_uses_job_specific_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = get_job_logger("job-id", directory)
            logger.info("job message")
            for handler in logger.handlers:
                handler.flush()

            self.assertTrue((Path(directory) / "meld.jobjob-id.log").exists())
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
