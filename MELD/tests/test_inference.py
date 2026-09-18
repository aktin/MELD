import io
import tempfile
import tarfile
import unittest
import zipfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import test_support

from ModelEnvironment import ExecutionStatus
from ModelEnvironment.inference import InferenceRunner
from ModelManager import Contract


class InferenceRunnerTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract.from_dict(test_support.contract_data())
        self.monitor = MagicMock()
        self.context = MagicMock()
        self.context.contract = self.contract
        self.context.image_ref = "image:tag@digest"
        self.context.logger = MagicMock()
        self.context.execution_id = "execution-id"
        self.context.cancel_requested = False

    def runner(self, directory):
        self.context.input_data_path = str(Path(directory) / "input")
        self.context.output_data_path = str(Path(directory) / "output")
        self.context.logs_path = str(Path(directory) / "logs")
        Path(self.context.input_data_path).mkdir()
        Path(self.context.output_data_path).mkdir()
        Path(self.context.logs_path).mkdir()
        return InferenceRunner(
            pd.DataFrame({"age": [1]}), self.context, self.monitor
        )

    def test_create_interface_folders_writes_input_and_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            container = MagicMock()
            runner.runtime_container = container

            runner.create_interface_folders()

        archive = container.put_archive.call_args.args[1]
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
            members = {member.name: member.mode for member in tar.getmembers()}
        self.assertEqual(set(members), {"input", "output"})
        self.assertEqual(members["input"], 0o555)
        self.assertEqual(members["output"], 0o755)

    def test_copy_data_to_container_writes_contract_and_input_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            container = MagicMock()
            runner.runtime_container = container

            runner.copy_data_to_container()

            self.assertIn("age", Path(runner.job_context.input_data_path, "input.csv").read_text())
            self.assertTrue(Path(runner.job_context.input_data_path, "contract.yaml").exists())
            container.put_archive.assert_called_once()
            self.monitor.update_metric_value.assert_any_call(ANY, ANY)

    def test_extract_result_file_reads_expected_member_and_updates_size(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            data = b"prediction\n0.5\n"
            info = tarfile.TarInfo("output/output.csv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            self.assertEqual(runner.extract_result_file(archive), b"prediction\n0.5\n")

        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w"):
                pass
            with self.assertRaises(FileNotFoundError):
                runner.extract_result_file(archive)

    def test_pack_archive_without_result_keeps_metadata_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            Path(runner.job_context.input_data_path, "input.csv").write_text("age\n1\n")
            Path(runner.job_context.logs_path, "runtime.log").write_text("failed")

            runner.pack_archive(None)

            with zipfile.ZipFile(runner.output_zip_path) as archive:
                self.assertNotIn("output/output.csv", archive.namelist())
                self.assertIn("input/input.csv", archive.namelist())
                self.assertIn("runtime.log", archive.namelist())

    def test_result_conversion_and_metrics_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            result = runner._csv_as_bytes_to_df(b"prediction\n0.5\n")
            runner.collect_result_metrics(result)

        self.assertEqual(result.iloc[0, 0], 0.5)
        self.assertEqual(self.monitor.update_metric_value.call_count, 4)

    def test_run_successfully_orchestrates_runtime_and_processes_result(self):
        output_df = pd.DataFrame({"prediction": [0.5]})
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            runner.monitor.stop_inference_time.return_value = timedelta(seconds=1)
            with patch("ModelEnvironment.inference.get_image_size", return_value=10), patch(
                "ModelEnvironment.inference.create_container", return_value=MagicMock()
            ) as create, patch("ModelEnvironment.inference.start_container") as start, patch(
                "ModelEnvironment.inference.wait_for_container", return_value=0
            ), patch("ModelEnvironment.inference.stop_container") as stop, patch.object(
                runner, "create_interface_folders"
            ), patch.object(runner, "copy_data_to_container"), patch.object(
                runner, "get_output_data_from_container", return_value=io.BytesIO()
            ), patch.object(runner, "extract_result_file", return_value=b"prediction\n0.5\n"), patch.object(
                runner, "_csv_as_bytes_to_df", return_value=output_df
            ), patch.object(runner, "pack_archive") as pack:
                result_path = runner.run()

        self.assertTrue(result_path.endswith("summarized_execution.zip"))
        create.assert_called_once_with("image:tag@digest", self.context)
        start.assert_called_once()
        stop.assert_called_once()
        pack.assert_called_once()
        self.context.set_status.assert_any_call(ExecutionStatus.SUCCESS)
        self.context.set_cancel_callback.assert_any_call(None)

    def test_run_stops_before_starting_container_when_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            with patch.object(runner, "_cancel_requested", side_effect=[False, True, True]), patch(
                "ModelEnvironment.inference.get_image_size", return_value=10
            ), patch("ModelEnvironment.inference.create_container", return_value=MagicMock()), patch(
                "ModelEnvironment.inference.start_container"
            ) as start, patch.object(runner, "create_interface_folders"), patch.object(
                runner, "copy_data_to_container"
            ), patch.object(runner, "pack_archive"):
                runner.run()

        start.assert_not_called()
        self.context.set_status.assert_called_with(ExecutionStatus.CANCELED)


if __name__ == "__main__":
    unittest.main()
