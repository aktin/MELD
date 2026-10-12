import csv
import io
import tarfile
import zipfile
from io import StringIO
from pathlib import Path
from typing import Optional

from docker.errors import APIError
from docker.models.containers import Container

import pandas as pd
import yaml

from ExecutionMonitor import ExecutionMonitor, Metrics
from utils import get_unexpected_features, validate_feature_datatypes

from .docker_runtime import (
    create_container,
    get_image_size,
    start_container,
    stop_container,
    wait_for_container,
)
from .execution_context import ExecutionContext, ExecutionStatus


class InferenceRunner:
    def __init__(self, input_data: pd.DataFrame, job_context: ExecutionContext, monitor: ExecutionMonitor):
        self.input_data = input_data
        self.job_context: ExecutionContext = job_context
        self.monitor: ExecutionMonitor = monitor
        self.image = job_context.image_ref
        self.output_zip_path = Path(job_context.output_data_path) / "summarized_execution.zip"
        self.runtime_container: Optional[Container] = None

    def run(self) -> Path | None:
        """
        Runs inference on the provided input data using the configured runtime environment.
        """
        archive_bytes = None

        self.job_context.logger.info("Running inference")
        if self._cancel_requested():
            return self.output_zip_path

        self.job_context.set_status(ExecutionStatus.RUNNING)
        self.monitor.update_metric_value(Metrics.INFERENCE_INPUT_ROW_COUNT, len(self.input_data.index))
        try:
            self.monitor.update_metric_value(Metrics.DOCKER_IMAGE_SIZE, get_image_size(self.image, self.job_context))

            self.runtime_container = create_container(self.image, self.job_context)

            self.create_interface_folders()
            self.copy_data_to_container()

            if self._cancel_requested():
                return self.output_zip_path

            self.monitor.start_inference_time()
            start_container(self.runtime_container, self.job_context)
            self.job_context.set_cancel_callback(self._stop_runtime_container)

            exit_code = wait_for_container(self.runtime_container, self.job_context)

            if self._cancel_requested():
                return self.output_zip_path

            stop_container(self.runtime_container, self.job_context)
            timespan = self.monitor.stop_inference_time()

            self.job_context.logger.info(f"Inference finished in {timespan.total_seconds():.3f} seconds")
            self.monitor.update_metric_value(Metrics.RUNTIME_EXIT_CODE, exit_code)

            if exit_code != 0:
                error = f"Inference failed with exit code {exit_code}"
                self.job_context.logger.error(error)
                raise Exception(error)
            else:
                self.job_context.logger.info("Inference has completed successfully")
                self.job_context.set_status(ExecutionStatus.SUCCESS)

            archived_output_data = self.get_output_data_from_container()
            archive_bytes = self.extract_result_file(archived_output_data)
            self._warn_if_invalid_csv(archive_bytes)
            output_df = self._csv_as_bytes_to_df(archive_bytes)
            self.verify_result_data(output_df)
            self.collect_result_metrics(output_df)

        except Exception as e:
            self.job_context.logger.exception(f"An exception occurred during inference: {e}")
            raise
        finally:
            try:
                self.pack_archive(archive_bytes)
            finally:
                self.job_context.set_cancel_callback(None)
                if self._cancel_requested():
                    self.job_context.set_status(ExecutionStatus.CANCELED)
                self.job_context.logger.info("Inference execution completed")

        return self.output_zip_path

    def _cancel_requested(self) -> bool:
        return self.job_context.cancel_requested

    def _stop_runtime_container(self) -> None:
        if self.runtime_container is None:
            return

        try:
            stop_container(self.runtime_container, self.job_context)
        except Exception:
            self.job_context.logger.exception("Failed to stop canceled runtime container")


    def pack_archive(self, archive_bytes: bytes | None):
        self.monitor.start_archive_packing_time()
        try:
            self.pack_result_file(archive_bytes)
        finally:
            self.pack_metadata_and_logs()
            self.monitor.stop_archive_packing_time()

    def create_interface_folders(self) -> None:
        """
        Creates and configures interface folders `/input` and `/output` in the provided container.
        """
        try:
            assert self.runtime_container is not None
            buf = io.BytesIO()

            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo("input")
                info.type = tarfile.DIRTYPE
                info.mode = 0o555
                tar.addfile(info)

                info = tarfile.TarInfo("output")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)

            buf.seek(0)
            self.runtime_container.put_archive("/", buf.read())
        except Exception as e:
            error = "Failed to create interface folders"
            raise RuntimeError(error) from e

    def copy_data_to_container(self) -> None:
        """
        Copies input data and related contract information into a specified runtime container.
        """
        try:
            assert self.runtime_container is not None

            self.job_context.logger.info("Copying input data into runtime container")
            input_csv = self.input_data.to_csv(index=False)
            self.monitor.update_metric_value(Metrics.INFERENCE_INPUT_SIZE, len(input_csv.encode("utf-8")))
            with (Path(self.job_context.input_data_path) / "input.csv").open("w") as f:
                f.write(input_csv)
            with (Path(self.job_context.input_data_path) / "contract.yaml").open("w") as f:
                yaml.dump(self.job_context.contract.to_dict(), f)

            buf = io.BytesIO()
            # tar because docker.models.containers.Container.put_archive expects a tar archive as stream or bytes
            with tarfile.open(fileobj=buf, mode="w:gz") as f:
                f.add(self.job_context.input_data_path, arcname="")
            self.monitor.start_input_data_copy_time()
            self.runtime_container.put_archive("/input", buf.getvalue())
            self.monitor.stop_input_data_copy_time()
        except APIError as e:
            error = "Failed to copy input data into runtime container"
            self.job_context.logger.error(error)
            self.job_context.set_status(ExecutionStatus.FAILED)
            raise RuntimeError(error) from e
        except Exception as e:
            error = "Failed to copy input data into runtime container"
            self.job_context.logger.error(error)
            self.job_context.set_status(ExecutionStatus.FAILED)
            raise RuntimeError(error) from e

    def extract_result_file(self, archive_bytes: io.BytesIO) -> bytes:
        self.job_context.logger.info("Extracting result file")
        archive_bytes.seek(0)
        extracted = None
        with tarfile.open(fileobj=archive_bytes, mode="r:*") as in_tar:
            for member in in_tar.getmembers():
                if not member.isfile() or not member.name == "output/output.csv":
                    continue

                extracted = in_tar.extractfile(member)

        if extracted is None:
            raise FileNotFoundError("Result file output.csv not found in inference runtime output folder")

        self.job_context.logger.info("Result file output.csv extracted successfully")

        extracted.seek(0)
        result_bytes = extracted.read()
        self.monitor.update_metric_value(Metrics.INFERENCE_RESULT_SIZE, len(result_bytes))
        return result_bytes

    def _warn_if_invalid_csv(self, csv_bytes: bytes) -> None:
        try:
            csv.Sniffer().sniff(csv_bytes.decode("utf-8"))
        except (csv.Error, UnicodeDecodeError):
            self.job_context.logger.warning("Likely invalid CSV format")

    def pack_result_file(self, extracted_file: bytes) -> None:
        """
        Packs result files from an input archive into a gzipped tar file.
        """
        self.job_context.logger.info("Packing result files")

        with zipfile.ZipFile(self.output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as out_zip:
            if extracted_file is not None:
                out_zip.writestr("output/output.csv", extracted_file)

    def pack_metadata_and_logs(self) -> None:
        """
        Packs metadata, logs, and input data into a compressed tarball.
        """
        self.job_context.logger.info("Packing metadata and logs")
        with zipfile.ZipFile(self.output_zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as out_zip:
            input_data_path = Path(self.job_context.input_data_path)
            for file_path in input_data_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(input_data_path).as_posix()
                    out_zip.write(file_path, f"input/{relative_path}")

            logs_path = Path(self.job_context.logs_path)
            if logs_path.is_dir():
                for file_path in logs_path.rglob("*"):
                    if file_path.is_file():
                        out_zip.write(
                            file_path,
                            file_path.relative_to(logs_path).as_posix(),
                        )
            else:
                out_zip.write(logs_path, "")

    def get_output_data_from_container(self) -> io.BytesIO:
        """
        Retrieves the output data from a specified container and returns it as a file-like object.
        """
        assert self.runtime_container is not None
        self.job_context.logger.info("Copying result from runtime container")
        # Convert Docker's byte-stream generator into a real file-like object.
        self.monitor.start_output_data_copy_time()
        archive_stream, _ = self.runtime_container.get_archive("/output/")
        self.monitor.stop_output_data_copy_time()
        archive_bytes = io.BytesIO(b"".join(archive_stream))
        archive_bytes.seek(0)
        return archive_bytes


    def _csv_as_bytes_to_df(self, csv_bytes: bytes) -> pd.DataFrame:
        # Convert bytes to string and then to DataFrame
        data_str = csv_bytes.decode("utf-8")
        output_df = pd.read_csv(StringIO(data_str))
        return output_df

    def verify_result_data(self, output_df: pd.DataFrame) -> None:
        self.job_context.logger.info("Verifying result data")

        labels = self.job_context.contract.output_schema.labels

        try:
            validate_feature_datatypes(output_df, labels)
        except ValueError as e:
            self.job_context.logger.warning(e)

        unexpected_labels = get_unexpected_features(output_df, labels)
        if unexpected_labels:
            self.job_context.logger.warning(f"Unexpected labels found: {', '.join(unexpected_labels)}")

        input_rows = len(self.input_data.index)
        output_rows = len(output_df.index)

        if input_rows != output_rows:
            self.job_context.logger.warning(
                "Result data does not match input data: "
                f"Expected {input_rows} rows, got {output_rows} rows"
            )

    def collect_result_metrics(self, output_df: pd.DataFrame) -> None:
        self.monitor.update_metric_value(Metrics.ACTUAL_FEATURE_COUNT, len(self.input_data.columns))
        self.monitor.update_metric_value(Metrics.ACTUAL_PREDICTOR_COUNT, len(output_df.columns))
        self.monitor.update_metric_value(Metrics.INFERENCE_INPUT_ROW_COUNT, len(self.input_data.index))
        self.monitor.update_metric_value(Metrics.INFERENCE_RESULT_ROW_COUNT, len(output_df.index))


def run_inference(input_data: pd.DataFrame, job_context: ExecutionContext, monitor: ExecutionMonitor) -> str | None:
    return InferenceRunner(input_data, job_context, monitor).run()
