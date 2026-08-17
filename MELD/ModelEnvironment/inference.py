import csv
import io
import os
import tarfile
import zipfile
from io import StringIO
from typing import Optional


import pandas as pd
import yaml
from docker.errors import APIError
from docker.models.containers import Container

from ExecutionMonitor.metrics import Metrics
from ExecutionMonitor.monitor import ExecutionMonitor
from ModelEnvironment.docker_runtime import create_container, start_container, wait_for_container, \
    stop_container, get_image_size
from ModelEnvironment.job_context import JobContext, JobStatus
from utils import validate_feature_datatypes, get_unexpected_features


class InferenceRunner:
    def __init__(self, input_data: pd.DataFrame, job_context: JobContext, monitor: ExecutionMonitor):
        self.input_data = input_data
        self.job_context: JobContext = job_context
        self.monitor: ExecutionMonitor = monitor
        self.image = job_context.image_ref
        self.output_zip_path = os.path.join(job_context.output_data_path, "summarized_execution.zip")
        self.runtime_container: Optional[Container] = None

    def run(self) -> str | None:
        """
        Runs inference on the provided input data using the configured runtime environment.
        """
        self.job_context.log_event("Running inference", JobStatus.RUNNING)
        self.monitor.update_metric_value(Metrics.INFERENCE_INPUT_ROW_COUNT, len(self.input_data.index))
        try:
            self.monitor.update_metric_value(Metrics.DOCKER_IMAGE_SIZE, get_image_size(self.image, self.job_context))

            self.runtime_container = create_container(self.image, self.job_context)

            self.create_interface_folders()
            self.copy_data_to_container()

            self.monitor.start_inference_time()
            start_container(self.runtime_container, self.job_context)

            exit_code = wait_for_container(self.runtime_container, self.job_context)

            stop_container(self.runtime_container, self.job_context)
            timespan = self.monitor.stop_inference_time()

            self.job_context.logger.info(f"Inference finished in {timespan.total_seconds():.3f} seconds")
            self.monitor.update_metric_value(Metrics.RUNTIME_EXIT_CODE, exit_code)

            if exit_code != 0:
                error = f"Inference failed with exit code {exit_code}"
                self.job_context.log_event(error, JobStatus.FAILED)
                raise Exception(error)
            else:
                self.job_context.log_event("Inference has completed successfully", JobStatus.SUCCESS)

            archived_output_data = self.get_output_data_from_container()
            archive_bytes = self.extract_result_file(archived_output_data)
            self._warn_if_invalid_csv(archive_bytes)
            output_df = self._csv_as_bytes_to_df(archive_bytes)
            self.verify_result_data(output_df)
            self.collect_result_metrics(output_df)

        except Exception as e:
            self.job_context.logger.exception(f"An exception occurred during inference: {e}")
        finally:
            self.pack_archive(archive_bytes)

            return self.output_zip_path


    def pack_archive(self, archive_bytes: bytes):
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
            with open(os.path.join(self.job_context.input_data_path, "input.csv"), "w") as f:
                f.write(input_csv)
            with open(os.path.join(self.job_context.input_data_path, "contract.yaml"), "w") as f:
                yaml.dump(self.job_context.contract, f)

            buf = io.BytesIO()
            # tar because docker.models.containers.Container.put_archive expects a tar archive as stream or bytes
            with tarfile.open(fileobj=buf, mode="w:gz") as f:
                f.add(self.job_context.input_data_path, arcname="")
            self.monitor.start_input_data_copy_time()
            self.runtime_container.put_archive("/input", buf.getvalue())
            self.monitor.stop_input_data_copy_time()
        except APIError as e:
            error = "Failed to copy input data into runtime container"
            self.job_context.log_event(error, JobStatus.FAILED, error=str(e))
            raise RuntimeError(error) from e
        except Exception as e:
            error = "Failed to copy input data into runtime container"
            self.job_context.log_event(error, JobStatus.FAILED, error=str(e))
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
            out_zip.writestr("output/output.csv", extracted_file)

    def pack_metadata_and_logs(self) -> None:
        """
        Packs metadata, logs, and input data into a compressed tarball.
        """
        self.job_context.logger.info("Packing metadata and logs")
        with zipfile.ZipFile(self.output_zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as out_zip:
            for root, _, files in os.walk(self.job_context.input_data_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.join("input", os.path.relpath(file_path, self.job_context.input_data_path))
                    out_zip.write(file_path, arcname)

            for path, arcname in (
                (self.job_context.logs_path, "logs"),
                # (self.job_context.status_path, "status"),
            ):
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            out_zip.write(
                                file_path,
                                os.path.join(arcname, os.path.relpath(file_path, path)),
                            )
                else:
                    out_zip.write(path, arcname)

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

        predictors = self.job_context.contract["output_schema"]["predictor"]

        try:
            validate_feature_datatypes(output_df, predictors)
        except ValueError as e:
            self.job_context.logger.warning(e)

        unexpected_predictors = get_unexpected_features(output_df, predictors)
        if unexpected_predictors:
            self.job_context.logger.warning(f"Unexpected predictors found: {', '.join(unexpected_predictors)}")

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


def run_inference(input_data: pd.DataFrame, job_context: JobContext, monitor: ExecutionMonitor) -> str | None:
    return InferenceRunner(input_data, job_context, monitor).run()
