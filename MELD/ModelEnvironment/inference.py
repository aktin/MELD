import io
import os
import tarfile

from docker.errors import APIError
from docker.models.containers import Container

import pandas as pd
import yaml
from ModelEnvironment.docker_runtime import ensure_image_exists, create_container, start_container, wait_for_container, \
    stop_container, destroy_container
from ModelEnvironment.job_context import JobContext, JobStatus

def copy_data_to_container(container: Container, input_data: pd.DataFrame, job_context: JobContext) -> None:
    """
    Copies input data and related contract information into a specified runtime container.

    This function handles the transfer of input data and a contract from the runtime job
    context to a container through archiving and file system operations.

    Parameters:
        container (Container): The runtime container instance where data will be copied.
        input_data (pd.DataFrame): The input data to be converted into a CSV format and copied.
        job_context (JobContext): The context of the current job, including paths,
                                   logging mechanisms, and contract details.

    Raises:
        RuntimeError: Raised when the process of copying input data into the container
                      fails due to API errors or other unforeseen exceptions.
    """
    try:
        job_context.logger.info("Copying input data into runtime container")
        input_data.to_csv(os.path.join(job_context.input_data_path, "input.csv"), index=False)
        with open(os.path.join(job_context.input_data_path, "contract.yaml"), "w") as f:
            yaml.dump(job_context.contract, f)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as f:
            f.add(job_context.input_data_path, arcname="")

        container.put_archive("/input", buf.getvalue())
    except APIError as e:
        error = (f"Failed to copy input data into runtime container"
                 f"Folders /input and /output must exist in the container and must be empty")
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e
    except Exception as e:
        error = f"Failed to copy input data into runtime container"
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e


def pack_result_files(output_tar_path: str, archive_bytes: io.BytesIO, job_context: JobContext) -> None:
    """
    Packs result files from an input archive into a gzipped tar file.

    Parameters:
        output_tar_path: str
            The path where the gzipped tar archive will be created.
        archive_bytes: io.BytesIO
            A byte stream containing the input tar archive.
        job_context: JobContext
            The job context object, typically used for logging.

    Raises:
        KeyError: If an invalid tar member is accessed.
        tarfile.TarError: If there's an error processing the tar files.
    """
    job_context.logger.info(f"Packing result files")
    with tarfile.open(output_tar_path, mode="w:gz") as out_tar:
        with tarfile.open(fileobj=archive_bytes, mode="r:*") as in_tar:
            for member in in_tar.getmembers():
                extracted = in_tar.extractfile(member) if member.isfile() else None
                if extracted is not None:
                    data = extracted.read()
                    info = tarfile.TarInfo(name=member.name)
                    info.size = len(data)
                    out_tar.addfile(info, io.BytesIO(data))
                else:
                    out_tar.addfile(member, fileobj=None)


def pack_metadata_and_logs(output_tar_path: str, job_context: JobContext) -> None:
    """
    Packs metadata, logs, and input data into a compressed tarball.

    Parameters:
        output_tar_path (str): The file path where the tarball will be created.
        job_context (JobContext): The context object containing job-related paths
            and logging information.

    Raises:
        TarError: If an error occurs during the tarball creation process.
    """
    job_context.logger.info(f"Packing metadata and logs")
    with tarfile.open(output_tar_path, mode="w:gz") as out_tar:
        out_tar.add(job_context.query_path, arcname="output/query.sql")
        out_tar.add(job_context.contract_path, arcname="output/contract.yaml")
        out_tar.add(job_context.logs_path, arcname="output/logs")
        out_tar.add(job_context.status_path, arcname="output/status")


def get_output_data_from_container(container: Container, job_context: JobContext) -> io.BytesIO:
    """
    Retrieves the output data from a specified container and returns it as a file-like object.

    Parameters:
    container (Container): The container instance from which the output data is retrieved.
    job_context (JobContext): The execution context used during the job, which includes logging
        and operational information.

    Returns:
    io.BytesIO: A file-like object containing the retrieved archive content in memory.
    """
    job_context.logger.info("Copying result from runtime container")
    # Convert Docker's byte-stream generator into a real file-like object.
    archive_stream, _ = container.get_archive("/output/")
    archive_bytes = io.BytesIO(b"".join(archive_stream))
    archive_bytes.seek(0)
    return archive_bytes


def run_inference(input_data: pd.DataFrame, job_context: JobContext) -> None:
    """
    Runs inference on the provided input data using the configured runtime environment.

    Parameters:
    input_data (pd.DataFrame): The data to be processed during the inference.
    job_context (JobContext): The context object containing configuration, logging,
    job metadata, and other task-specific settings.

    Raises:
    Exception: If an error occurs during any part of the inference process.
    """
    job_context.log_event("Running inference", JobStatus.RUNNING)
    image = f"{job_context.contract['inference']['image_tag']}:{job_context.contract['inference']['version']}"
    output_tar_path = os.path.join(job_context.output_data_path, "output.tar.gz")
    runtime_container = None
    try:
        ensure_image_exists(image, job_context)

        runtime_container = create_container(image, job_context, )

        copy_data_to_container(runtime_container, input_data, job_context)

        start_container(runtime_container, job_context, )

        wait_for_container(runtime_container, job_context)

        stop_container(runtime_container, job_context)

        archive_bytes = get_output_data_from_container(runtime_container, job_context)
        pack_result_files(output_tar_path, archive_bytes, job_context)
    except Exception as e:
        job_context.logger.exception(f"An exception occurred during inference: {e}")
    finally:
        pack_metadata_and_logs(output_tar_path, job_context)
        if runtime_container is not None:
            destroy_container(runtime_container, job_context)
