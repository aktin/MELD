import sys
import threading
from collections.abc import Callable
from typing import Any

import docker
import requests
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container

from Logger import get_inference_logger, get_meld_logger

from .job_context import JobContext, JobStatus

client = docker.from_env()

logger = get_meld_logger()


def stream_container_logs(container: Container, job_context: JobContext) -> None:
    """
    Streams the stdout and stderr output of a container and processes them in real-time.

    Parameters:
        container (Container): The container instance whose logs are streamed.
        job_context (JobContext): The job context holding the logger for
            processing log messages.

    Raises:
        Exception: If an error occurs while streaming container logs.
    """
    try:
        container_logger = get_inference_logger(job_context.job_id)
        for stdout_chunk, stderr_chunk in container.attach(stdout=True, stderr=True, stream=True, logs=True,
                                                           demux=True):
            if stdout_chunk:
                text = stdout_chunk.decode("utf-8", errors="replace")
                container_logger.info(text.rstrip())

            if stderr_chunk:
                text = stderr_chunk.decode("utf-8", errors="replace")
                container_logger.error(text.rstrip())

            sys.stdout.flush()
            sys.stderr.flush()
    except Exception:
        job_context.logger.exception("Error while streaming container logs")


def pull_image(
    image: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    registry_api_key: str | None = None,
) -> None:
    """
    Pulls a runtime image from the Docker registry.

    Parameters:
    image: str
        The name of the Docker image to pull, including the tag (if applicable).
    progress_callback: Callable[[dict[str, Any]], None] | None
        Optional callback invoked for each decoded Docker pull event.
    registry_api_key: str | None
        Optional one-off registry identity token for this pull request.

    Raises:
    RuntimeError
        If the specified image is not found in the Docker registry or if there is
        an error during the attempt to pull the image.
    """
    try:
        logger.info(f"Pulling runtime image {image}")
        auth_config = (
            {"identitytoken": registry_api_key}
            if registry_api_key
            else None
        )
        if progress_callback is None:
            if auth_config:
                client.images.pull(image, auth_config=auth_config)
            else:
                client.images.pull(image)
        else:
            pull_kwargs: dict[str, Any] = {"stream": True, "decode": True}
            if auth_config:
                pull_kwargs["auth_config"] = auth_config
            for event in client.api.pull(image, **pull_kwargs):
                if "error" in event:
                    error_detail = event.get("errorDetail") or {}
                    message = error_detail.get("message") or event["error"]
                    raise RuntimeError(message)
                progress_callback(event)
        logger.info(f"Pulled runtime image {image}")
    except NotFound as e:
        error = f"Runtime image {image} not found"
        # logger.exception(error)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to pull image {image}"
        # logger.exception(error)
        raise RuntimeError(error) from e


def delete_image(image: str, ) -> None:
    """
    Deletes a runtime image identified by the given image name.

    Args:
        image (str): The name of the runtime image to delete.

    Raises:
        RuntimeError: If the image is not found.
        RuntimeError: If an API error occurs while attempting to delete the image.
    """
    try:
        logger.info(f"Deleting runtime image {image}")
        client.images.remove(image)
        logger.info(f"Deleted runtime image {image}")
    except NotFound as e:
        error = f"Runtime image {image} not found"
        # logger.exception(error)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to delete image {image}"
        # logger.exception(error)
        raise RuntimeError(error) from e


def ensure_image_exists(job_context: JobContext) -> None:
    """
    Ensures that a runtime image specified in the job context exists locally by checking
    with the container client.

    Parameters:
    job_context (JobContext): The context of the job, which contains information
    such as the image tag and logging functionality.

    Raises:
    RuntimeError: If the runtime image specified in the job context is not found.
    """
    image = job_context.image_ref
    try:
        client.images.get(image)

        job_context.logger.info(f"Runtime image {image} already pulled")
        job_context.set_status(JobStatus.IMAGE_PULLED)
    except ImageNotFound as e:
        error = f"Runtime image {image} not found. Make sure to pull the image first."
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error) from e

def image_exists(image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except ImageNotFound:
        return False

def start_container(container: Container, job_context: JobContext, ) -> None:
    """
    Starts a runtime container and updates the job context accordingly.

    Parameters:
        container (Container): The container instance to be started.
        job_context (JobContext): The context of the job, including logging and
            status tracking.

    Raises:
        RuntimeError: If the container fails to start due to an API error, this
            exception is raised with the relevant error message.
    """
    try:
        job_context.logger.info(f"Starting runtime container {container.name}")
        container.start()
        job_context.logger.info("Started runtime container")
        job_context.set_status(JobStatus.RUNNING)
    except APIError as e:
        error = f"Failed to start runtime container"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error) from e


def wait_for_container(container: Container, job_context: JobContext, timeout_seconds: int = 500) -> int:
    """
    Waits for a specified container to complete its execution within a given timeout period.

    Parameters:
        container (Container): The container whose execution is being monitored.
        job_context (JobContext): The context of the job being executed, used for logging and status tracking.
        timeout_seconds (int): The maximum time, in seconds, to wait for the container to complete execution.
                               Defaults to 500 seconds.

    Raises:
        RuntimeError: If the container exits with a non-zero status code, indicating a runtime failure.
        TimeoutError: If the container exceeds the specified timeout period during execution.
    """
    # read container stdout in another thread
    log_thread = threading.Thread(
        target=stream_container_logs,
        args=(container, job_context),
        daemon=True,
    )
    log_thread.start()

    exit_code = -1

    try:
        result = container.wait(timeout=timeout_seconds)
        exit_code = result["StatusCode"]

        if exit_code != 0:
            error = f"Runtime container failed with exit code {exit_code}"
            job_context.logger.error(error)
            job_context.set_status(JobStatus.FAILED)
            # raise RuntimeError(error)
        else:
            job_context.logger.info("Inference has completed successfully")
            job_context.set_status(JobStatus.SUCCESS)
    except requests.exceptions.ReadTimeout:
        error = f"Runtime container timed out after {timeout_seconds} seconds"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.TIMEOUT)
        container.kill()
        raise TimeoutError(error)
    finally:
        # clean up thread
        log_thread.join(timeout=5)

        if log_thread.is_alive():
            logger.warning("Container log streaming thread is still running")

    return exit_code


def stop_container(container: Container, job_context: JobContext):
    """
    Stops the given runtime container.

    Args:
        container (Container): The runtime container to be stopped.
        job_context (JobContext): The context of the job, providing access to
            logging and event recording.

    Raises:
        RuntimeError: If the container could not be stopped due to an API error.
    """
    try:
        job_context.logger.info(f"Stopping runtime container {container.name}")
        container.stop()
        job_context.logger.info(f"Stopped runtime container {container.name}")
    except APIError as e:
        error = f"Failed to stop runtime container"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error) from e


def destroy_container(container: Container, job_context: JobContext) -> None:
    """
    Destroys a specified container and logs the operation.

    Args:
        container (Container): The container object that needs to be destroyed.
        job_context (JobContext): The context of the job, which includes logging
                                  and status tracking capabilities.
    """
    job_context.logger.info(f"Destroying container {container.name}")
    container.remove()
    job_context.logger.info("Container destroyed")
    job_context.set_status(JobStatus.DESTROYED)


def create_container(image: str, job_context: JobContext, ) -> Container:
    """
    Creates a container using the specified runtime image and job context.

    Parameters:
    image (str): The name of the Docker image to be used for creating the container.
    job_context (JobContext): An object representing the context of the job,
    including configuration data, logging, and utilities.

    Returns:
    Container: The created Docker container object.

    Raises:
    RuntimeError: If the specified Docker image is not found, or if there is an
    API-related error during container creation.
    """
    try:
        job_context.logger.info(f"Creating runtime container")
        environment_variables = job_context.contract.runtime.environment_variables or {}
        runtime_container = client.containers.create(image,
                                                     environment=environment_variables,
                                                     name=f"runtime_{job_context.contract.id}_{job_context.job_id}",
                                                     )
        job_context.logger.info(f"Created runtime container {runtime_container.name}")
        job_context.set_status(JobStatus.CREATED)
        return runtime_container
    except NotFound as e:
        error = f"Runtime image {image} not found"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to create container {image}"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error) from e


def get_image_size(image: str, job_context: JobContext,):
    try:
        docker_image = client.images.get(image)
        return docker_image.attrs["Size"]
    except NotFound as e:
        error = f"Runtime image {image} not found"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to get image {image}"
        job_context.logger.error(error)
        job_context.set_status(JobStatus.FAILED)
        raise RuntimeError(error) from e
