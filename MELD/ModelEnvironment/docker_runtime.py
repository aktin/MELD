import sys
import threading

import docker
from docker.errors import APIError, NotFound, ImageNotFound
from docker.models.containers import Container

import requests
from ModelEnvironment.job_context import JobStatus, JobContext
from Logger import setup_logger

client = docker.from_env()

logger = setup_logger("meld")


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
        container_logger = setup_logger(f"meld.job{job_context.job_id}.inference", logs_path=job_context.logs_path)
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
        logger.exception("Error while streaming container logs")


def pull_image(image: str, ) -> None:
    """
    Pulls a runtime image from the Docker registry.

    Parameters:
    image: str
        The name of the Docker image to pull, including the tag (if applicable).

    Raises:
    RuntimeError
        If the specified image is not found in the Docker registry or if there is
        an error during the attempt to pull the image.
    """
    try:
        logger.info(f"Pulling runtime image {image}")
        client.images.pull(image)
        logger.info(f"Pulled runtime image {image}")
    except NotFound as e:
        error = f"Runtime image {image} not found"
        logger.exception(error)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to pull image {image}"
        logger.exception(error)
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
        logger.exception(error)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to delete image {image}"
        logger.exception(error)
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
    image = job_context.image_tag
    try:
        client.images.get(image)

        job_context.log_event(
            f"Runtime image {image} already pulled",
            JobStatus.IMAGE_PULLED,
            image=image,
        )
    except ImageNotFound as e:
        error = f"Runtime image {image} not found"
        job_context.log_event(error, JobStatus.FAILED, image=image)
        raise RuntimeError(error) from e


def start_container(container: Container, job_context: JobContext, ) -> None:
    """
    Starts a runtime container and updates the job context accordingly.

    Parameters:
        container (Container): The container instance to be started.
        job_context (JobContext): The context of the job, including logging and
            event handling.

    Raises:
        RuntimeError: If the container fails to start due to an API error, this
            exception is raised with the relevant error message.
    """
    try:
        job_context.logger.info(f"Starting runtime container {container.id}")
        container.start()
        job_context.log_event("Started runtime container", JobStatus.RUNNING)
    except APIError as e:
        error = f"Failed to start runtime container"
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e


def wait_for_container(container: Container, job_context: JobContext, timeout_seconds: int = 500):
    """
    Waits for a specified container to complete its execution within a given timeout period.

    Parameters:
        container (Container): The container whose execution is being monitored.
        job_context (JobContext): The context of the job being executed, used for logging and event handling.
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

    try:
        result = container.wait(timeout=timeout_seconds)
        exit_code = result["StatusCode"]

        if exit_code != 0:
            error = f"Runtime container failed with exit code {exit_code}"
            job_context.log_event("Inference failed", JobStatus.FAILED,
                                  error=error)
            raise RuntimeError(error)
        job_context.log_event("Inference has completed successfully", JobStatus.SUCCESS)
    except requests.exceptions.ReadTimeout:
        error = f"Runtime container timed out after {timeout_seconds} seconds"
        job_context.log_event(error, JobStatus.TIMEOUT, timeout=timeout_seconds)
        container.kill()
        raise TimeoutError(error)
    finally:
        # clean up thread
        log_thread.join(timeout=5)

        if log_thread.is_alive():
            logger.warning("Container log streaming thread is still running")


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
        job_context.logger.info(f"Stopping runtime container {container.id}")
        container.stop()
        job_context.logger.info(f"Stopped runtime container {container.id}")
    except APIError as e:
        error = f"Failed to stop runtime container"
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e


def destroy_container(container: Container, job_context: JobContext) -> None:
    """
    Destroys a specified container and logs the operation.

    Args:
        container (Container): The container object that needs to be destroyed.
        job_context (JobContext): The context of the job, which includes logging
                                  and event tracking capabilities.
    """
    job_context.logger.info(f"Destroying container {container.id}")
    container.remove()
    job_context.log_event("Container destroyed", JobStatus.DESTROYED)


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
        environment_variables = job_context.contract["runtime"]["environment_variables"]
        runtime_container = client.containers.create(image,
                                                     environment=environment_variables, )
        job_context.log_event(f"Created runtime container {runtime_container.id}",
                              JobStatus.CREATED,
                              image=image,
                              environment_variables=environment_variables, )
        return runtime_container
    except NotFound as e:
        error = f"Runtime image {image} not found"
        job_context.log_event(error, JobStatus.FAILED, error=str(e), image=image)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to create image {image}"
        job_context.log_event(error, JobStatus.FAILED, error=str(e), image=image)
        raise RuntimeError(error) from e


def login():
    client.login()