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


def stream_container_logs(container: Container, job_context: JobContext):
    try:
        for stdout_chunk, stderr_chunk in container.attach(stdout=True, stderr=True, stream=True, logs=True,
                                                           demux=True):
            if stdout_chunk:
                text = stdout_chunk.decode("utf-8", errors="replace")
                job_context.logger.info(text.rstrip())

            if stderr_chunk:
                text = stderr_chunk.decode("utf-8", errors="replace")
                job_context.logger.error(text.rstrip())

            sys.stdout.flush()
            sys.stderr.flush()
    except Exception:
        logger.exception("Error while streaming container logs")


def pull_image(image: str, job_context: JobContext, ):
    try:
        job_context.log_event(f"Pulling runtime image {image}", JobStatus.PULLING_IMAGE, image=image)
        client.images.pull(image)
        job_context.log_event(f"Pulled runtime image {image}", JobStatus.IMAGE_PULLED, image=image)
    except NotFound as e:
        error = f"Runtime image {image} not found"
        job_context.log_event(error, JobStatus.FAILED, error=str(e), image=image)
        raise RuntimeError(error)
    except APIError as e:
        error = f"Failed to pull image {image}"
        job_context.log_event(error, JobStatus.FAILED, error=str(e), image=image)
        raise RuntimeError(error) from e


def ensure_image_exists(image: str, job_context: JobContext):
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


def start_container(container: Container, job_context: JobContext, ):
    try:
        job_context.logger.info(f"Starting runtime container {container.id}")
        container.start()
        job_context.log_event("Starting runtime container", JobStatus.RUNNING)
    except APIError as e:
        error = f"Failed to start runtime container"
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e


def wait_for_container(container: Container, job_context: JobContext):
    TIMEOUT_SECONDS = 500

    # read container stdout in another thread
    log_thread = threading.Thread(
        target=stream_container_logs,
        args=(container, job_context),
        daemon=True,
    )
    log_thread.start()

    try:
        result = container.wait(timeout=TIMEOUT_SECONDS)
        exit_code = result["StatusCode"]

        if exit_code != 0:
            error = f"Runtime container failed with exit code {exit_code}"
            job_context.log_event("Inference failed", JobStatus.FAILED,
                                  error=error)
            raise RuntimeError(error)
        job_context.log_event("Inference has completed successfully", JobStatus.SUCCESS)
    except requests.exceptions.ReadTimeout:
        error = f"Runtime container timed out after {TIMEOUT_SECONDS} seconds"
        job_context.log_event(error, JobStatus.TIMEOUT, timeout=TIMEOUT_SECONDS)
        container.kill()
        raise TimeoutError(error)
    finally:
        # clean up thread
        log_thread.join(timeout=5)

        if log_thread.is_alive():
            logger.warning("Container log streaming thread is still running")


def stop_container(container: Container, job_context: JobContext):
    try:
        job_context.logger.info(f"Stopping runtime container {container.id}")
        container.stop()
        job_context.logger.info(f"Stopped runtime container {container.id}")
    except APIError as e:
        error = f"Failed to stop runtime container"
        job_context.log_event(error, JobStatus.FAILED, error=str(e))
        raise RuntimeError(error) from e


def destroy_container(container: Container, job_context: JobContext):
    job_context.logger.info(f"Destroying container {container.id}")
    container.remove()
    job_context.log_event("Container destroyed", JobStatus.DESTROYED)


def create_container(image: str, job_context: JobContext, ) -> Container:
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
