import logging
import os

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _has_handler(logger: logging.Logger, handler_name: str) -> bool:
    return any(getattr(handler, "name", None) == handler_name for handler in logger.handlers)


def _setup_logger(name: str, level: int = logging.INFO, propagate: bool = False, log_dir: str | None = None, console: bool = True) -> logging.Logger:
    """
    
    :param name: 
    :param level: 
    :param propagate: 
    :param log_dir: 
    :param console: 
    :return: 
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    stream_handler_name = f"{name}_stream"

    # avoid duplicates
    if console and not _has_handler(logger, stream_handler_name):
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(level)
        stdout_handler.name = stream_handler_name
        logger.addHandler(stdout_handler)

    file_handler_name = f"{name}_file"

    if log_dir and not _has_handler(logger, file_handler_name):
        all_file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        all_file_handler.setFormatter(formatter)
        all_file_handler.setLevel(level)
        logger.addHandler(all_file_handler)

    logger.debug(f"Logger {name} initialized")

    return logger


def get_meld_logger() -> logging.Logger:
    """
    Sets up and retrieves a logger for the namespace "meld".

    :return: A configured logger instance for the "meld" namespace.
    :rtype: logging.Logger
    """
    return _setup_logger("meld", log_dir="/logs")

def get_job_logger(job_id: str, log_path: str) -> logging.Logger:
    """
    Retrieve a logger instance configured for a specific job.

    :param job_id: Unique identifier for the job.
    :type job_id: str
    :param log_path: Path to the directory where log files should be stored.
    :type log_path: str
    :return: Configured logger instance for the specified job.
    :rtype: logging.Logger
    """
    return _setup_logger(f"meld.job{job_id}", log_dir=log_path, propagate=True, console=False)

def get_inference_logger(job_id: str) -> logging.Logger:
    """
    Fetches and returns a logger instance that logs the stdout/stderr streams from a inference runtime container

    :param job_id: The identifier for the job. Used to create a unique logger name.
    :type job_id: str
    :return: A configured logger instance for the specified job's inference operations.
    :rtype: logging.Logger
    """
    return _setup_logger(f"meld.job{job_id}.inference", propagate=True, console=False)

