import logging
import os

 def setup_logger(name: str, level: int = logging.INFO, logs_path: str = "/logs") -> logging.Logger:
    """
    Configures and initializes a logger with specific settings for logging to console and files.

    Parameters:
    name: str
        The name of the logger instance to be created.
    level: int, optional
        The logging level (e.g., DEBUG, INFO, WARNING) for the logger. Defaults to logging.INFO.
    logs_path: str, optional
        The directory path where log files should be saved. If not provided, the default directory
        "/logs" will be used.

    Returns:
    logging.Logger
        A configured logger instance with the specified settings.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # avoid duplicate
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(level)
        logger.addHandler(stdout_handler)

        all_file_handler = logging.FileHandler(os.path.join(logs_path, f"{name}.log"))
        all_file_handler.setFormatter(formatter)
        all_file_handler.setLevel(level)
        logger.addHandler(all_file_handler)

        stderr_file_handler = logging.FileHandler(os.path.join(logs_path, f"{name}.err.log"))
        stderr_file_handler.setLevel(logging.ERROR)
        logger.addHandler(stderr_file_handler)

        logger.info(f"Logger {name} initialized")


    return logger