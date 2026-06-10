import logging
import os

def setup_logger(name: str, level: int = logging.INFO, logs_path: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logs_path = logs_path or "/logs"

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