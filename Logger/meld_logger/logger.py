import logging

def setup_logger(name: str, config: dict = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(level)

        # TODO add file handler

        logger.addHandler(stdout_handler)

        logger.info(f"Logger {name} initialized")


    return logger