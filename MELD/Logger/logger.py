import logging

def setup_logger(config: dict):
    logger = logging.getLogger("meld")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)

    # TODO add file handler

    logger.addHandler(stdout_handler)

    return logger