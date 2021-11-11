import os
import logging

_logger_instance = None


def create_logger_singleton(name="rotary",
                            log_level="INFO",
                            log_file=None,
                            file_mode="a"):
    """
        file mode: the default is a, which means append
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s -  %(filename)-30s:%(lineno)-4d - %(funcName)s() %(message)s"
    )

    # create file handler which logs even debug messages
    if log_file is not None:
        fid = os.path.realpath(log_file)
        fh = logging.FileHandler(fid, file_mode)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger_instance():
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = create_logger_singleton()

    return _logger_instance
