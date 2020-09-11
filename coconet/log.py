import logging
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):

    # Create the Logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # We already set a logger with the same name
        return logger

    logger.setLevel(logging.DEBUG)

    # Create the Handler for logging data to a file
    Path(log_file.parent).mkdir(exist_ok=True)
    logger_handler = logging.FileHandler(str(log_file))
    logger_handler.setLevel(logging.DEBUG)

    # Create the Handler for logging data to console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.getLevelName(level))

    # Create a Formatter for formatting the log messages
    formatter = logging.Formatter(
        '%(asctime)s (%(name)s) %(levelname)-8s: %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )

    # Add the Formatter to the Handler
    logger_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    logger.addHandler(console_handler)

    return logger
