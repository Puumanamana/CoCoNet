import logging
from pathlib import Path


def setup_logger(name, log_file, level=logging.INFO):

    # Create the Logger
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    if not any(isinstance(hdl, logging.FileHandler) for hdl in logger.handlers):
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False
        
        # Create a Formatter for formatting the log messages
        formatter = logging.Formatter(
            '%(asctime)s (%(name)s) %(levelname)s: %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )    

        # Create the Handler for logging data to a file
        Path(log_file.parent).mkdir(exist_ok=True)
        logger_handler = logging.FileHandler(str(log_file))
        logger_handler.setLevel(logging.DEBUG)
        logger_handler.setFormatter(formatter)
        logger.addHandler(logger_handler)

        # Create the Handler for logging data to console.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.getLevelName(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
