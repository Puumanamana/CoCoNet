import logging
import psutil
from pathlib import Path


class MemoryTracer(logging.Filter):
    def filter(self, record):
        current_process = psutil.Process()
        vmem = current_process.memory_full_info().rss

        for child in current_process.children(recursive=True):
            vmem += child.memory_full_info().rss
        
        record.vmem = f'{int(vmem/2**20):,} MB'
        return True


def setup_logger(name, log_file, level=logging.INFO, pid=None):
    """
    Setup logging if not set, or return logger if already exists

    Args:
        name (str): name of logger
        log_file (str): path to save logs
        level (int): log level for stderr
    Returns:
        logging.Logger
    """

    # Create the Logger
    logger = logging.getLogger(name)
    logger.addFilter(MemoryTracer())

    logger.setLevel(logging.DEBUG)

    if not any(isinstance(hdl, logging.FileHandler) for hdl in logger.handlers):
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False

        # Create a Formatter for formatting the log messages
        formatter = logging.Formatter(
            '{asctime} (Mem: {vmem}) {name:^15} {levelname}: {message}',
            '%Y-%m-%d %H:%M:%S',
            style="{"
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
