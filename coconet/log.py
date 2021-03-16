"""
Logging configuration and initialization
"""

from pathlib import Path
import logging
import psutil


class MemoryTracer(logging.Filter):
    """
    To track memory usage at different steps
    Used memory is computed as the PSS of the main program
    + the sum of the PSS of its children for linux, and RSS
    + the sum of the USS of its children for MacOS
    """

    def filter(self, record):
        process = psutil.Process()
        mem = process.memory_full_info()

        if hasattr(mem, 'pss'):
            mem = mem.pss
        else: # No PSS info for MacOS
            mem = mem.rss

        for child in process.children(recursive=True):
            try:
                mem += child.memory_full_info().pss
            except AttributeError: # No PSS info for MacOS
                mem += child.memory_full_info().uss
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                pass

        record.mem = f'{mem/2**30:>5.1f} GB'

        return True


def setup_logger(name, log_file, level=logging.INFO):
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
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.propagate = False

        # Create a Formatter for formatting the log messages
        formatter = logging.Formatter(
            '{asctime} (Mem:{mem}) {name:^15} {levelname}: {message}',
            '%H:%M:%S',
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
