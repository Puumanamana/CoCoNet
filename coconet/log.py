import logging
import psutil
from pathlib import Path


def get_mem(proc):
    info = proc.memory_full_info()

    return (info.rss, info.vms)

class MemoryTracer(logging.Filter):
    def __init__(self, process, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.process = process

    def mem_info(p):
        info = child.memory_full_info()
        return (info)

    def filter(self, record):
        (total_rss, total_vms) = get_mem(self.process)

        for child in self.process.children(recursive=True):
            (rss, vms) = get_mem(child)
            total_rss += rss
            total_vms += vms

        record.vms = f'{total_vms/2**30:>5.1f} GB'
        record.rss = f'{total_rss/2**30:>4.1f} GB'

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
    logger.addFilter(MemoryTracer(psutil.Process()))

    logger.setLevel(logging.DEBUG)

    if not any(isinstance(hdl, logging.FileHandler) for hdl in logger.handlers):
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False

        # Create a Formatter for formatting the log messages
        formatter = logging.Formatter(
            '{asctime} (VM:{vms}, RS:{rss}) {name:^15} {levelname}: {message}',
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
