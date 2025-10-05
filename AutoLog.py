import logging
import coloredlogs
from termcolor import colored

class CustomFormatter(logging.Formatter):
    LOG_COLORS = {
        logging.DEBUG: 'blue',
        logging.INFO: 'green',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red',
    }

    def format(self, record):
        log_format = "%(asctime)s ==========%(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        formatted_message = formatter.format(record)

        color = self.LOG_COLORS.get(record.levelno, 'white')
        return colored(formatted_message, color)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Stop log messages from being propagated to the parent logger, avoid repetition

    # Only add a new handler if the logger doesn't have any, avoid repetitive messages
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = CustomFormatter()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger