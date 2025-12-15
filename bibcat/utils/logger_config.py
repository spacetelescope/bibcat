import logging
import os
import time

from bibcat import config

# Define the logs directory at output root path with current date
LOG_DIR = os.path.join(config.output.root_path, "logs", time.strftime("%Y-%m-%d"))

# Get the start time in a readable format
start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Set up logger"""
    # Create a custom logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    log_file_path = os.path.join(LOG_DIR, f"bibcat_log_{start_time}.log")

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path)
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
