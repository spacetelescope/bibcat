"""
:title: utils.py

This module stores any utility functions necessary for bibcat.

"""

import json
from pathlib import Path

import numpy as np

from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def convert_sets(obj: object) -> list | dict | object:
    """
    When data structure is complex or contains many sets, this preprocessing function walks through the entire data structure (e.g., nested dictionaries, lists) and converts all sets to lists.
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_sets(value) for key, value in obj.items()}
    return obj


def load_json_file(path: Path) -> None:
    try:
        logger.info(f"Loading {path}!")
        with open(path, "r") as openfile:
            return json.load(openfile)
    except IOError as e:
        logger.error(f"An error occurred while reading the file: {e}")


def save_json_file(path: Path, dataset: list[dict], indent: int = 2) -> None:
    try:
        logger.info(f"Saving {path}!")
        with open(path, "w") as openfile:
            json.dump(dataset, openfile, indent=indent)
    except IOError as e:
        logger.error(f"An error occurred while saving the file: {e}")


# Create a class for numpy encoder to convert a numpy array into a list
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
