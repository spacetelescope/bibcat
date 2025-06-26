"""
:title: utils.py

This module stores any utility functions necessary for bibcat.

"""

import json
from pathlib import Path
from typing import Any

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


def save_json_file(path: Path, dataset: list[dict] | dict, indent: int = 2) -> None:
    try:
        logger.info(f"Saving {path}!")
        with open(path, "w") as openfile:
            json.dump(dataset, openfile, indent=indent, cls=NumpyEncoder)
    except IOError as e:
        logger.error(f"An error occurred while saving the file: {e}")


# Create a class for numpy encoder to convert numpy types into native data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(NumpyEncoder, self).default(obj)


# Fetch a keyword object that matches the given lookup
def fetch_keyword_object(
    keyword_objs, lookup: str, do_raise_emptyerror: bool = True, verbose: bool = False
) -> Any | None:
    """Fetch a keyword object

    Given an input lookup string, tries to match it to a stored Keyword instance.

    Parameters
    ----------
    lookup : str
        the input string
    do_raise_emptyerror : bool, optional
        Flag to raise an error, by default True

    Returns
    -------
    Any | None
        the matching Keyword instance

    Raises
    ------
    ValueError
        when no match is found
    """

    # Print some notes
    if verbose:
        logger = setup_logger(__name__)
        logger.info(f"> Running _fetch_keyword_object() for lookup term {lookup}.")

    # Find keyword object that matches to given lookup term
    match = None
    for kobj in keyword_objs:
        # If current keyword object matches, record and stop loop
        if kobj.identify_keyword(lookup)["bool"]:
            match = kobj
            break

    # Throw error if no matching keyword object found
    if match is None:
        errstr = f"No matching keyword object for {lookup}.\n"
        errstr += "Available keyword objects are:\n"
        # just use the names of the keywords
        names = ", ".join(a._get_info("name") for a in keyword_objs)
        errstr += f"{names}\n"

        # Raise error if so requested
        if do_raise_emptyerror:
            raise ValueError(errstr)

    # Return the matching keyword object
    return match
