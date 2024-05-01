"""
:title: utils.py

This module stores any utility functions necessary for bibcat.

"""


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
