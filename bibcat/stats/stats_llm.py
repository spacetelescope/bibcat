import json
import os
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import save_json_file

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def unique_mission_papertypes(data: Dict[str, Any], target_key: str) -> Set[Tuple[str, str]]:
    """Extracts a unique set of mission-papertype pairs from the given target_key, "llm" or "human"

    Parameters
    ==========
    data: Dict[str, Any]
        the dict of the evaluation data of `config.llms.eval_output_file `
    target_key: str
        target key (the second-level key) to search for, e.g., "llm" or "human"

    Returns
    =======
    ordered_unique_set: Set[Tuple[str, str]]
        A unique set of mission-papertype pairs
    """

    # def ordered_set(iterable):
    #     return dict.fromkeys(iterable).keys()

    unique_set = set()
    for item in data.values():
        if target_key in item:
            if isinstance(item[target_key], list):
                for entry in item[target_key]:
                    if isinstance(entry, dict):
                        for mission, papertype in entry.items():
                            unique_set.add((mission, papertype))
            elif isinstance(item[target_key], dict):
                for mission, papertype in item[target_key].items():
                    unique_set.add((mission, papertype))
    # ordered_unique_set = ordered_set(unique_set)

    return unique_set


def count_mission_papertype_occurences(data: Dict[str, Any], target_key: str, mission: str, papertype: str) -> int:
    """Count the occurences of a specific pair of mission and papertype in a second-level specific key

    Parameters
    ==========
    data: Dict[str, Any]
        the dict of the evaluation data of `config.llms.eval_output_file `
    target_key: str
        target key (the second-level key) to search for, e.g., "llm" or "human"
    mission: str
        mission name, e.g., "HST"
    papertype: str
        papertype, e.g., "SCIENCE" or "MENTION"

    Returns
    =======
    count
        int
        The count of target_key
    """
    count = 0

    for item in data.values():  # item = `each bibcode value
        if target_key in item:
            if isinstance(item[target_key], list):  # e.g.,  "llm"'s value
                for entry in item[target_key]:
                    if entry.get(mission) == papertype:
                        count += 1
            elif isinstance(item[target_key], dict):  # e.g., the value of "human"
                if item[target_key].get(mission) == papertype:
                    count += 1
        else:
            raise KeyError(f"{target_key} doesn't exist")
    logger.debug(f"The number of {mission}_{papertype} : {count}")

    return count


def get_threshold(data: Dict[str, Any]) -> float:
    """Get the threshold value for acceptance assuming the papertype was determined by one fixed threshold value.

    Parameters
    ==========
    data: dict
        the dict of the evaluation data of `config.llms.eval_output_file`

    Returns
    =======
    threshold:float
        threshold_acceptance in the eval_output_file
    """

    first_key = next(iter(data))
    threshold = data[first_key].get("threshold_acceptance")
    return threshold


def create_stats_table(data: Dict[str, Any], target_key: str) -> Dict[str, float]:
    """Create a JSON file of the counts of all distict mission + papertype classifications

    Parameters
    ==========
    data: Dict[str, Any]
        the dict of the evaluation data of `config.llms.eval_output_file`

    Returns
    =======
    stats_dict: dict
        dictionary of mission+papertype counts
    """
    threshold = get_threshold(data)
    stats_dict = {"threshold": threshold}
    unique_types = unique_mission_papertypes(data, target_key=target_key)
    for mission, papertype in unique_types:
        count = count_mission_papertype_occurences(data, target_key, mission, papertype)
        stats_dict.update({target_key + "_" + mission + "_" + papertype: count})
    logger.info(f"Mission stats table = {stats_dict}")
    return stats_dict


def save_evaluation_stats(filepath: Path) -> None:
    """Save the evaluation stats in a json file

    Parameters
    ==========
    filepath: Path
        filepath to save the JSON file named 'config.llms.eval_stats_file'.

    Returns
    =======
    """

    # read the evaluation summary output file
    eval_output = Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_output_file}"
    logger.info(f"reading {eval_output}")
    data = read_output(filename=eval_output)

    stats_table = create_stats_table(data, target_key="llm")
    stats_table.update(create_stats_table(data, target_key="human"))

    # writing the stats table JSON
    if not os.path.exists(filepath):
        save_json_file(
            filepath,
            stats_table,
        )
    else:
        raise FileExistsError(
            f"{filepath} already exists. Are you sure you want to overwrite the file? Choose a different name for the output in 'config.llms.eval_stats_file', if you want to keep the existing file"
        )
