import json
import os
import pathlib

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def count_mission_papertype_occurences(data: dict, target_key: str, mission: str, papertype: str) -> int:
    """Count the occurences of a specific mission

    Count the occurences of a specific mission in a second-level specific key

    Parameters
    ==========
    data: dict
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
            if isinstance(item, list):  # e.g.,  "llm"'s value
                for entry in item[target_key]:
                    if {mission: papertype} in entry:
                        count += 1
            elif isinstance(item, dict):  # e.g., the value of "human"
                if {mission: papertype} in item.values():
                    count += 1
        else:
            KeyError(f"{target_key} doesn't exist")
    logger.debug(f"The number of {mission}_{papertype} : {count}")

    return count


def get_threshold(data: dict) -> float:
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


def evaluation_stats(threshold: float) -> None:
    """Create a statistics table json file for distict mission + papertype classifications

    Parameters
    ==========
    threshold: float
        threshold_acceptance in the eval_output_file

    Returns
    =======
    """

    # read the evaluation summary output file
    eval_output = (
        pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_output_file}"
    )
    logger.info(f"reading {eval_output}")
    data = read_output(filename=eval_output)

    stats_table = {}
    missions = config.missions
    papertypes = config.llms.classifications

    for mission in missions:
        for papertype in papertypes:
            count = count_mission_papertype_occurences(data, "llm", mission, papertype)
            stats_table.update({mission + "_" + papertype: count})
    logger.info(f"Mission stats table = {stats_table}")

    out = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_stats_file}"

    if not os.path.exists(out):
        # create a new file
        logger.info(f"Writing output to {out}")
        with open(out, "w") as f:
            json.dump(stats_table, f, indent=2, sort_keys=False)
    else:
        FileExistsError(
            f"{out} already exists. Are you sure you want to overwrite the file? Choose a different name for the output in 'config.llms.eval_stats_file', if you want to keep the existing file"
        )
