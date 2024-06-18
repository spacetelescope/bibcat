"""
:title: build_dataset.py

This module will produce the input corpus data in JSON format by
combining the MAST papertrack JSON file and the ADS full text JSON file.

Run example: bibcat train

"""

import json
import os

import numpy as np

from bibcat import config
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import load_json_file, save_json_file

logger = setup_logger(__name__)


def file_exists(filelist: list) -> bool:
    return any([os.path.isfile(item) for item in filelist])


def save_numpy_file(path: str, bibcodes: list[str]) -> None:
    np.savetxt(
        path,
        np.asarray(bibcodes).astype(str),
        delimiter="\n",
        fmt="%s",
    )


def load_datasets(path_papertext: str, path_papertrack: str) -> tuple[list, list]:
    # Load paper texts and papertrack classes
    papertext_dataset = load_json_file(path_papertext)
    papertrack_dataset = load_json_file(path_papertrack)
    return papertext_dataset, papertrack_dataset


def extract_papertext_info(dataset) -> tuple[list, str, str]:
    bibcodes = [item["bibcode"] for item in dataset]
    pubdates = [item["pubdate"] for item in dataset]
    logger.debug(f"The earliest date of papers within text database: {min(pubdates)}.")
    logger.debug(f"The latest date of papers within text database: {max(pubdates)}.")

    return bibcodes, pubdates


def extract_papertrack_info(dataset) -> tuple[list[None | dict], list[None | str], list[None | dict]]:
    searches = [entry["searches"] for entry in dataset]
    bibcodes = [entry["bibcode"] for entry in dataset]
    missions_and_papertypes = [entry["class_missions"] for entry in dataset]

    # Throw an error if there are duplicate bibcodes in the papertrack classification dataset
    if len(set(bibcodes)) != len(bibcodes):
        raise ValueError("Err: Duplicate bibcodes in database of paper classifications!")

    return searches, bibcodes, missions_and_papertypes


def missing_bibcodes_in_papertext(bibcodes_papertrack: list[str], bibcodes_papertext: list[str]) -> list[str] | None:
    # Verify that all papers within papertrack are within the papertext database
    bibcodes_notin_papertext = [val for val in np.unique(bibcodes_papertrack) if (val not in bibcodes_papertext)]
    if len(bibcodes_notin_papertext) > 0:
        errstr = (
            "Note! Papers in papertrack not in text database!"
            + f"\n{bibcodes_notin_papertext}\n{len(bibcodes_notin_papertext)} of {len(bibcodes_papertrack)} papertrack entries in all.\n"
        )
        # raise ValueError(errstr)
        logger.debug(errstr)
    return bibcodes_notin_papertext


def trim_papertext_dict(dataset: dict, keys: list) -> list[dict]:
    try:
        storage_combined_dataset = [
            {key: value for key, value in thisdict.items() if (key in keys)}.copy() for thisdict in dataset
        ]
    except AttributeError:  # If this error raised, probably earlier Python vers.
        storage_combined_dataset = [
            {key: value for key, value in thisdict.iteritems() if (key in keys)}.copy() for thisdict in dataset
        ]
    return storage_combined_dataset


def combine_datasets(papertrack_data, papertext_data) -> None:
    # First, store trimmed papertext dictionary down to only columns to include
    data_storage = trim_papertext_dict(papertext_data, config.inputs.keys_papertext)

    # Extract information from the papertrack classification dataset
    ads_searches, bibcodes_papertrack, missions_and_papertypes = extract_papertrack_info(papertrack_data)

    # Extract information from the paper text dataset
    bibcodes_papertext, _ = extract_papertext_info(papertext_data)

    # Verify that all papers within papertrack are within the papertext database and return the bibcodes
    bibcodes_notin_papertext = missing_bibcodes_in_papertext(bibcodes_papertrack, bibcodes_papertext)

    bibcodes_notin_papertrack = []

    for curr_index, curr_dict in enumerate(data_storage):
        # Extract information for current paper within text database
        curr_bibcode = curr_dict["bibcode"]

        # Extract index for current paper within papertrack (paper classification database)
        curr_index_papertrack = None
        try:
            curr_index_papertrack = bibcodes_papertrack.index(curr_bibcode)
        except ValueError:
            logger.debug(f"Bibcode ({curr_index}, {curr_bibcode}) not in papertrack database. Continuing...")
            bibcodes_notin_papertrack.append(curr_bibcode)
            continue

        # Copy over data from papertrack into text database

        # curr_dict["class_missions"] = {}
        for _, dict_content in enumerate(missions_and_papertypes[curr_index_papertrack]):
            curr_mission = dict_content["mission"]
            # curr_papertype = dict_content["paper_type"]
            # Store inner dictionary under mission name
            # inner_dict = {}
            # curr_dict["class_missions"][curr_mission] = inner_dict
            curr_dict[curr_mission] = {
                "class_missions": {
                    "bibcode": bibcodes_papertrack[curr_index_papertrack],
                    "papertype": dict_content["paper_type"],
                }
            }
            # Store information in inner dict
            # inner_dict["bibcode"] = bibcodes_papertrack[curr_index_papertrack]
            # inner_dict["papertype"] = curr_papertype

        # Store search ignore flags
        for _, searches in enumerate(ads_searches[curr_index_papertrack]):
            curr_searchname = searches["search_key"]
            curr_dict[f"is_ignored_{curr_searchname}"] = searches["ignored"]

        logger.debug("Done generating dictionaries of combined papertrack+text data.")
    logger.debug(f"NOTE: {len(bibcodes_notin_papertrack)} papers in text data that were not in papertrack.")

    return data_storage, bibcodes_notin_papertrack, bibcodes_notin_papertext


def save_files(dataset: dict, missing_papertrack_bibcodes: list, missing_papertext_bibcodes: list):
    # Save the combined dataset
    save_json_file(config.inputs.path_source_data, dataset)
    # Also save the bibcodes of the paper-texts not found in papertrack and papertext
    save_numpy_file(config.inputs.path_not_in_papertext, missing_papertext_bibcodes)
    save_numpy_file(config.inputs.path_not_in_papertrack, missing_papertrack_bibcodes)

    logger.debug("Dataset generation complete.\n")
    logger.debug(f"Combined .json file saved to:\n{config.inputs.path_source_data}\n")
    logger.debug(f"Bibcodes not in papertext saved to:\n{config.inputs.path_not_in_papertext}\n")
    logger.debug(f"Bibcodes not in papertrack saved to:\n{config.inputs.path_not_in_papertrack}\n")


def build_dataset() -> None:
    logger.info("The script is building the dataset for bibcat!")

    # Throw an error if any of these files already exist
    tmp_list = [
        config.inputs.path_source_data,
        config.output.path_not_in_papertext,
        config.output.path_not_in_papertrack,
    ]
    if file_exists(tmp_list):
        logger.error(
            "Err: File in the following list already exists and will not be overwritten."
            + f"\nPlease change the save destination for the notebook file(s) or move the existing files.\n{tmp_list}"
        )
        return

    else:
        # Load paper texts and papertrack classes
        dataset_papertext_orig, dataset_papertrack_orig = load_datasets(
            config.inputs.path_papertext, config.inputs.path_papertrack
        )
        # combine the papertrack and papertext into one dataset
        storage_combined_dataset, bibcodes_notin_papertrack, bibcodes_notin_papertext = combine_datasets(
            dataset_papertrack_orig, dataset_papertext_orig
        )

        # Save the combined dataset
        save_files(storage_combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack)
