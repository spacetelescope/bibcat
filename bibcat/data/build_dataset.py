"""
:title: build_dataset.py

This module will produce the input corpus data in JSON format by
combining the MAST papertrack JSON file and the ADS full text JSON file.

Run example: bibcat train

"""

import json
import os
from pathlib import Path

import numpy as np

from bibcat import config
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import load_json_file, save_json_file

logger = setup_logger(__name__)


def file_exists(filelist: list) -> bool:
    return any([os.path.isfile(item) for item in filelist])


def save_text_file(path_filename: Path, bibcodes: list[str]) -> None:
    np.savetxt(
        path_filename,
        np.array(bibcodes),
        delimiter="\n",
        fmt="%s",
    )


def load_datasets(path_papertext: Path, path_papertrack: Path) -> tuple[list[dict], list[dict]]:
    # Load paper texts and papertrack classes
    papertext_dataset = load_json_file(path_papertext)
    papertrack_dataset = load_json_file(path_papertrack)
    return papertext_dataset, papertrack_dataset


def extract_papertext_info(dataset) -> tuple[list[str], list[str]]:
    bibcodes = [entry["bibcode"] for entry in dataset]
    pubdates = [entry["pubdate"] for entry in dataset]
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


def missing_bibcodes_in_papertext(
    bibcodes_papertext: list[str],
    bibcodes_papertrack: list[str],
) -> list[str] | None:
    # Verify that all papers within papertrack are within the papertext database
    bibcodes_notin_papertext = [val for val in np.unique(bibcodes_papertrack) if (val not in bibcodes_papertext)]
    if len(bibcodes_notin_papertext) > 0:
        errstr = (
            "Note! Papers in papertrack not in text database!"
            + f"\n{bibcodes_notin_papertext}\n{len(bibcodes_notin_papertext)} of {len(bibcodes_papertrack)} papertrack entries in all.\n"
        )
        logger.error(errstr)
    return bibcodes_notin_papertext


def trim_papertext_dict(dataset: dict, keys: list) -> list[dict]:
    storage_combined_dataset = [
        {key: value for key, value in thisdict.items() if (key in keys)}.copy() for thisdict in dataset
    ]
    return storage_combined_dataset


def combine_datasets(papertext_data, papertrack_data):
    # First, store trimmed papertext dictionary down to only columns to include
    data_storage = trim_papertext_dict(papertext_data, config.inputs.keys_papertext)

    # Extract information from the papertrack classification dataset
    ads_searches, bibcodes_papertrack, missions_and_papertypes = extract_papertrack_info(papertrack_data)

    # Extract information from the paper text dataset
    bibcodes_papertext, _ = extract_papertext_info(papertext_data)

    # Verify that all papers within papertrack are within the papertext database and return the bibcodes
    bibcodes_notin_papertext = missing_bibcodes_in_papertext(bibcodes_papertext, bibcodes_papertrack)

    bibcodes_notin_papertrack = []

    for curr_index, curr_dict in enumerate(data_storage):
        # Extract information for current paper within text database
        curr_bibcode = curr_dict["bibcode"]

        if curr_bibcode in bibcodes_papertrack:
            index = bibcodes_papertrack.index(curr_bibcode)
            curr_dict["class_missions"] = {
                mission["mission"]: {"bibcode": curr_bibcode, "papertype": mission["paper_type"]}
                for mission in missions_and_papertypes[index]
            }
            for search in ads_searches[index]:
                curr_dict[f"is_ignored_{search['search_key']}"] = search["ignored"]
        else:
            logger.debug(f"Bibcode ({curr_dict['bibcode']}) not in papertrack database. Continuing...")
            bibcodes_notin_papertrack.append(curr_bibcode)
    logger.info(f"NOTE: {len(bibcodes_notin_papertrack)} papers in text data that were not in papertrack.")
    logger.info("Done generating dictionaries of combined papertrack+text data.")

    return (
        data_storage,
        bibcodes_notin_papertext,
        bibcodes_notin_papertrack,
    )


def save_files(dataset: dict, missing_papertext_bibcodes: list, missing_papertrack_bibcodes: list):
    # Save the combined dataset
    save_json_file(config.inputs.path_source_data, dataset)
    # Also save the bibcodes of the paper-texts not found in papertrack and papertext
    save_text_file(config.output.path_not_in_papertext, missing_papertext_bibcodes)
    save_text_file(config.output.path_not_in_papertrack, missing_papertrack_bibcodes)

    logger.info("Dataset generation complete.\n")
    logger.info(f"Combined .json file saved to:\n{config.inputs.path_source_data}\n")
    logger.info(f"Bibcodes not in papertext saved to:\n{config.output.path_not_in_papertext}\n")
    logger.info(f"Bibcodes not in papertrack saved to:\n{config.output.path_not_in_papertrack}\n")


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
        storage_combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack = combine_datasets(
            dataset_papertext_orig,
            dataset_papertrack_orig,
        )

        # Save the combined dataset
        save_files(storage_combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack)
        save_files(storage_combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack)
