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
    "Check if any file exists among the list of files"
    return any([os.path.isfile(item) for item in filelist])


def save_text_file(path_filename: Path, bibcodes: list[str]) -> None:
    try:
        with open(path_filename, mode="w") as file:
            file.write("\n".join(bibcodes))
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")


def load_datasets(path_papertext: Path, path_papertrack: Path) -> tuple[list[dict], list[dict]]:
    # Load paper texts and papertrack classes
    logger.info("Loading papertext and papertrack datasets!")
    papertext_dataset = load_json_file(path_papertext)
    papertrack_dataset = load_json_file(path_papertrack)
    logger.info("Loaded papertext and papertrack datasets!")
    return papertext_dataset, papertrack_dataset


def extract_papertext_info(dataset) -> tuple[list[str], list[str]]:
    bibcodes = [entry["bibcode"] for entry in dataset]
    pubdates = [entry["pubdate"] for entry in dataset]
    logger.info(f"The earliest date of papers within text database: {min(pubdates)}.")
    logger.info(f"The latest date of papers within text database: {max(pubdates)}.")

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
        logger.warning(
            "Note! Papers in papertrack not in text database!"
            + f"\n{bibcodes_notin_papertext}\n{len(bibcodes_notin_papertext)} of {len(bibcodes_papertrack)} papertrack entries in all.\n"
        )
    return bibcodes_notin_papertext


def trim_dict(dataset: dict, keys: list) -> list[dict]:
    logger.info(f"trimming the papertext dict with {keys}")
    logger.debug(f"the first entry of the loaded_papertext = {dataset[0]}")

    trimmed_dict = [{key: value for key, value in thisdict.items() if (key in keys)}.copy() for thisdict in dataset]
    logger.debug(f"Show the first entry of the trimmed data: \n {trimmed_dict[0]}")
    logger.debug(f"Dict trimming is complete.")

    return trimmed_dict


def combine_datasets(trimmed_papertext_data: list[dict], papertrack_data: list[dict]):
    logger.info("Start combining the two datasets.")
    # Extract information from the papertrack classification dataset
    ads_searches, bibcodes_papertrack, missions_and_papertypes = extract_papertrack_info(papertrack_data)

    # Extract information from the paper text dataset
    bibcodes_papertext, _ = extract_papertext_info(trimmed_papertext_data)

    # Verify that all papers within papertrack are within the papertext database and return the bibcodes
    bibcodes_notin_papertext = missing_bibcodes_in_papertext(bibcodes_papertext, bibcodes_papertrack)

    bibcodes_notin_papertrack = []
    papertext_index_notin_papertrack = []
    combined_dataset = []
    new_dict = {}
    for curr_index, curr_dict in enumerate(trimmed_papertext_data):
        # Extract information for current paper within text database
        curr_bibcode = curr_dict["bibcode"]

        if curr_bibcode in bibcodes_papertrack:
            index = bibcodes_papertrack.index(curr_bibcode)
            new_dict["class_missions"] = {
                mission["mission"]: {"bibcode": curr_bibcode, "papertype": mission["paper_type"]}
                for mission in missions_and_papertypes[index]
            }

            for search in ads_searches[index]:
                new_dict[f"is_ignored_{search['search_key']}"] = search["ignored"]

            combined_dataset.append({**curr_dict, **new_dict})
        else:
            logger.warning(f"Bibcode ({curr_dict['bibcode']}) not in papertrack database. Continuing...")
            bibcodes_notin_papertrack.append(curr_bibcode)
            papertext_index_notin_papertrack.append(curr_index)

    logger.info(f"NOTE: {len(bibcodes_notin_papertrack)} papers in text data that were not in papertrack.")
    logger.info("Done generating dictionaries of combined papertrack+text data.")

    return (
        combined_dataset,
        bibcodes_notin_papertext,
        bibcodes_notin_papertrack,
        papertext_index_notin_papertrack,
    )


def save_text_files(missing_papertext_bibcodes: list, missing_papertrack_bibcodes: list):
    # Also save the bibcodes of the paper-texts not found in papertrack and papertext
    save_text_file(config.output.path_not_in_papertext, missing_papertext_bibcodes)
    save_text_file(config.output.path_not_in_papertrack, missing_papertrack_bibcodes)

    logger.info(f"Bibcodes not in papertext saved to:\n{config.output.path_not_in_papertext}\n")
    logger.info(f"Bibcodes not in papertrack saved to:\n{config.output.path_not_in_papertrack}\n")


def build_dataset() -> None:
    logger.info("The script is building the dataset for bibcat!")

    # Load paper texts and papertrack classes
    dataset_papertext_orig, dataset_papertrack_orig = load_datasets(
        config.inputs.path_papertext, config.inputs.path_papertrack
    )
    # First, store trimmed papertext dictionary down to only columns to include
    trimmed_papertext_dataset = trim_dict(dataset_papertext_orig, config.inputs.keys_papertext)
    # combine the papertrack and papertext into one dataset
    combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack, papertext_index_notin_papertrack = (
        combine_datasets(
            trimmed_papertext_dataset,
            dataset_papertrack_orig,
        )
    )
    # Save the combined dataset and other files
    save_json_file(config.inputs.path_source_data, combined_dataset)
    logger.info("The combined dataset is saved!")

    # Save papertext data missing in the papertrack dataset; this may be used for ChatGPT use cases
    save_json_file(
        config.output.path_papertext_not_in_papertrack,
        [dataset_papertext_orig[index] for index in papertext_index_notin_papertrack],
    )
    logger.info("Saved the papertext data not in papertrack!")
    # Save missing bibcodes from the datasets.
    save_text_files(bibcodes_notin_papertext, bibcodes_notin_papertrack)

    logger.info("Saved bibcodes_notin_papertext and bibcodes_notin_papertrack!")
