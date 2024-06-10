"""
:title: fetch_papers.py

This module fetches test input data for classification.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.path_source_data configured in bibcat/config.py.

- It fetches the set number of papers or all in the `bibcat/data/model_name/dir_test` folder
for classification.

"""

import json
import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


# load TVTinfo when using JSON data in TVT directories
def load_tvt_info(dir_datasets: str) -> dict:
    path = os.path.join(dir_datasets, "dict_TVTinfo.npy")
    logger.info(f"loading TVT info: {path}!")
    return np.load(path, allow_pickle=True).item()


# getting bibcodes for using dir_test
def get_bibcodes(dict_TVTinfo: dict, dir_test: str) -> list:
    return [key for key in dict_TVTinfo if dict_TVTinfo[key]["folder_TVT"] == dir_test]


# loading json dataset
def load_json_dataset(path: str) -> list:
    logger.info(f"Loading {path}!")
    with open(path, "r") as file:
        return json.load(file)


# Extract text information for the bibcodes reserved for testing
# Data for test set
def get_data(dataset: list, bibcodes: list) -> list:
    return [(i, data) for i, data in enumerate(dataset) if data["bibcode"] in bibcodes]


#  Shuffle, if requested
def shuffle_data(data: list) -> None:
    np.random.shuffle(data)


# set the maximum number of texts
def set_max_data(data: list, max_texts: int | None) -> list:
    return data[:max_texts] if max_texts is not None else data


# Process the text input into dictionary format for input text
def process_text_data(data: list) -> dict:
    texts = {}
    for idx, entry in data:
        info = {
            "text": entry["body"],
            "id": str(idx),
            "bibcode": entry["bibcode"],
            "missions": {},
        }
        # Iterate through missions for this paper
        for mission in entry["class_missions"]:
            for kobj in params.all_kobjs:
                name = kobj.get_name()
                if kobj.identify_keyword(mission)["bool"]:
                    info["missions"][name] = {
                        "mission": name,
                        "class": entry["class_missions"][mission]["papertype"],
                    }
                else:
                    info["missions"][name] = {
                        "mission": name,
                        "class": config.results.verdict_rejection,
                    }
        texts[str(idx)] = info
    return texts


# logging summary
def log_summary(texts: dict) -> None:
    logger.debug("Text Summary")

    for key, value in texts.items():
        logger.debug(f"Entry: {key}")
        logger.debug(f"ID: {value['id']}")
        logger.debug(f"Bibcode: {value['bibcode']}")
        logger.debug(f"Missions: {value['missions']}")
        logger.debug(f"Start of text:\n{value['text'][:500]}")
        logger.debug("-")


def fetch_papers(
    do_evaluation: bool = False,
    do_shuffle: bool | None = None,
    do_verbose_text_summary: bool = False,
    max_texts: int | None = None,
) -> dict:
    # perpare papers to perform model evaluation

    if not do_evaluation:
        return load_json_dataset(config.inputs.path_ops_data)

    # For use of real papers from test dataset to test on
    # Load information for processed bibcodes reserved for testing
    else:
        dir_datasets = os.path.join(config.paths.partitioned, config.output.name_model)
        dir_test = config.output.folders_TVT["test"]
        dict_TVTinfo = load_tvt_info(dir_datasets)
        test_bibcodes = get_bibcodes(dict_TVTinfo, dir_test)
        dataset = load_json_dataset(config.inputs.path_source_data)
        test_data = get_data(dataset, test_bibcodes)

        if do_shuffle:
            shuffle_data(test_data)

        selected_data = set_max_data(test_data, max_texts)
        texts = process_text_data(selected_data)

        if do_verbose_text_summary:
            log_summary(texts)

        return texts


if __name__ == "__main__":
    logger.info("The script is running as a standalone script. Fetching papers!")
    fetch_papers()
