"""
:title: fetch_papers.py

This module fetches test input data for classification.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.path_source_data configured in bibcat/config.py.

- It fetches the set number of papers or all in the `bibcat/data/partitioned_datasets/model_name/dir_test` folder
for classification.

"""

import json
import os

import numpy as np

from bibcat import config
from bibcat import parameters as params


# Fetch papers of
def fetch_papers(
    dir_datasets: str,
    dir_test: str,
    do_shuffle: bool = True,
    do_verbose_text_summary: bool = True,
    max_tests: None | int = None,
) -> dict:
    # perpare papers to test on
    # For use of real papers from test dataset to test on
    # Load information for processed bibcodes reserved for testing
    dict_TVTinfo = np.load(os.path.join(dir_datasets, "dict_TVTinfo.npy"), allow_pickle=True).item()
    list_test_bibcodes = [key for key in dict_TVTinfo if (dict_TVTinfo[key]["folder_TVT"] == dir_test)]

    # Load the original data
    with open(config.inputs.path_source_data, "r") as openfile:
        dataset = json.load(openfile)
    # Extract text information for the bibcodes reserved for testing
    # Data for test set
    list_test_indanddata_raw = [
        (ii, dataset[ii]) for ii in range(0, len(dataset)) if (dataset[ii]["bibcode"] in list_test_bibcodes)
    ]
    # Shuffle, if requested
    if do_shuffle:
        np.random.shuffle(list_test_indanddata_raw)

    # Extract target number of test papers from the test bibcodes
    if max_tests is not None:  # Fetch subset of tests
        list_test_indanddata = list_test_indanddata_raw[0:max_tests]
    else:  # Use all tests
        list_test_indanddata = list_test_indanddata_raw
    # Process the text input into dictionary format for inputting into the codebase
    dict_texts = {}  # To hold formatted text entries
    for ii in range(0, len(list_test_indanddata)):
        curr_ind = list_test_indanddata[ii][0]
        curr_data = list_test_indanddata[ii][1]
        # Convert this data entry into dictionary with: key:text,id,bibcode,
        # mission structure
        curr_info = {
            "text": curr_data["body"],
            "id": str(curr_ind),
            "bibcode": curr_data["bibcode"],
            "missions": {},
        }
        # Iterate through missions for this paper
        for curr_mission in curr_data["class_missions"]:
            # Iterate through declared Keyword objects
            for curr_kobj in params.all_kobjs:
                curr_name = curr_kobj.get_name()
                # Store mission data under keyword name, if applicable
                if curr_kobj.identify_keyword(curr_mission)["bool"]:
                    curr_info["missions"][curr_name] = {
                        "mission": curr_name,
                        "class": curr_data["class_missions"][curr_mission]["papertype"],
                    }
                # Otherwise, store that this mission was not detected for this text
                else:
                    curr_info["missions"][curr_name] = {"mission": curr_name, "class": config.results.verdict_rejection}
        # Store this data entry
        dict_texts[str(curr_ind)] = curr_info

    # Print some notes about the testing data
    # print(f"Number of texts in text set: {dict_texts}")
    if do_verbose_text_summary:
        print("Text Summary\n")
        for key in dict_texts:
            print(f"Entry: {key}")
            print(f"ID: {dict_texts[key]['id']}")
            print(f"Bibcode: {dict_texts[key]['bibcode']}")
            print(f"Missions: {dict_texts[key]['missions']}")
            print(f"Start of text:\n{dict_texts[key]['text'][0:500]}")
            print("-\n")

    return dict_texts


# This section checks if the script is the main program
if __name__ == "__main__":
    # Code here will only execute if the script is run directly, not if it's imported as a module
    # Currently, text data is fed from the TVT test folder but this can be changed when a need arises.
    print("The script is running as a standalone script though I don't see yet a purpose for it.\n Fetching papers!")
    fetch_papers(
        dir_datasets=os.path.join(config.paths.partitioned, config.output.name_model), dir_test=config.output.folders_TVT["test"]
    )
