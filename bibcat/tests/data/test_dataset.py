import json
import os

import numpy as np
import pytest

from bibcat import config
from bibcat.core import parameters as params

mapper = config.pretrained.map_papertypes

# This test currently relies on the existing the TVT directory data. NEED TO REVISE OR UPDATE THIS SCRIPT not to rely on it if we decide to use the TVT spliting method. But we'd better use other ways to prepare Training/Validation/Test datasets in the future.


def file_exists(file_path):
    return os.path.exists(file_path)


# Test to verify the combined dataset
# @pytest.mark.skipif(
#     not (
#         file_exists(config.inputs.path_source_data)
#         and file_exists(config.inputs.path_papertext)
#         and file_exists(config.inputs.path_papertrack)
#     ),
#     reason="Required data files do not exist.",
# )
# def test_combined_dataset():
#     """test the combined dataset"""
#     print("Running test_combined_dataset.")

#     # Load each of the datasets
#     # For the combined dataset
#     with open(config.inputs.path_source_data, "r") as openfile:
#         data_combined = json.load(openfile)

#     # For the original text data
#     with open(config.inputs.path_papertext, "r") as openfile:
#         data_text = json.load(openfile)

#     # For the original classification data
#     with open(config.inputs.path_papertrack, "r") as openfile:
#         data_classif = json.load(openfile)

#     # Build list of bibcodes for the original data sources
#     list_bibcodes_text = [item["bibcode"] for item in data_text]
#     list_bibcodes_classif = [item["bibcode"] for item in data_classif]

#     # Check each combined data entry against original data sources
#     for ii in range(len(data_combined)):
#         # Skip if no text stored for this entry
#         if "body" not in data_combined[ii]:
#             print(f"No text for index {ii}.")
#             continue

#         # Extract bibcode
#         curr_bibcode = data_combined[ii]["bibcode"]

#         # Fetch indices of entries in original data sources
#         ind_text = list_bibcodes_text.index(curr_bibcode)
#         try:
#             ind_classif = list_bibcodes_classif.index(curr_bibcode)
#         except ValueError:
#             print(f"{curr_bibcode} not classified bibcode.")
#             continue

#         # Verify that combined data entry values match back to originals
#         # Check abstract
#         assert data_combined[ii].get("abstract") == data_text[ind_text].get("abstract")

#         # Check text
#         assert data_combined[ii]["body"] == data_text[ind_text]["body"]

#         # Check bibcodes (redundant test but that's ok)
#         assert curr_bibcode == data_text[ind_text]["bibcode"]
#         assert curr_bibcode == data_classif[ind_classif]["bibcode"]

#         # Check missions and classes
#         assert len(data_combined[ii]["class_missions"]) == len(data_classif[ind_classif]["class_missions"])
#         for curr_mission in data_combined[ii]["class_missions"]:
#             tmp_list = [item["mission"] for item in data_classif[ind_classif]["class_missions"]]
#             tmp_ind = tmp_list.index(curr_mission)
#             assert (
#                 data_combined[ii]["class_missions"][curr_mission]["papertype"]
#                 == data_classif[ind_classif]["class_missions"][tmp_ind]["paper_type"]
#             )

#     print("Run of test_combined_dataset complete.")


# Test to verify the TVT directory
@pytest.mark.skipif(
    not file_exists(config.paths.TVTinfo) or not file_exists(config.paths.modiferrors),
    reason="TVT directory files do not exist.",
)
def test_TVT_directory():
    """test the TVT directory"""

    print("Running test_TVT_directory.")
    dict_info = np.load(config.paths.TVTinfo, allow_pickle=True).item()
    dict_errors = np.load(config.paths.modiferrors, allow_pickle=True).item()

    # Dataset of text and classification information
    with open(config.inputs.path_source_data, "r") as openfile:
        dataset = json.load(openfile)

    list_bibcodes_dataset = [item["bibcode"] for item in dataset]

    # Ensure each entry in TVT storage has correct mission and class
    for curr_key in dict_info:
        ind_dataset = list_bibcodes_dataset.index(curr_key)

        # Fetch only allowed missions and classifs. from dataset
        curr_actuals = {
            key: dataset[ind_dataset]["class_missions"][key]
            for key in dataset[ind_dataset]["class_missions"]
            if any(item.identify_keyword(key)["bool"] for item in params.all_kobjs)
            and dataset[ind_dataset]["class_missions"][key]["papertype"] in config.pretrained.papertypes
        }

        # Fetch results of test and combine with any errors for bibcode
        curr_test = dict_info[curr_key]["storage"].copy()
        if curr_key in dict_errors:
            curr_test.update(dict_errors[curr_key])

        # Ensure same number of missions for this bibcode
        assert len(curr_actuals) == len(curr_test)

        # Check each mission and class
        for curr_textid in curr_test:
            curr_mission = curr_test[curr_textid]["mission"]
            curr_class = curr_test[curr_textid]["class"]
            assert curr_class == mapper[curr_actuals[curr_mission]["papertype"].lower()]

    print("Run of test_TVT_directory complete.")
