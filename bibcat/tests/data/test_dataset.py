"""
:title: test_dataset.py

Testing constructing datasets and TVT directories.

"""

import json
import os
import unittest

import numpy as np

from bibcat import config
from bibcat import parameters as params

mapper = params.map_papertypes


class TestData(unittest.TestCase):
    # For tests of .json dataset file combined from papertrack and papertext:
    if True:
        # Test that combined .json file components are correct
        def test__combined_dataset(self):
            print("Running test__combined_dataset.")
            # Load each of the datasets
            # For the combined dataset
            with open(config.inputs.path_source_data, "r") as openfile:
                data_combined = json.load(openfile)

            # For the original text data
            with open(config.inputs.path_papertext, "r") as openfile:
                data_text = json.load(openfile)

            # For the original classification data
            with open(config.inputs.path_papertrack, "r") as openfile:
                data_classif = json.load(openfile)

            # Build list of bibcodes for the original data sources
            list_bibcodes_text = [item["bibcode"] for item in data_text]
            list_bibcodes_classif = [item["bibcode"] for item in data_classif]

            # Check each combined data entry against original data sources
            for ii in range(0, len(data_combined)):
                # Skip if no text stored for this entry
                if "body" not in data_combined[ii]:
                    print("test__combined_dataset: No text for index {0}.".format(ii))
                    continue

                # Extract bibcode
                curr_bibcode = data_combined[ii]["bibcode"]  # Curr. bibcode

                # Fetch indices of entries in original data sources
                ind_text = list_bibcodes_text.index(curr_bibcode)
                try:
                    ind_classif = list_bibcodes_classif.index(curr_bibcode)
                except ValueError:
                    print("test__combined_dataset: {0} not classified bibcode.".format(curr_bibcode))
                    continue

                # Verify that combined data entry values match back to originals
                # Check abstract, if exists
                try:
                    if "abstract" in data_combined[ii]:
                        self.assertEqual(data_combined[ii]["abstract"], data_text[ind_text]["abstract"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Diff. abstract in bibcode {0}:\n\n{1}...\nvs\n{2}...".format(
                            curr_bibcode, data_combined[ii]["abstract"][0:500], data_text[ind_text]["abtract"][0:500]
                        )
                    )
                    print("---")
                    print("")

                    raise AssertionError()

                # Check text
                try:
                    self.assertEqual(data_combined[ii]["body"], data_text[ind_text]["body"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Different text in bibcode {0}:\n\n{1}...\nvs\n{2}...".format(
                            curr_bibcode, data_combined[ii]["body"][0:500], data_text[ind_text]["body"][0:500]
                        )
                    )
                    print("---")
                    print("")

                    raise AssertionError()

                # Check bibcodes (redundant test but that's ok)
                try:
                    self.assertEqual(curr_bibcode, data_text[ind_text]["bibcode"])
                    self.assertEqual(curr_bibcode, data_classif[ind_classif]["bibcode"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        ("Different bibcodes:\nCombined: {0}" + "\nText: {1}\nClassif: {2}").format(
                            curr_bibcode, data_text[ind_text]["bibcode"], data_classif[ind_classif]["bibcode"]
                        )
                    )
                    print("---")
                    print("")

                    raise AssertionError()

                # Check missions and classes
                try:
                    self.assertEqual(
                        len(data_combined[ii]["class_missions"]), len(data_classif[ind_classif]["class_missions"])
                    )  # Ensure equal number of classes
                    for curr_mission in data_combined[ii]["class_missions"]:
                        tmp_list = [item["mission"] for item in data_classif[ind_classif]["class_missions"]]
                        tmp_ind = tmp_list.index(curr_mission)
                        self.assertEqual(
                            data_combined[ii]["class_missions"][curr_mission]["papertype"],
                            data_classif[ind_classif]["class_missions"][tmp_ind]["paper_type"],
                        )
                except (IndexError, AssertionError):
                    print("")
                    print(">")
                    print(
                        ("Different missions and classes:\nCombined: {0}" + "\nClassif: {1}").format(
                            data_combined[ii]["class_missions"], data_classif[ind_classif]["class_missions"]
                        )
                    )
                    print("---")
                    print("")

                    raise AssertionError()

            print("Run of test__combined_dataset complete.")

        # Test that files of TVT directory (if exist) are correctly stored
        def test__TVT_directory(self):
            print("Running test__TVT_directory.")
            # Load the datasets
            # Check if TVT information exists
            if not os.path.isfile(config.paths.TVTinfo):
                raise AssertionError("TVT directory does not exist yet at: {0}".format(config.paths.TVTinfo))

            # If exists, carry on with this test
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
                    if (
                        (any([item.is_keyword(key) for item in params.all_kobjs]))
                        and (dataset[ind_dataset]["class_missions"][key]["papertype"] in params.allowed_classifications)
                    )
                }

                # Fetch results of test and combine with any errors for bibcode
                curr_test = dict_info[curr_key]["storage"].copy()
                if curr_key in dict_errors:
                    curr_test.update(dict_errors[curr_key])

                # Check each entry
                try:
                    ind_dataset = list_bibcodes_dataset.index(curr_key)
                    # Ensure same number of missions for this bibcode
                    self.assertEqual(len(curr_actuals), len(curr_test))
                    #
                    for curr_textid in curr_test:
                        curr_mission = curr_test[curr_textid]["mission"]
                        curr_class = curr_test[curr_textid]["class"]
                        # Check each mission and class
                        self.assertEqual(curr_class, mapper[curr_actuals[curr_mission]["papertype"].lower()])

                except (KeyError, AssertionError):
                    print("")
                    print(">")
                    print(
                        ("Diff mission info for {0}:\n{1}\n-\n" + "{2}\nand {3}\n-\nvs\n{4}\n---\n").format(
                            curr_key, curr_test, dict_info[curr_key], dict_errors[curr_key], curr_actuals
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()

            print("Run of test__TVT_directory complete.")
