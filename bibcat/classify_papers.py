"""
:title: classify_papers.py

This module fetches test input data for classification, classifies the texts
into three categories; science, mention, data-influenced. Finally, it produces
performance results such as a confusion matrix.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Classfication data: this text data is used for prediction (classification),
  for now, it is fetched from the `bibcat/model/dir_test` folder via
  the `folder_test` variable below. However, we will need to modify the
  codebase to  set up a designated folder for operational papers.

- Run example: python classify_papers.py
"""

import json
import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.core import operator, performance
from bibcat.core.classifiers import ml, rules


# Fetch filepath for model
name_model = config.output.name_model
dir_model = os.path.join(config.paths.partitioned, name_model)

# Set directories for fetching test text
dir_info = dir_model
folder_test = config.output.folders_TVT["test"]
dir_test = os.path.join(dir_model, folder_test)


# Set parameters for each operator and its internal classifier

# Global parameters
do_verify_truematch = config.classify.do_verify_truematch  # used in Performance class

# do_raise_innererror: if True, will stop if exception encountered;
# if False, will print error and continue
do_raise_innererror = config.classify.do_raise_innererror
do_verbose_text_summary = config.classify.do_verbose_text_summary  # print input text data summary

# For uncertainty test
list_threshold_arrays = [np.linspace(0.5, 0.95, 20)] * 2
# Mapper for class types; None for no mapper
class_mapper = params.map_papertypes

# Set some overarching global variables

# Random seed for shuffling text dataset
np.random.seed(config.classify.shuffle_seed)
# Whether or not to shuffle the text dataset
do_shuffle = config.classify.do_shuffle

# do_real_testdata: If True, will use real papers to test performance;
# if False, will use fake texts but we will implement the fake data
# if we need. For now, we keep this variable and only the real text.
do_real_testdata = config.classify.do_real_testdata

# Number of text entries to test the performance for; None for all tests
max_tests = config.classify.max_tests
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = config.classify.mode_modif
# Prepare some Keyword objects
all_kobjs = params.all_kobjs
lookup = config.classify.lookup


# Set two operators: machine learning (ML) and rule-based (RB)
# - buffer: the number of sentences to include within paragraph around
#           each sentence with target terms
# - threshold: Uncertainty threshold
# - mapper : Mapper to mask classifications; None if no masking

# For operator ML
mapper_ML = class_mapper
threshold_ML = config.classify.threshold_ML
buffer_ML = config.classify.buffer_ML

# For operator RL
mapper_RB = class_mapper
threshold_RB = config.classify.threshold_RB
buffer_RB = config.classify.buffer_RB

# Gather parameters into lists
list_mappers = [mapper_ML, mapper_RB]
list_thresholds = [threshold_ML, threshold_RB]
list_buffers = [buffer_ML, buffer_RB]

# perpare papers to test on
# For use of real papers from test dataset to test on
if do_real_testdata:
    # Load information for processed bibcodes reserved for testing
    dict_TVTinfo = np.load(os.path.join(dir_info, "dict_TVTinfo.npy"), allow_pickle=True).item()
    list_test_bibcodes = [key for key in dict_TVTinfo if (dict_TVTinfo[key]["folder_TVT"] == folder_test)]

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
        curr_info = {"text": curr_data["body"], "id": str(curr_ind), "bibcode": curr_data["bibcode"], "missions": {}}
        # Iterate through missions for this paper
        for curr_mission in curr_data["class_missions"]:
            # Iterate through declared Keyword objects
            for curr_kobj in all_kobjs:
                curr_name = curr_kobj.get_name()
                # Store mission data under keyword name, if applicable
                if curr_kobj.is_keyword(curr_mission):
                    curr_info["missions"][curr_name] = {
                        "mission": curr_name,
                        "class": curr_data["class_missions"][curr_mission]["papertype"],
                    }
                # Otherwise, store that this mission was not detected for this text
                else:
                    curr_info["missions"][curr_name] = {"mission": curr_name, "class": config.ml.verdict_rejection}
        # Store this data entry
        dict_texts[str(curr_ind)] = curr_info

    # Print some notes about the testing data
    print(f"Number of texts in text set: {dict_texts}")
    print("")
    if do_verbose_text_summary:
        for key in dict_texts:
            print(f"Entry: {key}")
            print(f"ID: {dict_texts[key]['id']}")
            print(f"Bibcode: {dict_texts[key]['bibcode']}")
            print(f"Missions: {dict_texts[key]['missions']}")
            print(f"Start of text:\n{dict_texts[key]['text'][0:500]}")
            print("-\n")

# Store texts for each operator and its internal classifier
# For operator ML, Dictionary of texts to classify
dict_texts_ML = dict_texts
# For operator RL, Dictionary of texts to classify
dict_texts_RB = dict_texts
# Gather into list
list_dict_texts = [dict_texts_ML, dict_texts_RB]

# Create a list of classifiers
# This can be modified to use whatever classifiers you'd like.
# initialize classifiers by loading a previously trained ML model
filepath_model = os.path.join(dir_model, (name_model + ".npy"))
fileloc_ML = os.path.join(dir_model, (config.output.tfoutput_prefix + name_model))
classifier_ML = ml.MachineLearningClassifier(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

# Load a rule-based classifier
classifier_RB = rules.RuleBasedClassifier()


# Initialize operators by loading models into instances of the Operator class

# The machine learning operator
operator_ML = operator.Operator(
    classifier=classifier_ML,
    mode=mode_modif,
    keyword_objs=all_kobjs,
    name="Operator_ML",
    do_verbose=True,
    load_check_truematch=True,
    do_verbose_deep=False,
)
# The rule-based operator
operator_RB = operator.Operator(
    classifier=classifier_RB,
    name="Operator_RB",
    mode=mode_modif,
    keyword_objs=all_kobjs,
    do_verbose=True,
    do_verbose_deep=False,
)
# Feel free to add more/less operators here.
list_operators = [operator_ML, operator_RB]


# Run the operators
# The machine learning operator
results_ML = operator_ML.classify(
    text=dict_texts_ML,
    lookup=lookup,
    buffer=buffer_ML,
    modif=mode_modif,
    threshold=threshold_ML,
    do_raise_innererror=False,
    do_check_truematch=True,
    do_verbose_deep=False,
)

# The rule-based operator
results_RB = operator_RB.classify(
    text=dict_texts_RB,
    lookup=lookup,
    buffer=buffer_RB,
    modif=mode_modif,
    threshold=threshold_RB,
    do_raise_innererror=False,
    do_check_truematch=True,
    do_verbose_deep=False,
)

# Print the results
# Print the text and classes as a reminder
# print("Text:\n\"\n{0}\n\"\n".format(dict_texts))
print(f"Lookup: {lookup}\n")

# The machine learning results
print("> Machine learning results:")
print(f'Paragraph:\n"\n{results_ML["modif"]}\n"')
print(f"Verdict: {results_ML['verdict']}")
print(f"Probabilities: {results_ML['uncertainty']}")
print("-\n")

# The rule-based results
print("> Rule-based results:")
print(f'Paragraph:\n"\n{results_RB["modif"]}\n"')
print(f"Verdict: {results_RB['verdict']}")
print(f"Probabilities: {results_RB['uncertainty']}")
print("-\n")


# Performance tests:
# - Basic: We generate confusion matrices for the set of Operators
#          (containing the different classifiers).
# - Uncertainty: We plot performance as a function of uncertainty level
#                for the set of Operators.

# Create an instance of the Performance class
performer = performance.Performance()

# Run the pipeline for a basic evaluation of model performance
performer.evaluate_performance_basic(
    operators=list_operators,
    dicts_texts=list_dict_texts,
    mappers=list_mappers,
    thresholds=list_thresholds,
    buffers=list_buffers,
    is_text_processed=False,
    do_verify_truematch=do_verify_truematch,
    do_raise_innererror=do_raise_innererror,
    do_save_evaluation=True,
    do_save_misclassif=True,
    filepath_output=config.paths.output,
    fileroot_evaluation="test_eval_basic",
    fileroot_misclassif="test_misclassif_basic",
    print_freq=1,
    do_verbose=True,
    do_verbose_deep=False,
    figsize=(20, 12),
)

# Run the pipeline for an evaluation of model performance
# as a function of uncertainty
performer.evaluate_performance_uncertainty(
    operators=list_operators,
    dicts_texts=list_dict_texts,
    mappers=list_mappers,
    threshold_arrays=list_threshold_arrays,
    buffers=list_buffers,
    is_text_processed=False,
    do_verify_truematch=do_verify_truematch,
    do_raise_innererror=do_raise_innererror,
    do_save_evaluation=True,
    do_save_misclassif=True,
    filepath_output=config.paths.output,
    fileroot_evaluation="test_eval_uncertainty",
    fileroot_misclassif="test_misclassif_uncertainty",
    print_freq=25,
    do_verbose=True,
    do_verbose_deep=False,
    figsize=(40, 12),
)
