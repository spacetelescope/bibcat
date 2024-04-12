"""
:title: classify_papers.py

This module fetches test input data for classification, classifies the texts
into three categories; science, mention, data-influenced. Finally, it produces
performance results such as a confusion matrix.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Classfication data: this text data is used for prediction (classification),
  for now, it is fetched from the `bibcat/data/partitioned_datasets/model_name/dir_test` folder via `config.folders_TVT["test"]` below. However, we will need to modify the
  codebase to  set up a designated folder for operational papers.

- Run example: python classify_papers.py
"""

import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.core import operator, performance
from bibcat.core.classifiers import ml, rules
from bibcat.fetch_papers import fetch_papers

# Fetch filepath for model
name_model = config.name_model
dir_model = os.path.join(config.PATH_MODELS, name_model)

# Set directories for fetching test text
dir_datasets = os.path.join(config.path_partitioned_data, name_model)
dir_test = os.path.join(dir_datasets, config.folders_TVT["test"])

# Set parameters for each operator and its internal classifier

# Global parameters
do_verify_truematch = True  # used in Performance class

# do_raise_innererror: if True, will stop if exception encountered;
# if False, will print error and continue
do_raise_innererror = False
do_verbose_text_summary = False  # print input text data summary

# For uncertainty test
list_threshold_arrays = [np.linspace(0.5, 0.95, 20)] * 2


# Set some overarching global variables

# Random seed for shuffling text dataset
np.random.seed(10)
# Whether or not to shuffle the text dataset
do_shuffle = True

# do_real_testdata: If True, will use real papers to test performance;
# if False, will use fake texts but we will implement the fake data
# if we need. For now, we keep this variable and only the real text.
do_real_testdata = True

# Number of text entries to test the performance for; None for all tests
max_tests = 30
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = "anon"
# Prepare some Keyword objects
lookup = "HST"


# Set two operators: machine learning (ML) and rule-based (RB)
# - buffer: the number of sentences to include within paragraph around
#           each sentence with target terms
# - threshold: Uncertainty threshold
# - mapper : Mapper to mask classifications; None if no masking

# For operator ML
mapper_ML = params.map_papertypes
threshold_ML = 0.70
buffer_ML = 0

# For operator RL
mapper_RB = params.map_papertypes
threshold_RB = 0.70
buffer_RB = 0

# Gather parameters into lists
list_mappers = [mapper_ML, mapper_RB]
list_thresholds = [threshold_ML, threshold_RB]
list_buffers = [buffer_ML, buffer_RB]

if do_real_testdata:
    # Fetching text blurbs to classify
    dict_texts = fetch_papers(
        do_real_testdata=do_real_testdata,
        do_shuffle=do_shuffle,
        do_verbose_text_summary=do_verbose_text_summary,
        max_tests=max_tests,
    )
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
fileloc_ML = os.path.join(dir_model, (config.tfoutput_prefix + name_model))
classifier_ML = ml.MachineLearningClassifier(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

# Load a rule-based classifier
classifier_RB = rules.RuleBasedClassifier()


# Initialize operators by loading models into instances of the Operator class

# The machine learning operator
operator_ML = operator.Operator(
    classifier=classifier_ML,
    mode=mode_modif,
    keyword_objs=params.all_kobjs,
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
    keyword_objs=params.all_kobjs,
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

# Parameters for this evaluation
# Root name of the file within which to store the performance evaluation output
fileroot_evaluation = "test_eval_basic"
# Root name of the file within which to store misclassified text information
fileroot_misclassif = "test_misclassif_basic"
figsize = (20, 12)

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
    filepath_output=config.PATH_OUTPUT,
    fileroot_evaluation=fileroot_evaluation,
    fileroot_misclassif=fileroot_misclassif,
    print_freq=1,
    do_verbose=True,
    do_verbose_deep=False,
    figsize=figsize,
)

# Parameters for this evaluation
# Root name of the file within which to store the performance evaluation output
fileroot_evaluation = "test_eval_uncertainty"
# Root name of the file within which to store misclassified text information
fileroot_misclassif = "test_misclassif_uncertainty"
figsize = (40, 12)

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
    filepath_output=config.PATH_OUTPUT,
    fileroot_evaluation=fileroot_evaluation,
    fileroot_misclassif=fileroot_misclassif,
    print_freq=25,
    do_verbose=True,
    do_verbose_deep=False,
    figsize=figsize,
)
