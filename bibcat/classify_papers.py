"""
:title: classify_papers.py

This module fetches test input data for classification, classifies the texts
into three categories; science, mention, data-influenced. Finally, it produces
performance results such as a confusion matrix.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Classfication data: this text data is used for prediction (classification),
  for now, it is fetched from the `bibcat/data/partitioned_datasets/model_name/dir_test`
  folder via `config.folders_TVT["test"]` below. However, we will need to modify the
  codebase to  set up a designated folder for operational papers.

- Run example: python classify_papers.py
"""

import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.core import operator, performance
from bibcat.core.classifiers import ml, rules
from bibcat.core.classifiers.textdata import ClassifierBase
from bibcat.fetch_papers import fetch_papers
from bibcat.operate_classifiers import operate_classifiers

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
do_verbose_text_summary = True  # print input text data summary


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
max_tests = 2
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = "anon"
# Prepare some Keyword objects
lookup = "HST"


# Set two operators: machine learning (ML) and rule-based (RB)
# - buffer: the number of sentences to include within paragraph around
#           each sentence with target terms
# - threshold: Uncertainty threshold
threshold = 0.70
buffer = 0

# For operator ML
# mapper_ML = params.map_papertypes
threshold_ML = 0.70
buffer_ML = 0

# For operator RL
# mapper_RB = params.map_papertypes
threshold_RB = 0.70
buffer_RB = 0

if do_real_testdata:
    # Fetching text blurbs to classify
    dicts_texts = fetch_papers(
        do_real_testdata=do_real_testdata,
        do_shuffle=do_shuffle,
        do_verbose_text_summary=do_verbose_text_summary,
        max_tests=max_tests,
    )
# Store texts for each operator and its internal classifier
# For operator ML, Dictionary of texts to classify
dicts_texts_ML = dicts_texts

# This can be modified to use whatever classifiers you'd like.
# initialize classifiers


filepath_model = os.path.join(dir_model, (name_model + ".npy"))
fileloc_ML = os.path.join(dir_model, (config.tfoutput_prefix + name_model))

# Machine-Learning Classifier
classifier_ML = ml.MachineLearningClassifier(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

# Rule-Based Classifier
classifier_RB = rules.RuleBasedClassifier(which_classifs=None, do_verbose=True, do_verbose_deep=False)

classifier: ClassifierBase
classifier_name = "RB"

if classifier_name == "ML":
    classifier = classifier_ML
elif classifier_name == "RB":
    classifier = classifier_RB
else:
    raise ValueError(
        "Annvalid value! Choose either 'ML' for the machine learning classifier or 'RB' for the rule-based classifier!"
    )

operate_classifiers(
    classifier_name=classifier_name,
    classifier=classifier,
    dicts_texts=dicts_texts,
    keyword_objs=params.all_kobjs,
    mode_modif=mode_modif,
    buffer=buffer,
    threshold=threshold,
    print_freq=25,
    is_text_processed=False,
    load_check_truematch=True,
    do_verbose=True,
    do_verbose_deep=False,
    do_raise_innererror=False,
)


# Performance tests:
# - Basic: We generate confusion matrices for the set of Operators
#          (containing the different classifiers).
# - Uncertainty: We plot performance as a function of uncertainty level
#                for the set of Operators.

# Gather parameters into lists

# #=========

# list_mappers = [
#     params.map_papertypes,
#     params.map_papertypes,
# ]  # - mapper : Mapper to mask classifications; None if no masking
# list_thresholds = [threshold_ML, threshold_RB]
# list_buffers = [buffer_ML, buffer_RB]

# # For uncertainty test
# list_threshold_arrays = [np.linspace(0.5, 0.95, 20)] * 2

# # Store texts for each operator and its internal classifier
# # Gather into list
# list_dicts_texts = [dicts_texts_ML, dicts_texts_RB]

# # Feel free to add more/less operators here.
# list_operators = [operator_ML, operator_RB]


# # Create an instance of the Performance class
# performer = performance.Performance()

# # Parameters for this evaluation
# # Root name of the file within which to store the performance evaluation output
# fileroot_evaluation = "test_eval_basic"
# # Root name of the file within which to store misclassified text information
# fileroot_misclassif = "test_misclassif_basic"
# figsize = (20, 12)

# # Run the pipeline for a basic evaluation of model performance
# performer.evaluate_performance_basic(
#     operators=list_operators,
#     dicts_texts=list_dicts_texts,
#     mappers=list_mappers,
#     thresholds=list_thresholds,
#     buffers=list_buffers,
#     is_text_processed=False,
#     do_verify_truematch=do_verify_truematch,
#     do_raise_innererror=do_raise_innererror,
#     do_save_evaluation=True,
#     do_save_misclassif=True,
#     filepath_output=config.PATH_OUTPUT,
#     fileroot_evaluation=fileroot_evaluation,
#     fileroot_misclassif=fileroot_misclassif,
#     print_freq=1,
#     do_verbose=True,
#     do_verbose_deep=False,
#     figsize=figsize,
# )

# # Parameters for this evaluation
# # Root name of the file within which to store the performance evaluation output
# fileroot_evaluation = "test_eval_uncertainty"
# # Root name of the file within which to store misclassified text information
# fileroot_misclassif = "test_misclassif_uncertainty"
# figsize = (40, 12)

# # Run the pipeline for an evaluation of model performance
# # as a function of uncertainty
# performer.evaluate_performance_uncertainty(
#     operators=list_operators,
#     dicts_texts=list_dicts_texts,
#     mappers=list_mappers,
#     threshold_arrays=list_threshold_arrays,
#     buffers=list_buffers,
#     is_text_processed=False,
#     do_verify_truematch=do_verify_truematch,
#     do_raise_innererror=do_raise_innererror,
#     do_save_evaluation=True,
#     do_save_misclassif=True,
#     filepath_output=config.PATH_OUTPUT,
#     fileroot_evaluation=fileroot_evaluation,
#     fileroot_misclassif=fileroot_misclassif,
#     print_freq=25,
#     do_verbose=True,
#     do_verbose_deep=False,
#     figsize=figsize,
# )
