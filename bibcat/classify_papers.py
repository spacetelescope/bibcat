"""
:title: classify_papers.py

This module fetches test input data for classification, and produces
performance results such as a confusion matrix (if ML method) and result numpy save files.
It also can classify streamlined JSON paper text(s) with a given classfier.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.inputs.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Classfication data: this text data is used for prediction (classification),
  for now, it is fetched from the `bibcat/data/partitioned_datasets/model_name/dir_test`
  folder via `config.output.folders_TVT["test"]` below. However, we will need to modify the
  codebase to  set up a designated folder of operational papers.

- Run example: python classify_papers.py
"""

import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.core import performance
from bibcat.core.classifiers import ml, rules
from bibcat.core.classifiers.textdata import ClassifierBase
from bibcat.evaluate_basic_performance import generate_performance_evaluation_output
from bibcat.fetch_papers import fetch_papers
from bibcat.operate_classifier import operate_classifier

# Fetch filepath for model
name_model = config.output.name_model
dir_model = os.path.join(config.paths.models, name_model)
dir_output = os.path.join(config.paths.output, name_model)
os.makedirs(dir_output, exist_ok=True)


# Set directories for fetching test text

# `partitioned_datasets/name_model/` folder
dir_datasets = os.path.join(config.paths.partitioned, name_model)
dir_test = config.output.folders_TVT[
    "test"
]  # the directory name "dir_test" in the partitioned_datasets/name_model/ folder # can be an CLI option


# Set parameters for each operator and its internal classifier

# do_raise_innererror: if True, will stop if exception encountered;
# if False, will print error and continue
do_raise_innererror = config.textprocessing.do_raise_innererror  # CLI option?
do_verbose_text_summary = config.textprocessing.do_verbose_text_summary  # print input text data summary ; CLI option?


# Set some overarching global variables

# Random seed for shuffling text dataset
np.random.seed(config.textprocessing.shuffle_seed)
# Whether or not to shuffle the text dataset
do_shuffle = config.textprocessing.do_shuffle

# do_real_testdata: If True, will use real papers to test performance;
# if False, will use fake texts but we will implement the fake data
# if we need. For now, we keep this variable and only the real text.
do_real_testdata = config.textprocessing.do_real_testdata  # can an CLI option?

# Number of text entries to test the performance for; None for all tests
max_tests = config.textprocessing.max_tests
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = config.textprocessing.mode_modif
# Prepare some Keyword objects
lookup = config.textprocessing.lookup


# threshold: Uncertainty threshold
threshold = config.textprocessing.threshold
# buffer: the number of sentences to include within paragraph around each sentence with target terms
buffer = config.textprocessing.buffer
# For uncertainty test
threshold_array = np.linspace(0.5, 0.95, 20)

# Fetching real JSON paper text
dict_texts = fetch_papers(
    dir_datasets=dir_datasets,
    dir_test=dir_test,
    do_shuffle=do_shuffle,
    do_verbose_text_summary=do_verbose_text_summary,
    max_tests=max_tests,
)

# We will choose which operator/method to classify the papers and evaluate performance below.

# The classifier_name could eventually be chosen in a user run setting or as a CLI option
# in the future refactoring.
classifier: ClassifierBase
classifier_name = "ML"  # CLI option

# Rule-Based Classifier
classifier_RB = rules.RuleBasedClassifier(
    which_classifs=None, do_verbose=True, do_verbose_deep=False
)


# ML model file location
filepath_model = os.path.join(dir_model, (name_model + ".npy"))
fileloc_ML = os.path.join(dir_model, (config.output.tfoutput_prefix + name_model))

# initialize classifiers
# Machine-Learning Classifier
classifier_ML = ml.MachineLearningClassifier(
    filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True
)


if classifier_name == "ML":
    classifier = classifier_ML
elif classifier_name == "RB":
    classifier = classifier_RB
else:
    raise ValueError(
        "An invalid value! Choose either 'ML' for the machine learning classifier or 'RB' for the rule-based classifier!"
    )

# Performance tests: This can be an CLI option.
# Create an instance of the Performance class
performer = performance.Performance()

# Parameters for this evaluation
# Root name of the file within which to store the performance evaluation output
fileroot_evaluation = f"test_eval_basic_{classifier_name}"
# Root name of the file within which to store misclassified text information
fileroot_misclassif = f"test_misclassif_basic_{classifier_name}"
# Root name of the file within which to store classified result information
fileroot_class_results = f"classification_results_{classifier_name}"

figsize = (20, 12)

# Run the pipeline for a basic evaluation and an evaluation of model performance as a function of uncertainty
generate_performance_evaluation_output(
    classifier_name=classifier_name,
    classifier=classifier,
    dict_texts=dict_texts,
    is_text_processed=False,
    mapper=params.map_papertypes,
    keyword_objs=params.all_kobjs,
    mode_modif=mode_modif,
    buffer=buffer,
    threshold=threshold,
    threshold_array=threshold_array,
    print_freq=1,
    filepath_output=dir_output,
    fileroot_evaluation=fileroot_evaluation,
    fileroot_misclassif=fileroot_misclassif,
    fileroot_confusion_matrix_plot=f"performance_confmatr_basic_{classifier_name}.png",
    fileroot_uncertainty_plot=f"performance_grid_uncertainty_{classifier_name}.png",
    figsize=figsize,
    load_check_truematch=True,
    do_save_evaluation=True,
    do_save_misclassif=True,
    do_raise_innererror=False,
    do_verbose=True,
    do_verbose_deep=False,
)


# Operation: classifying paper(s)
# Currently it pulls the papers from in test folder (`bibcat/data/partitioned_datasets/model_name/dir_test`)
# but we will have to change a new text feed directory for operation.

operate_classifier(
    classifier_name=classifier_name,
    classifier=classifier,
    dict_texts=dict_texts,
    keyword_objs=params.all_kobjs,
    mode_modif=mode_modif,
    buffer=buffer,
    threshold=threshold,
    print_freq=25,
    filepath_output=dir_output,
    fileroot_class_results=fileroot_class_results,
    is_text_processed=False,
    load_check_truematch=True,
    do_verbose=True,
    do_verbose_deep=False,
    do_raise_innererror=False,
)
