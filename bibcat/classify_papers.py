"""
:title: classify_papers.py

This module fetches test input data for classification and classify streamlined JSON paper text(s) with a given classfier.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.inputs.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Classfication data: this text data is used for prediction (classification),
  for now, it is fetched from the `bibcat/data/partitioned_datasets/model_name/dir_test`
  folder via `config.output.folders_TVT["test"]` below. However, we will need to modify the
  codebase to  set up a designated folder of operational papers.

- Run example: bibcat classify
"""

import os

import numpy as np

from bibcat import config
from bibcat import parameters as params
from bibcat.core.classifiers import ml, rules
from bibcat.core.classifiers.textdata import ClassifierBase
from bibcat.fetch_papers import fetch_papers
from bibcat.operate_classifier import operate_classifier


def classify_papers(classifier_name: str = "ML") -> None:
    """Classify papers

    Classify papers using machine-learning or rule-based classifiers.

    Parameters
    ----------
    classifier_name : str, optional
        the type of classifier to use, by default "ML"

    Raises
    ------
    ValueError
        when an invalid classifier name is provided
    """

    # Fetch filepath for model
    name_model = config.output.name_model
    dir_model = os.path.join(config.paths.models, name_model)
    filepath_model = os.path.join(dir_model, (name_model + ".npy"))
    fileloc_ML = os.path.join(dir_model, (config.output.tfoutput_prefix + name_model))

    # Fetch filepath for output or create the directory if not exists.
    dir_output = os.path.join(config.paths.output, name_model)
    os.makedirs(dir_output, exist_ok=True)

    # Set directories for fetching test text

    # `partitioned_datasets/name_model/` folder
    dir_datasets = os.path.join(config.paths.partitioned, name_model)
    dir_test = config.output.folders_TVT[
        "test"
    ]  # the directory name "dir_test" in the partitioned_datasets/name_model/ folder # can be an CLI option

    # do_real_testdata: If True, will use real papers to test performance;
    # if False, will use fake texts but we will implement the fake data
    # if we need. For now, we keep this variable and only the real text.
    do_real_testdata = config.textprocessing.do_real_testdata  # can an CLI option?

    # Random seed for shuffling text dataset
    np.random.seed(config.textprocessing.shuffle_seed)

    # Fetching real JSON paper text
    dict_texts = fetch_papers(
        dir_datasets=dir_datasets,
        dir_test=dir_test,
        do_shuffle=config.textprocessing.do_shuffle,
        do_verbose_text_summary=config.textprocessing.do_verbose_text_summary,
        max_tests=config.textprocessing.max_tests,
    )

    # We will choose which operator/method to classify the papers and evaluate performance below.

    # The classifier_name will be selected as a CLI option: "ML" or "RB" or something else
    classifier: ClassifierBase
    # initialize classifiers

    # Rule-Based Classifier
    classifier_RB = rules.RuleBasedClassifier(which_classifs=None, do_verbose=True, do_verbose_deep=False)

    # Machine-Learning Classifier
    classifier_ML = ml.MachineLearningClassifier(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

    # CLI option
    if classifier_name == "ML":
        classifier = classifier_ML
    elif classifier_name == "RB":
        classifier = classifier_RB
    else:
        raise ValueError(
            "An invalid value! Choose either 'ML' for the machine learning classifier or 'RB' for the rule-based classifier!"
        )

    # Operation: classifying paper(s)
    # Currently it pulls the papers from in test folder (`bibcat/data/partitioned_datasets/model_name/dir_test`)
    # but we will have to change a new text feed directory for operation.

    # if text_format == "ascii":
    #     ops_text
    # elif text_format == "json":
    #     ops_texts = fetch_papers(
    #         dir_datasets=dir_datasets,
    #         dir_test=dir_test,
    #         do_shuffle=do_shuffle,
    #         do_verbose_text_summary=do_verbose_text_summary,
    #         max_tests=max_tests,
    #     )
    # else:
    #     raise ValueError("An invalid file format! Prepare for your texts either in ascii or JSON format!")

    operate_classifier(
        classifier_name=classifier_name,
        classifier=classifier,
        dict_texts=dict_texts,
        keyword_objs=params.all_kobjs,
        mode_modif=config.textprocessing.mode_modif,
        buffer=config.textprocessing.buffer,
        threshold=config.textprocessing.threshold,
        print_freq=25,
        filepath_output=dir_output,
        fileroot_class_results=config.results.fileroot_class_results + f"{classifier_name}",
        is_text_processed=False,
        load_check_truematch=True,
        do_verbose=True,
        do_verbose_deep=False,
        do_raise_innererror=False,
    )
