"""
:title: evaluate_basic_performance.py


This module performs the basic evaluation performance results such as a confusion matrix
(if ML method) and result numpy save files.

- Context: the input full text JSON file (papertrack + ADS full texts) is
  called via config.inputs.path_source_data configured in bibcat/config.py and is used for
  training, validating, and testing the trained model.

- Run example: `bibcat evaluate` or `bibcat -n ML`

This module employs

    1) performance.evaluate_performance_basic() to generates a file of dictionary of the basic performance evaluation information and a plot of a confusion matrix after the classification of input test texts conducted based on a trained ML model or the rule-based model. It script also produces a list of the mis-classified papers.

    2) performance.evaluate_performance_uncertainty() to generate uncertainty estimates.

All the output are saved in the output folder.

- Basic: generate confusion matrices for the set of Operators
         (containing the different classifiers).
- Uncertainty: plot performance as a function of uncertainty level
               for the set of Operators.


"""

import os

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from bibcat import config
from bibcat import parameters as params
from bibcat.core import operator, performance
from bibcat.core.classifiers import ml
from bibcat.core.classifiers.textdata import ClassifierBase
from bibcat.fetch_papers import fetch_papers


def evaluate_basic_performance(classifier_name: str = "ML") -> None:
    """Evaluate performance

    Evaluate basic performance using machine-learning or rule-based classifiers.

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

    is_text_processed = False

    # Random seed for shuffling text dataset before fetching
    np.random.seed(config.textprocessing.shuffle_seed)
    # Fetching real JSON paper text
    dict_texts = fetch_papers(
        do_evaluation=True,
        do_shuffle=config.textprocessing.do_shuffle,
        do_verbose_text_summary=config.textprocessing.do_verbose_text_summary,
        max_texts=config.textprocessing.max_tests,  # Number of text entries to test the performance for; None for all tests
    )

    # We will choose which operator/method to evaluate performance below.
    classifier: ClassifierBase

    # initialize classifiers
    # Machine-Learning Classifier
    classifier_ML = ml.MachineLearningClassifier(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

    # CLI option
    if classifier_name == "ML":
        classifier = classifier_ML
    else:
        raise ValueError(
            "An invalid value! Choose 'ML' for the machine learning classifier!"
        )

    # Initialize operators by loading models into instances of the Operator class
    op = operator.Operator(
        classifier=classifier,
        name=classifier_name,
        mode=config.textprocessing.mode_modif,
        keyword_objs=params.all_kobjs,
        load_check_truematch=config.textprocessing.do_verify_truematch,
        do_verbose=True,
        do_verbose_deep=False,
    )

    # Create an instance of the Performance class
    performer = performance.Performance()

    # Run the pipeline for a basic evaluation of model performance
    # Multiple operators can work as list in args.
    performer.evaluate_performance_basic(
        operators=[op],
        dicts_texts=[dict_texts],
        mappers=[params.map_papertypes],
        thresholds=[config.performance.threshold],
        buffers=[config.textprocessing.buffer],
        is_text_processed=is_text_processed,
        filepath_output=dir_output,
        filename_plot=config.performance.fileroot_confusion_matrix_plot + f"{classifier_name}.png",
        fileroot_evaluation=config.performance.fileroot_evaluation + f"{classifier_name}",
        fileroot_misclassif=config.performance.fileroot_misclassif + f"{classifier_name}",
        figsize=config.performance.figsize,
        print_freq=config.performance.print_freq,
        do_verbose=True,
        do_verbose_deep=False,
        do_raise_innererror=config.textprocessing.do_raise_innererror,
        do_save_evaluation=True,
        do_save_misclassif=True,
        do_check_truematch=config.textprocessing.do_verify_truematch,
    )

    # Run the pipeline for an evaluation of model performance
    # as a function of uncertainty
    performer.evaluate_performance_uncertainty(
        operators=[op],
        dicts_texts=[dict_texts],
        mappers=[params.map_papertypes],
        threshold_arrays=[np.linspace(*config.performance.threshold_array)],
        buffers=[config.textprocessing.buffer],
        is_text_processed=is_text_processed,
        filepath_output=dir_output,
        filename_plot=config.performance.fileroot_uncertainty_plot + f"{classifier_name}.png",
        fileroot_evaluation=config.performance.fileroot_evaluation + f"{classifier_name}",
        fileroot_misclassif=config.performance.fileroot_misclassif + f"{classifier_name}",
        figsize=config.performance.figsize,
        print_freq=config.performance.print_freq,
        do_check_truematch=config.textprocessing.do_verify_truematch,
        do_raise_innererror=config.textprocessing.do_raise_innererror,
        do_save_evaluation=True,
        do_save_misclassif=True,
        do_verbose=True,
        do_verbose_deep=True,
    )
