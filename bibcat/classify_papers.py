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

from bibcat import config
from bibcat import parameters as params
from bibcat.core.classifiers import ml
from bibcat.core.classifiers.textdata import ClassifierBase
from bibcat.fetch_papers import fetch_papers
from bibcat.operate_classifier import operate_classifier
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


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

    # Fetch filepath for output or create the directory if not exists.
    name_model = config.output.name_model
    dir_output = os.path.join(config.paths.output, name_model)
    os.makedirs(dir_output, exist_ok=True)

    # Fetching JSON paper text
    dict_texts = fetch_papers(do_evaluation=False)

    # We will choose which operator/method to classify the papers and evaluate performance below.
    # The classifier_name will be selected as a CLI option: "ML" or "RB" or something else
    classifier: ClassifierBase
    # initialize classifiers

    # Machine-Learning Classifier
    classifier_ML = ml.MachineLearningClassifier(load=True, verbose=True)

    # CLI option
    if classifier_name == "ML":
        classifier = classifier_ML
    else:
        raise ValueError("An invalid value! Choose 'ML' for the machine learning classifier!")

    operate_classifier(
        classifier_name=classifier_name,
        classifier=classifier,
        dict_texts=dict_texts,
        keyword_objs=params.all_kobjs,
        mode_modif=config.textprocessing.mode_modif,
        buffer=config.textprocessing.buffer,
        print_freq=25,
        filepath_output=dir_output,
        fileroot_class_results=config.results.fileroot_class_results + f"{classifier_name}",
        is_text_processed=False,
        load_check_truematch=True,
        verbose=True,
        deep_verbose=False,
    )
