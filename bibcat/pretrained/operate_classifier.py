"""
:title: operate_classifier.py

This module applies a given classifier method on the input data for classification.

- Context: once the input dict texts of papers returned from fetch_papers.py are fed into this script, this will classify the input text(s) into the paper classification.

"""

import json
import os
import pathlib

from bibcat.core import operator
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import convert_sets

logger = setup_logger(__name__)

# Store texts for each operator and its internal classifier
# For operator ML, Dictionary of texts to classify


# TODO - do we need this?
def operate_classifier(
    classifier_name: str,
    classifier: object,
    dict_texts: dict,
    keyword_objs: list,
    buffer: int,
    print_freq: int,
    filepath_output: str,
    fileroot_class_results: str,
    mode_modif: str,
    is_text_processed: bool = False,
    load_check_truematch: bool = True,
    verbose: bool = True,
    deep_verbose: bool = False,
):
    """
    Method: operate_classifiers
    Purpose:
    - Classify a set of texts based on a given classifier
    Arguments:
    - classifier_name: the name of the classifier such as 'ML' for a machine learning classifier or 'RB' for a rule based classifier.
    - classifier: instance object of an classifier
    - dict_texts: a nested dictionary of dictionary of input texts to classify
    - keyword_obj [<Keyword object> or None]: Target Keyword instance. If None, the input variable lookup will be used to look up the Keyword instance.
    - buffer [int (default=0)]: Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
    - threshold:
    - print_freq: print some notes at given frequency
    - is_text_processed: if a given text is preprocessed ('True') or raw ('False')
    - do_check_truematch [bool]: Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
    - lookup [str or None]: A term for looking up the target Keyword instance (e.g. 'HST'). Required if keyobj is None.
    - do_verbose [bool (default=False)]: Whether or not to print surface-level log information and tests.
    - do_verbose_deep [bool (default=False)]: Whether or not to print inner log information and tests.
    """

    # Initialize operators by loading models into instances of the Operator class
    op = operator.Operator(
        classifier=classifier,
        name=classifier_name,
        mode=mode_modif,
        keyword_objs=keyword_objs,
        load_check_truematch=load_check_truematch,
        verbose=verbose,
        deep_verbose=deep_verbose,
    )

    # Print some notes
    if verbose:
        logger.info(f"Classifying with Operator {classifier_name}...")

    # Load in as either raw or preprocessed data
    if is_text_processed:  # If given text preprocessed
        curr_texts = None
        curr_modifs = [dict_texts[key]["text"] for key in dict_texts]
        # curr_forests = [dict_texts[key]["forest"] for key in dict_texts]
    else:  # If raw text given, text needs to be preprocessed
        # curr_texts = [dict_texts[keys[jj]]["text"] for jj in range(0, len(keys))]
        curr_texts = [dict_texts[key]["text"] for key in dict_texts]
        curr_modifs = None
        # curr_forests = None

    # Classify texts with current operator
    results = op.classify_set(
        curr_texts, modifs=curr_modifs, buffer=buffer, do_check_truematch=load_check_truematch, print_freq=print_freq
    )

    # Print some notes
    if verbose:
        logger.info(f"Classification complete for Operator {classifier_name}...")
        logger.info("Generating the performance counter...")

    # We want to prepend the text ID and its bibcode to the results.
    text_ids = [{"id": index} for index in dict_texts]
    text_bibcodes = [{"bibcode": dict_texts[index]["bibcode"]} for index in dict_texts]

    if verbose:
        logger.info(f"{classifier_name} results: \n")
        for index, dict_content in enumerate(results):
            logger.info(f"\nText {index + 1}: \n")
            logger.info(f"Text id: {text_ids[index].get('id')}")
            logger.info(f"Text bibcode: {text_bibcodes[index].get('bibcode')}")
            for key in dict_content:
                logger.info(f"Mission: {key}")
                logger.info(f"Verdict: {dict_content[key]['verdict']}")
                logger.info(f"Probability: {dict_content[key]['uncertainty']}")
                logger.info(f"Supporting texts:\n'\n{dict_content[key]['modif']}\n' ")

    # Save the classification results into a JSON file
    classification_results = [
        {**text_ids[index], **text_bibcodes[index], **dict_content} for index, dict_content in enumerate(results)
    ]

    # set file save location and filename
    tmp_filepath = os.path.join(filepath_output, f"{fileroot_class_results}.json")

    if not pathlib.Path(tmp_filepath).exists():
        pathlib.Path(tmp_filepath).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"Saving '{fileroot_class_results}.json' under '{filepath_output}/'!")
    json_dump = convert_sets(classification_results)  # converts any sets in the results to lists
    with open(tmp_filepath, "w") as f:
        json.dump(json_dump, f, indent=4)
