"""
:title: operate_classifier.py

This module applies a given classifier method on the input data for classification.

- Context: once the input dict texts of papers returned from fetch_papers.py are fed into this script, this will classify the input text(s) into the paper classification.

"""

import json
import os

from bibcat.core import operator
from bibcat.utils.utils import convert_sets

# Store texts for each operator and its internal classifier
# For operator ML, Dictionary of texts to classify


def operate_classifier(
    classifier_name: str,
    classifier: object,
    dict_texts: dict,  # this dict is very complex so a proper type annotation can be determined later
    keyword_objs: list,
    mode_modif: str,
    buffer: int,
    threshold: float,
    print_freq: int,
    filepath_output: str,
    fileroot_class_results: str,
    is_text_processed: bool = False,
    load_check_truematch: bool = True,
    do_verbose: bool = True,
    do_verbose_deep: bool = False,
    do_raise_innererror: bool = False,
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
        do_verbose=do_verbose,
        do_verbose_deep=do_verbose_deep,
    )

    # Print some notes
    if do_verbose:
        print(f"Classifying with Operator {classifier_name}...")

    # Load in as either raw or preprocessed data
    if is_text_processed:  # If given text preprocessed
        curr_texts = None
        curr_modifs = [dict_texts[key]["text"] for key in dict_texts]
        curr_forests = [dict_texts[key]["forest"] for key in dict_texts]
    else:  # If raw text given, text needs to be preprocessed
        # curr_texts = [dict_texts[keys[jj]]["text"] for jj in range(0, len(keys))]
        curr_texts = [dict_texts[key]["text"] for key in dict_texts]
        curr_modifs = None
        curr_forests = None

    # Classify texts with current operator
    results = op.classify_set(
        texts=curr_texts,
        modifs=curr_modifs,
        forests=curr_forests,
        threshold=threshold,
        buffer=buffer,
        do_check_truematch=load_check_truematch,
        do_raise_innererror=do_raise_innererror,
        print_freq=print_freq,
        do_verbose=do_verbose,
        do_verbose_deep=do_verbose_deep,
    )

    # Print some notes
    if do_verbose:
        print(f"Classification complete for Operator {classifier_name}...")
        print("Generating the performance counter...")

    # We want to prepend the text ID and its bibcode to the results.
    text_ids = [{"id": dict_texts[index]["id"]} for index in dict_texts]
    text_bibcodes = [{"bibcode": dict_texts[index]["bibcode"]} for index in dict_texts]

    if do_verbose:
        print(f"{classifier_name} results: \n")
        for index, dict_content in enumerate(results):
            print(f"\nText {index+1}: \n")
            print(f"Text id: {text_ids[index].get('id')}")
            print(f"Text bibcode: {text_bibcodes[index].get('bibcode')}")
            for key in dict_content:
                print(f"Mission: {key}")
                print(f"Verdict: {dict_content[key]['verdict']}")
                print(f"Probability: {dict_content[key]['uncertainty']}")
                print(f"Supporting texts:\n'\n{dict_content[key]['modif']}\n' ")

    # Save the classification results into a JSON file
    classification_results = [
        {**text_ids[index], **text_bibcodes[index], **dict_content} for index, dict_content in enumerate(results)
    ]

    # set file save location and filename
    tmp_filepath = os.path.join(filepath_output, f"{fileroot_class_results}.json")
    if do_verbose:
        print(f"Saving '{fileroot_class_results}.json' under '{filepath_output}/'!")
    json_dump = convert_sets(classification_results)  # converts any sets in the results to lists
    with open(tmp_filepath, "w") as f:
        json.dump(json_dump, f)
