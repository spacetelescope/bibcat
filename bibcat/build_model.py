"""
:title: build_model.py

This module creates a new training ML model.

- It builds a model and trains the model using the partitioned and streamlined dataset.

- Once the model is trained, the `models` folder and its subdirectories for T/V/T
  are created along with various model related files.

:Run example: python build_model.py
"""

import os
import time

from bibcat import config
from bibcat import model_settings as settings
from bibcat import parameters as params
from bibcat.core import operator
from bibcat.core.classifiers import ml
from bibcat.data.streamline_dataset import load_source_dataset, streamline_dataset

# Fetch filepath for model
name_model = config.name_model
dir_data = os.path.join(config.path_partitioned_data, name_model)
dir_model = os.path.join(config.PATH_MODELS, name_model)


def build_model() -> None:
    if os.path.exists(dir_model):
        print(
            f"{dir_model} already exists. Change the model name in config.py. if you want to build and train a new model"
        )

    # do_check_truematch: the codebase that have unknown ambiguous phrases, then a note will be
    # printed and those papers will not be used for training-validation-testing.
    # Add the identified ambiguous phrase to the external ambiguous phrase
    # database and rerun to include those papers.
    do_check_truematch = True

    # print out the text info summary per paper.
    do_verbose_text_summary = True

    # For masking of classes (e.g., masking 'supermention' as 'mention')
    mapper = params.map_papertypes

    # filepath to save processing errors
    filesave_error = os.path.join(dir_model, f"{name_model}_processing_errors.txt")

    # do_reuse_run: Whether or not to reuse any existing output from previous
    # training+validation+testing (TVT) runs
    do_reuse_run = True
    # do_shuffle: Whether or not to shuffle contents of training vs. validation
    # vs. testing datasets
    do_shuffle = True

    # Initialize an empty ML classifier
    classifier_ML = ml.MachineLearningClassifier(filepath_model=None, fileloc_ML=None, do_verbose=True)

    # Initialize an Operator
    tabby_ML = operator.Operator(
        classifier=classifier_ML,
        mode=settings.mode_modif,
        keyword_objs=params.all_kobjs,
        do_verbose=True,
        load_check_truematch=do_check_truematch,
        do_verbose_deep=False,
    )
    # load source dataset
    source_dataset = load_source_dataset(do_verbose=True)

    # streamline text dictionary
    dict_texts = streamline_dataset(
        source_dataset=source_dataset, operator_ML=tabby_ML, do_verbose_text_summary=do_verbose_text_summary
    )

    # Use the Operator instance to train an ML model
    start = time.time()
    str_err = tabby_ML.train_model_ML(
        dir_model=dir_model,
        dir_data=dir_data,
        name_model=name_model,
        do_reuse_run=do_reuse_run,
        do_check_truematch=do_check_truematch,
        seed_ML=settings.seed_ML,
        seed_TVT=settings.seed_TVT,
        dict_texts=dict_texts,
        mapper=mapper,
        buffer=settings.buffer,
        fraction_TVT=settings.fraction_TVT,
        mode_TVT=settings.mode_TVT,
        do_shuffle=do_shuffle,
        do_verbose=True,
        do_verbose_deep=False,
    )

    print(f"Time to train the model with run = {time.time()-start} seconds.")

    # Save the output error string to a file
    with open(filesave_error, "w") as openfile:
        openfile.write(str_err)


# This section checks if the script is the main program
if __name__ == "__main__":
    # Code here will only execute if the script is run directly, not if it's imported as a module
    print("The script is running as a standalone script! \n Building and Training ML model!")
    build_model()
