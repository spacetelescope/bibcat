# !/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
:title: build_model.py

This module creates a new training ML model.

- It builds a model and trains the model using the partitioned and streamlined dataset.

- Once the model is trained, the `models` folder and its subdirectories for T/V/T
  are created along with various model related files.

:Run example: bibcat train
"""

import os
import time

from bibcat import config
from bibcat.core import operator
from bibcat.core import parameters as params
from bibcat.core.classifiers import ml
from bibcat.data.load_source_dataset import load_source_dataset
from bibcat.data.streamline_dataset import streamline_dataset
from bibcat.utils.logger_config import setup_logger

settings = config.dataprep

logger = setup_logger(__name__)


def build_model() -> None:
    # Fetch filepath for model
    name_model = config.output.name_model
    dir_data = os.path.join(config.paths.partitioned, name_model)
    dir_model = os.path.join(config.paths.models, name_model)

    logger.info(f"Starting buiding and train {dir_model}!")
    if os.path.exists(dir_model):
        logger.info(
            f"{dir_model} already exists. Change the model name in bibcat_config.yaml. if you want to build and train a new model"
        )

    # filepath to save processing errors
    filesave_error = os.path.join(dir_model, f"{name_model}_processing_errors.txt")

    # Initialize an empty ML classifier
    classifier_ML = ml.MachineLearningClassifier(verbose=True)

    # Initialize an Operator
    tabby_ML = operator.Operator(
        classifier=classifier_ML,
        name="ML",
        mode=config.textprocessing.mode_modif,
        keyword_objs=params.all_kobjs,
        verbose=True,
        load_check_truematch=config.textprocessing.do_verify_truematch,
        deep_verbose=False,
    )

    # load source dataset
    source_dataset = load_source_dataset(do_verbose=True)

    # streamline text dictionary
    dict_texts = streamline_dataset(
        source_dataset=source_dataset,
        operator_ML=tabby_ML,
        do_verbose_text_summary=config.textprocessing.do_verbose_text_summary,
    )

    # with open("bibcat/data/operational_data/fakedata.json", "w") as json_file:
    #    json.dump(dict_texts["97"], json_file, indent=4)

    # Use the Operator instance to train an ML model
    start = time.time()
    str_err = tabby_ML.train_model_ML(
        dir_model=dir_model,
        dir_data=dir_data,
        name_model=name_model,
        do_reuse_run=config.dataprep.do_reuse_run,
        do_check_truematch=config.textprocessing.do_verify_truematch,
        seed_TVT=settings.seed_TVT,
        dict_texts=dict_texts,
        mapper=config.pretrained.map_papertypes,  # For masking of classes (e.g., masking 'supermention' as 'mention')
        buffer=config.textprocessing.buffer,
        fraction_TVT=settings.fraction_TVT,
        mode_TVT=settings.mode_TVT,
        do_shuffle=config.textprocessing.do_shuffle,  # do_shuffle: Whether or not to shuffle contents of training vs. validation vs. testing datasets
    )

    logger.info(f"Time to train the model with run = {time.time() - start} seconds.")

    # Save the output error string to a file
    with open(filesave_error, "w") as openfile:
        openfile.write(str_err)
