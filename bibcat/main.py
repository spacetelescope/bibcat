# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Main entry point into bibcat
"""

import click

from bibcat import config
from bibcat.build_model import build_model
from bibcat.classify_papers import classify_papers
from bibcat.data.build_dataset import build_dataset
from bibcat.evaluate_basic_performance import evaluate_basic_performance

from bibcat.llm.openai import run


@click.group("bibcat")
def cli():
    """Command-line tool for running the bibcat package"""


@cli.command(help="build a combined dataset ")
def dataset():
    """build a combined dataset from the papertrack data and the ADS fulltext data

    Wraps the original build_dataset script.
    """
    build_dataset()


@cli.command(help="Build or train a classical NLP ML model")
@click.option("-l", "--library", default="tensorflow", type=str, show_default=True, help="The model library to use")
@click.option("-m", "--model", default="bert", type=str, show_default=True, help="The model type to use")
@click.option("-n", "--name", default=None, type=str, show_default=True, help="The name of the model training run to use")
@click.option("-k", "--key", default=None, type=str, show_default=True, help="The model key to use in the in preprocess/encoder mapping")
@click.option("-p", "--preprocessor", default=None, type=str, show_default=True, help="The model preprocessor to use")
@click.option("-e", "--encoder", default=None, type=str, show_default=True, help="The model encoder to use")
def train(library, model, name, key, preprocessor, encoder):
    """ Build and train a classical ML model

    Wraps the original build_model script. CLI inputs are used to override the user or default configuration settings.
    Alternatively, just edit your user configuration file directly.
    """
    # override the config inputs
    config.ml.ML_library = library
    config.ml.ML_model_type = model
    config.output.name_model = name or config.output.name_model
    config.ml.ML_model_key = key or config.ml.ML_model_key

    # raise error if model not found in config
    if model not in config.ml:
        raise KeyError(f"Model type {model} not found in config.ml")

    # hack the preprocessors and encoders into the config
    if preprocessor or encoder:
        config.ml.ML_model_key = f'custom_{config.ml.ML_model_type}_key'
        config.ml[model]['dict_ml_model_encoders'][config.ml.ML_model_key] = encoder
        config.ml[model]['dict_ml_model_preprocessors'][config.ml.ML_model_key] = preprocessor

    build_model()


@cli.command(help="fine-tune a LLM model")
def finetune() -> None:
    pass


@cli.command(help="classify a paper using a trained model")
@click.option(
    "-n",
    "--name",
    default="ML",
    type=click.Choice(["ML", "RB"]),
    show_default=True,
    help="The type of classifier to use.  Either machine-learning (ML) or rule-based (RB).",
)
def classify(name) -> None:
    """Classify a paper using a trained model

    Wraps the original classify_papers script.
    """
    classify_papers(classifier_name=name)


@cli.command(help="update the training dataset JSON file")
def update() -> None:
    pass


@cli.command(help="evaluate a trained model on efficacy and performance")
@click.option(
    "-n",
    "--name",
    default="ML",
    type=click.Choice(["ML", "RB"]),
    show_default=True,
    help="The type of classifier to use.  Either machine-learning (ML) or rule-based (RB).",
)
def evaluate(name) -> None:
    """Evaluate a trained model on efficacy and performance

    Wraps the original evaluate_basic_performance script.
    """
    evaluate_basic_performance(classifier_name=name)


@cli.command(help="run the GPT-4o model")
@click.option("-f", "--filename", default="", type=str, show_default=True, help="The path to a file to upload")
@click.option("-m", "--model", default="gpt-4o", type=str, show_default=True, help="The model type to use")
@click.option("-n", "--num_runs", default=1, type=int, show_default=True, help="The name of the model training run to use")
def run_gpt(filename, model, num_runs):
    run(filename, run=num_runs)


if __name__ == "__main__":
    cli()
