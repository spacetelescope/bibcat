# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Main entry point into bibcat
"""

import os
import click

from bibcat import config
from bibcat.build_model import build_model
from bibcat.classify_papers import classify_papers
from bibcat.data.build_dataset import build_dataset
from bibcat.evaluate_basic_performance import evaluate_basic_performance
from bibcat.llm.openai import send_prompt, OpenAIHelper


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


@cli.command(help="Send a prompt to an OpenAI LLM model")
@click.option("-f", "--filename", default=None, type=str, show_default=True, help="The path to a file to upload")
@click.option("-b", "--bibcode", default=None, type=str, show_default=True, help="A bibcode from the papertrack source combined_dataset")
@click.option("-i", "--index", default=None, type=str, show_default=True, help="An array index from the papertrack source combined_dataset")
@click.option("-m", "--model", default="gpt-4o-mini", type=str, show_default=True, help="The model type to use")
@click.option("-n", "--num_runs", default=1, type=int, show_default=True, help="The number of prompt runs to execute")
@click.option("--assistant", is_flag=True, show_default=True, default=False, help="Set to use the file-search assistant")
def run_gpt(filename, bibcode, index, model, num_runs, assistant):
    """ Send a prompt to an OpenAI LLM model """
    # override the config model
    if model:
        config.llms.openai.model = model

    send_prompt(file_path=filename, bibcode=bibcode, index=index, n_runs=num_runs, use_assistant=assistant)


@cli.command(help="Batch submit papers to an OpenAI LLM model")
@click.option("-f", "--files", default=None, type=str, show_default=True, multiple=True, help="A list of files or bibcodes to upload")
@click.option("-p", "--filename", default=None, type=click.File('rb'), show_default=True, help="The path to a file of bibcodes or papers to read in")
def batch_submit(files, filename):
    # get the list of files
    files = files or filename.read().splitlines()

    # iterate over the files
    for file in files:
        # check if file, bibcode, or index
        source = 'file' if os.path.isfile(file) else 'index' if file.isnumeric() else 'bibcode'

        send_prompt(file_path=file if source == 'file' else None,
                    bibcode=file if source == 'bibcode' else None,
                    index=file if source == 'index' else None,
                    n_runs=1, use_assistant=True if source == 'file' else False)


@cli.group('openai', short_help='OpenAI LLM commands')
def oacli():
    """ General OpenAI LLM commands """
    pass


@oacli.command('create_assistant', short_help="Create a new OpenAI Assistant",
               help="Create a new OpenAI Assistant.  See bibcat.llm.openai.create_assistant for more information.")
@click.option("-n", "--name", default=None, type=str, show_default=True, help="The name of the assistant")
@click.option("-i", "--vectorid", default=None, type=str, show_default=True, help="The id of the vector database to attach")
def create_oa_assistant(name, vectorid):
    """ Create a new assistant

    Creates a new OpenAI assistant with file search capabilities.  The llm model
    to use for the assistant is set in the config file by ``config.llms.openai.model``.
    Custom instructions and behavior for the assistant is set through a custom agent prompt file,
    or a config value, otherwise the default agent instructions will be used.
    See ``bibcat.llm.io.get_llm_prompt`` for more information.

    """
    oa = OpenAIHelper()
    asst = oa.create_assistant(name=name, vs_id=vectorid)
    click.echo(f'Assistant created: {asst.name} - {asst.id}')


@oacli.command('list_assistants', help="List all OpenAI Assistants")
def list_oa_assistants():
    """ List all assistants you have created """
    oa = OpenAIHelper()
    assts = oa.list_assistants()
    for i in assts:
        click.echo(f"Assistant: {i['name']} - {i['id']}")


if __name__ == "__main__":
    cli()
