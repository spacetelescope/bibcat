# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Main entry point into bibcat
"""

import os
import time
from pathlib import Path

import click

from bibcat import config
from bibcat.build_model import build_model
from bibcat.classify_papers import classify_papers
from bibcat.data.build_dataset import build_dataset
from bibcat.evaluate_basic_performance import evaluate_basic_performance
from bibcat.llm.evaluate import evaluate_output
from bibcat.llm.openai import OpenAIHelper, classify_paper
from bibcat.llm.plots import confusion_matrix_plot, roc_plot
from bibcat.stats.stats_llm import save_evaluation_stats
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


@click.group("bibcat")
def cli() -> None:
    """Command-line tool for running the bibcat package
    To see more options for each command, you can use `--help` after each command.
    For instance, `bibcat run-gpt --help`
    """


@cli.command(help="Build a combined dataset")
def dataset() -> None:
    """build a combined dataset from the papertrack data and the ADS fulltext data

    Wraps the original build_dataset script.
    """

    def file_exists(filelist: list) -> bool:
        "Check if any file exists among the list of files"
        return any([os.path.isfile(item) for item in filelist])

    file_list = [
        config.inputs.path_source_data,
        config.output.path_not_in_papertext,
        config.output.path_not_in_papertrack,
        config.output.path_papertext_not_in_papertrack,
    ]

    if file_exists(file_list):
        logger.warning(
            "One or more files in the following list already exist and will not be overwritten."
            + f"\nPlease change the save destination for these file(s) or move the existing files.\n{file_list}"
        )
        return

    else:
        logger.info("CLI option: 'dataset' selected")
        build_dataset()


@cli.command(help="Build or train a classical NLP ML model")
@click.option("-l", "--library", default="tensorflow", type=str, show_default=True, help="The model library to use")
@click.option("-m", "--model", default="bert", type=str, show_default=True, help="The model type to use")
@click.option(
    "-n", "--name", default=None, type=str, show_default=True, help="The name of the model training run to use"
)
@click.option(
    "-k",
    "--key",
    default=None,
    type=str,
    show_default=True,
    help="The model key to use in the in preprocess/encoder mapping",
)
@click.option("-p", "--preprocessor", default=None, type=str, show_default=True, help="The model preprocessor to use")
@click.option("-e", "--encoder", default=None, type=str, show_default=True, help="The model encoder to use")
def train(library, model, name, key, preprocessor, encoder) -> None:
    """Build and train a classical ML model

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
        config.ml.ML_model_key = f"custom_{config.ml.ML_model_type}_key"
        config.ml[model]["dict_ml_model_encoders"][config.ml.ML_model_key] = encoder
        config.ml[model]["dict_ml_model_preprocessors"][config.ml.ML_model_key] = preprocessor

    build_model()


@cli.command(help="Classify a paper using a trained model")
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
    logger.info("Selected ML classification!")
    classify_papers(classifier_name=name)


@cli.command(help="Update the training dataset JSON file")
def update() -> None:
    pass


@cli.command(help="Evaluate a trained model on efficacy and performance")
@click.option(
    "-n",
    "--name",
    default="ML",
    type=click.Choice(["ML", "RB"]),  # TODO: delete "RB"
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
@click.option(
    "-b",
    "--bibcode",
    default=None,
    type=str,
    show_default=True,
    help="A bibcode from the papertrack source combined_dataset",
)
@click.option(
    "-i",
    "--index",
    default=None,
    type=str,
    show_default=True,
    help="An array index from the papertrack source combined_dataset",
)
@click.option("-m", "--model", default=None, type=str, show_default=True, help="The model type to use")
@click.option("-n", "--num_runs", default=1, type=int, show_default=True, help="The number of prompt runs to execute")
@click.option(
    "--assistant", is_flag=True, show_default=True, default=False, help="Set to use the file-search assistant"
)
@click.option(
    "-u", "--user-prompt-file", default=None, type=str, show_default=True, help="The name of a custom user prompt file"
)
@click.option(
    "-a",
    "--agent-prompt-file",
    default=None,
    type=str,
    show_default=True,
    help="The name of a custom agent prompt file",
)
@click.option("-v", "--verbose", is_flag=True, show_default=True, help="Set to print verbose output")
@click.option("-o", "--ops", is_flag=True, show_default=False, help="Set to operational classification mode")
def run_gpt(filename, bibcode, index, model, num_runs, assistant, user_prompt_file, agent_prompt_file, verbose, ops):
    """Send a prompt to an OpenAI LLM model"""
    # override the config model
    start_time = time.time()
    logger.info("CLI option: 'run_gpt' selected")
    if model:
        config.llms.openai.model = model
    # override the config user prompt file
    if user_prompt_file:
        config.llms.llm_user_prompt = user_prompt_file
    # override the config agent prompt file
    if agent_prompt_file:
        config.llms.llm_agent_prompt = agent_prompt_file

    classify_paper(
        file_path=filename,
        bibcode=bibcode,
        index=index,
        n_runs=num_runs,
        use_assistant=assistant,
        verbose=verbose,
        ops=ops,
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for run_gpt for {num_runs} papers: {elapsed_time} seconds.")


@cli.command(help="Batch submit papers to an OpenAI LLM model")
@click.option(
    "-f",
    "--files",
    default=None,
    type=str,
    show_default=True,
    multiple=True,
    help="A list of files or bibcodes to upload",
)
@click.option(
    "-p",
    "--filename",
    default=None,
    type=click.File("r"),
    show_default=True,
    help="The path to a file of bibcodes or papers to read in",
)
@click.option("-m", "--model", default=None, type=str, show_default=True, help="The model type to use")
@click.option(
    "-u", "--user-prompt-file", default=None, type=str, show_default=True, help="The name of a custom user prompt file"
)
@click.option(
    "-a",
    "--agent-prompt-file",
    default=None,
    type=str,
    show_default=True,
    help="The name of a custom agent prompt file",
)
@click.option("-v", "--verbose", is_flag=True, show_default=True, help="Set to print verbose output")
@click.option("-o", "--ops", is_flag=True, show_default=False, help="Set to operational classification mode")
@click.option("-n", "--num_runs", default=1, type=int, show_default=True, help="The number of prompt runs to execute")
def run_gpt_batch(files, filename, model, user_prompt_file, agent_prompt_file, verbose, num_runs, ops):
    start_time = time.time()
    logger.info("CLI option: 'run_gpt_batch' selected")
    # override the config model
    if model:
        config.llms.openai.model = model
    # override the config user prompt file
    if user_prompt_file:
        config.llms.llm_user_prompt = user_prompt_file
    # override the config agent prompt file
    if agent_prompt_file:
        config.llms.llm_agent_prompt = agent_prompt_file
    if ops:
        logger.info("Run in the OPS MODE!")

    # get the list of files
    files = files or filename.read().splitlines()

    # iterate over the files
    for file in files:
        # check if file, bibcode, or index
        source = "file" if os.path.isfile(file) else "index" if file.isnumeric() else "bibcode"

        classify_paper(
            file_path=file if source == "file" else None,
            bibcode=file if source == "bibcode" else None,
            index=file if source == "index" else None,
            n_runs=num_runs,
            use_assistant=True if source == "file" else False,
            verbose=verbose,
            ops=ops,
        )
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for run_gpt_batch for {len(files)} papers: {elapsed_time} seconds.")


@cli.command(help="Evaluate the LLM output")
@click.option(
    "-b",
    "--bibcode",
    default=None,
    type=str,
    show_default=True,
    help="A bibcode from the papertrack source combined_dataset",
)
@click.option(
    "-i",
    "--index",
    default=None,
    type=str,
    show_default=True,
    help="An array index from the papertrack source combined_dataset",
)
@click.option("-m", "--model", default=None, type=str, show_default=True, help="The model type to use")
@click.option(
    "-f",
    "--file",
    default=None,
    type=str,
    show_default=True,
    help="The name of the output response file to use for evaluation",
)
@click.option("-s", "--submit", is_flag=True, show_default=True, help="Flag to submit the paper for classification")
@click.option(
    "-n",
    "--num_runs",
    default=1,
    type=int,
    show_default=True,
    help="The number of prompt runs to execute for classification",
)
@click.option(
    "-w/-now",
    "--write/--no-write",
    default=True,
    is_flag=True,
    show_default=True,
    help="Flag to write the output evaluation file",
)
@click.pass_context
def evaluate_llm(ctx, bibcode, index, model, file, submit, num_runs, write):
    """Evaluate the ouput JSON from a LLM model"""
    # override the config model
    if model:
        config.llms.openai.model = model
    # override the config output response file
    if file:
        config.llms.prompt_output_file = file

    # submit the paper for classification, if requested
    if submit:
        ctx.invoke(run_gpt, bibcode=bibcode, index=index, num_runs=num_runs)

    # evaluate the output
    evaluate_output(bibcode=bibcode, index=index, write_file=write)


@cli.command(help="Create evaulation plots for llm performance")
@click.option(
    "-c",
    "--cm",
    is_flag=True,
    show_default=False,
    help="Create a confusion matrix plot. This flag works with the '-m' flag with a mission name, for example, 'bibcat eval-plot -c -m JWST'",
)
@click.option(
    "-r",
    "--roc",
    is_flag=True,
    show_default=False,
    help="Create ROC curves. This flag works with the '-m' flag with a mission name, for example, 'bibcat eval-plot -r -m JWST'",
)
@click.option(
    "-m",
    "--missions",
    type=str,
    multiple=True,
    default=None,
    show_default=True,
    help="List mission names; this flag works with the '-c' flag, for instance, 'bibcat -c -m JWST -m HST -m TESS' ",
)
@click.option(
    "-a",
    "--all-missions",
    is_flag=True,
    show_default=False,
    help="Create plots for all missions, command example for a confusion matrix plot for all missions: 'bibcat eval-plot -c -a'",
)
def eval_plot(cm: bool, roc: bool, missions: str, all_missions: bool = False):
    """Create the evaluation plots from a LLM model"""
    if cm and all_missions:
        missions = config.missions
        confusion_matrix_plot(missions=missions)

    elif cm and missions:
        confusion_matrix_plot(missions=list(missions))

    if roc and all_missions:
        missions = config.missions
        roc_plot(missions=missions)

    elif roc and missions:
        roc_plot(missions=list(missions))


@cli.command(help="Create a statisics table for classification")
@click.option(
    "-e",
    "--eval",
    is_flag=True,
    show_default=False,
    help="Create a statistics table for llm and human mission-papertype pairs. This flag works with the '-e' flag. e.g., 'bibcat statistics -e'",
)
def statistics(llm: bool):
    if llm:
        filepath = Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_stats_file}"
        save_evaluation_stats(filepath)


@cli.group("openai", short_help="OpenAI LLM commands")
def oacli():
    """General OpenAI LLM commands"""
    pass


@oacli.command(
    "create_assistant",
    short_help="Create a new OpenAI Assistant",
    help="Create a new OpenAI Assistant.  See bibcat.llm.openai.create_assistant for more information.",
)
@click.option("-n", "--name", default=None, type=str, show_default=True, help="The name of the assistant")
@click.option(
    "-i", "--vectorid", default=None, type=str, show_default=True, help="The id of the vector database to attach"
)
def create_oa_assistant(name, vectorid):
    """Create a new assistant

    Creates a new OpenAI assistant with file search capabilities.  The llm model
    to use for the assistant is set in the config file by ``config.llms.openai.model``.
    Custom instructions and behavior for the assistant is set through a custom agent prompt file,
    or a config value, otherwise the default agent instructions will be used.
    See ``bibcat.llm.io.get_llm_prompt`` for more information.

    """
    oa = OpenAIHelper()
    asst = oa.create_assistant(name=name, vs_id=vectorid)
    click.echo(f"Assistant created: {asst.name} - {asst.id}")


@oacli.command("list_assistants", help="List all OpenAI Assistants")
def list_oa_assistants():
    """List all assistants you have created"""
    oa = OpenAIHelper()
    assts = oa.list_assistants()
    for i in assts:
        click.echo(f"Assistant: {i['name']} - {i['id']}")


if __name__ == "__main__":
    cli()
    cli()
