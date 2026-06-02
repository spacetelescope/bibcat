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
from bibcat.data.build_dataset import build_dataset, load_source_dataset
from bibcat.llm.chunker import ChunkPlanner, SubmissionManager
from bibcat.llm.evaluate import evaluate_output
from bibcat.llm.llm_io import adjust_model, read_output
from bibcat.llm.metrics import evaluate_multiple_llm_runs, extract_eval_data_for_run
from bibcat.llm.openai import OpenAIHelper, classify_paper
from bibcat.llm.plots import confusion_matrix_plot, roc_plot
from bibcat.llm.roc import evaluate_multiple_llm_runs_with_roc, extract_roc_metrics_for_run
from bibcat.llm.run_eval import build_source_lookup
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import save_json_file

logger = setup_logger(__name__)


def _llm_output_dir() -> Path:
    """Return the model-specific llm output directory."""
    return Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}"


def _summary_output_path() -> Path:
    """Return the thresholded summary output path."""
    return _llm_output_dir() / f"{config.llms.eval_output_file}_t{config.llms.performance.threshold}.json"


def _cm_metrics_output_path(metrics_type: str) -> Path:
    """Return the confusion-matrix metrics output path."""
    return _llm_output_dir() / f"{config.llms.cm_metrics_file}_{metrics_type}_t{config.llms.performance.threshold}.json"


def _roc_metrics_output_path(metrics_type: str) -> Path:
    """Return the ROC metrics output path."""
    return (
        _llm_output_dir() / f"{config.llms.roc_metrics_file}_{metrics_type}_t{config.llms.performance.threshold}.json"
    )


def _prompt_output_path() -> Path:
    """Return the raw llm prompt output path."""
    return _llm_output_dir() / config.llms.prompt_output_file


def _read_bibcodes_file(filename) -> list[str]:
    """Read non-empty bibcodes from a CLI file option."""
    bibcodes = [line.strip() for line in filename.read().splitlines() if line.strip()]
    if not bibcodes:
        raise click.UsageError("Bibcode file is empty.")
    return bibcodes


def _read_required_metrics_json(path: Path, description: str) -> dict:
    """Read a required saved metrics JSON file or raise a CLI error."""
    if not path.exists():
        raise click.ClickException(f"{description} not found at {path}.")
    return read_output(filename=path)


@click.group("bibcat")
def cli() -> None:
    """Command-line tool for running the bibcat package
    To see more options for each command, you can use `--help` after each command.
    For instance, `bibcat llm run --help`
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
        logger.debug("CLI option: 'dataset' selected")
        build_dataset()


# LLM command group


@cli.group("llm", short_help="LLM-based paper classification")
def llmcli():
    """CLI for classifying papers using LLMs, specifically OpenAI models."""
    pass


@llmcli.command("run", help="Send a prompt to an OpenAI LLM model")
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
def run_gpt(filename, bibcode, index, model, num_runs, user_prompt_file, agent_prompt_file, verbose, ops):
    """Send a prompt to an OpenAI LLM model"""
    # override the config model
    start_time = time.time()
    logger.debug("CLI option: 'llm run' selected")
    if model:
        config.llms.openai.model = model
    # override the config user prompt file
    if user_prompt_file:
        config.llms.llm_user_prompt = user_prompt_file
    # override the config agent prompt file
    if agent_prompt_file:
        config.llms.llm_agent_prompt = agent_prompt_file
    # override the config ops flag
    if ops:
        config.llms.ops = ops

    classify_paper(file_path=filename, bibcode=bibcode, index=index, n_runs=num_runs, verbose=verbose)
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for run_gpt for {num_runs} papers: {elapsed_time} seconds.")


@llmcli.command("evaluate", help="Evaluate the LLM output")
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
@click.option(
    "-t",
    "--threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="The threshold value to accept the llm papertype",
)
@click.pass_context
def evaluate_llm(ctx, bibcode, index, model, file, submit, num_runs, write, threshold):
    """Evaluate the ouput JSON from a LLM model"""
    logger.debug("CLI option: 'llm evaluate' selected")
    # override the config model
    if model:
        config.llms.openai.model = model
    # override the config threshold
    if threshold:
        config.llms.performance.threshold = threshold
    # override the config output response file
    if file:
        config.llms.prompt_output_file = file

    # submit the paper for classification, if requested
    if submit:
        ctx.invoke(run_gpt, bibcode=bibcode, index=index, num_runs=num_runs)

    # evaluate the output
    evaluate_output(bibcode=bibcode, index=index, write_file=write)


@llmcli.command("plot", help="Create evaluation plots for llm performance")
@click.option(
    "-c",
    "--cm",
    is_flag=True,
    show_default=False,
    help="Create a confusion matrix plot. This flag works with the '-m' flag with a mission name, for example, 'bibcat llm plot -c -m JWST'",
)
@click.option(
    "-r",
    "--roc",
    is_flag=True,
    show_default=False,
    help="Create ROC curves. This flag works with the '-m' flag with a mission name, for example,'bibcat llm plot -r -m JWST'",
)
@click.option(
    "-m",
    "--missions",
    type=str,
    multiple=True,
    default=None,
    show_default=True,
    help="List mission names; this flag works with the '-c' flag, for instance, 'bibcat llm plot -c -m JWST -m HST -m TESS' ",
)
@click.option(
    "-a",
    "--all-missions",
    is_flag=True,
    show_default=False,
    help="Create plots for all missions, command example for a confusion matrix plot for all missions: 'bibcat llm plot -c -a'",
)
@click.option(
    "--run-index",
    default=0,
    type=click.IntRange(min=0),
    show_default=True,
    help="Run index to plot from saved single-run metrics JSON.",
)
def eval_plot(cm: bool, roc: bool, missions: str, all_missions: bool = False, run_index: int = 0):
    """Create the evaluation plots from a LLM model"""
    logger.debug("CLI option: 'llm plot' selected")

    requested_missions = config.missions if all_missions else list(missions)
    metrics_type = f"single_r{run_index}"

    if cm and requested_missions:
        metrics_data = _read_required_metrics_json(
            _cm_metrics_output_path(metrics_type),
            "Confusion matrix metrics file",
        )
        confusion_matrix_plot(metrics_data=metrics_data, missions=requested_missions, metrics_type=metrics_type)

    if roc and requested_missions:
        roc_data = _read_required_metrics_json(
            _roc_metrics_output_path(metrics_type),
            "ROC metrics file",
        )
        roc_plot(roc_data=roc_data, missions=requested_missions, metrics_type=metrics_type)


@llmcli.command("cm-metrics", help="Save Confusion Matrix metrics for llm performance")
@click.option("-a", "--aggregate", is_flag=True, show_default=False, help="Calculate aggregate metrics across missions")
@click.option(
    "-f",
    "--filename",
    required=True,
    type=click.File("r"),
    help="A file containing bibcodes to evaluate, one per line.",
)
@click.option(
    "-r",
    "--run-index",
    default=None,
    type=click.IntRange(min=0),
    show_default=True,
    help="Run index to evaluate in non-aggregate mode. Defaults to 0.",
)
@click.option(
    "-m",
    "--missions",
    type=str,
    multiple=True,
    default=None,
    show_default=True,
    help="List mission names; this flag works with the '-m' flag, for instance, 'bibcat llm cm-metrics -m JWST -m HST -m TESS'; if not provided, the metrics will be extracted for all missions by default.",
)
def cm_metrics(filename, run_index, missions: str, aggregate: bool):
    """Extract evaluation metrics from a LLM model and save to a JSON file"""
    logger.debug("CLI option: 'llm cm-metrics' selected")

    if aggregate and run_index is not None:
        raise click.UsageError("--run-index cannot be used with -a/--aggregate.")

    if missions:
        missions = list(missions)
    else:
        missions = config.missions

    llm_output_path = _prompt_output_path()
    bibcodes = _read_bibcodes_file(filename)
    llm_multi_runs_data = read_output(filename=llm_output_path)

    if aggregate:
        logger.info("Calculating aggregate metrics across mutiple runs.")
        metrics_type = "aggregate"
        source_lookup = build_source_lookup(load_source_dataset(do_verbose=False))
        metrics_data = evaluate_multiple_llm_runs(
            llm_runs_data=llm_multi_runs_data,
            missions=missions,
            bibcodes=bibcodes,
            source_lookup=source_lookup,
        )
        metrics_data_to_save = metrics_data

    else:
        selected_run_index = 0 if run_index is None else run_index
        logger.info(f"Calculating metrics for run index {selected_run_index}.")
        metrics_data_to_save = extract_eval_data_for_run(
            llm_runs_data=llm_multi_runs_data,
            missions=missions,
            run_index=selected_run_index,
            bibcodes=bibcodes,
        )
        metrics_type = f"single_r{selected_run_index}"

    output_path = _cm_metrics_output_path(metrics_type)

    save_json_file(
        path=output_path,
        dataset=metrics_data_to_save,
    )
    logger.info(f"Evaluation metrics saved to {output_path}")


@llmcli.command("roc-metrics", help="Save ROC metrics for llm performance")
@click.option(
    "-a", "--aggregate", is_flag=True, show_default=False, help="Calculate aggregate ROC metrics across missions"
)
@click.option(
    "-f",
    "--filename",
    required=True,
    type=click.File("r"),
    help="A file containing bibcodes to evaluate, one per line.",
)
@click.option(
    "-r",
    "--run-index",
    default=None,
    type=click.IntRange(min=0),
    show_default=True,
    help="Run index to evaluate in non-aggregate mode. Defaults to 0.",
)
@click.option(
    "-m",
    "--missions",
    type=str,
    multiple=True,
    default=None,
    show_default=True,
    help="List mission names; for instance, 'bibcat llm roc-metrics -m JWST -m HST -m TESS'; if not provided, metrics are extracted for all missions by default.",
)
def roc_metrics(filename, run_index, missions: str, aggregate: bool):
    """Extract ROC metrics from a LLM model and save to a JSON file"""
    logger.debug("CLI option: 'llm roc-metrics' selected")

    if aggregate and run_index is not None:
        raise click.UsageError("--run-index cannot be used with -a/--aggregate.")

    if missions:
        missions = [mission.upper() for mission in missions]
    else:
        missions = [mission.upper() for mission in config.missions]

    llm_output_path = _prompt_output_path()
    bibcodes = _read_bibcodes_file(filename)
    llm_multi_runs_data = read_output(filename=llm_output_path)

    if aggregate:
        logger.info("Calculating aggregate ROC metrics across multiple runs.")
        metrics_type = "aggregate"
        source_lookup = build_source_lookup(load_source_dataset(do_verbose=False))
        roc_data = evaluate_multiple_llm_runs_with_roc(
            llm_runs_data=llm_multi_runs_data,
            missions=missions,
            bibcodes=bibcodes,
            source_lookup=source_lookup,
        )
    else:
        selected_run_index = 0 if run_index is None else run_index
        logger.info(f"Calculating ROC metrics for run index {selected_run_index}.")
        metrics_type = f"single_r{selected_run_index}"
        roc_data = extract_roc_metrics_for_run(
            llm_runs_data=llm_multi_runs_data,
            missions=missions,
            run_index=selected_run_index,
            bibcodes=bibcodes,
        )

    output_path = _roc_metrics_output_path(metrics_type)

    save_json_file(
        path=output_path,
        dataset=roc_data,
    )
    logger.info(f"ROC metrics saved to {output_path}")


# Batch LLM command group


@llmcli.group("batch", short_help="Batch processing of papers with an LLM")
def llmbatch():
    """Batch run LLM commands"""
    pass


@llmbatch.command("run", help="Batch submit papers to an OpenAI LLM model.")
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
    """Batch submit papers to an OpenAI LLM model.

    Example Usage
    =============
        bibcat llm batch run -f /path/to/paper.pdf
    """
    start_time = time.time()
    logger.debug("CLI option: 'llm batch run' selected")
    # override the config model
    if model:
        config.llms.openai.model = model
        logger.debug(f"openai model = {model}")
    # override the config user prompt file
    if user_prompt_file:
        config.llms.llm_user_prompt = user_prompt_file
        logger.debug(f"user_prompt_file: {user_prompt_file}")
    # override the config agent prompt file
    if agent_prompt_file:
        config.llms.llm_agent_prompt = agent_prompt_file
        logger.debug(f"agent_prompt_file: {agent_prompt_file}")
    # override the config ops flag
    if ops:
        config.llms.ops = ops
        logger.info("Run in the OPS MODE!")

    # get the list of files
    files = files or filename.read().splitlines()
    if filename:
        logger.info(f"batch filename: {filename.name}")

    # iterate over the files
    for file in files:
        # check if file, bibcode, or index
        source = "file" if os.path.isfile(file) else "index" if file.isnumeric() else "bibcode"

        classify_paper(
            file_path=file if source == "file" else None,
            bibcode=file if source == "bibcode" else None,
            index=file if source == "index" else None,
            n_runs=num_runs,
            verbose=verbose,
        )
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for run_gpt_batch for {len(files)} papers: {elapsed_time} seconds.")


@llmbatch.command("evaluate", help="Batch evaluate the LLM output")
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
@click.option("-s", "--submit", is_flag=True, show_default=True, help="Flag to submit the paper for classification")
@click.option("-n", "--num_runs", default=1, type=int, show_default=True, help="The number of prompt runs to execute")
@click.pass_context
def evaluate_llm_batch(ctx, files, filename, model, submit, num_runs):
    """Batch evaluate a list of papers"""
    start_time = time.time()
    logger.debug("CLI option: 'llm batch evaluate' selected")

    # override the config model
    if model:
        config.llms.openai.model = model

    # get the list of files
    files = files or filename.read().splitlines()

    # submit the paper for classification, if requested
    if submit:
        ctx.invoke(run_gpt_batch, files=files, filename=filename, num_runs=num_runs)

    for file in files:
        # check if file, bibcode, or index
        source = "file" if os.path.isfile(file) else "index" if file.isnumeric() else "bibcode"

        evaluate_output(
            bibcode=file if source == "bibcode" else None, index=file if source == "index" else None, write_file=True
        )
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time for evaluate_llm_batch for {len(files)} papers: {elapsed_time} seconds.")


@llmbatch.command("submit", help="Submit a batch of papers using the OpenAI Batch API")
@click.option(
    "-p",
    "--filename",
    default=None,
    type=click.File("r"),
    show_default=True,
    help="The path to a file of bibcodes to read in",
)
@click.option(
    "-b",
    "--batch-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="The jsonl batch file for submission",
)
@click.option("-m", "--model", default=None, type=str, show_default=True, help="The model type to use")
@click.option("-v", "--verbose", is_flag=True, show_default=True, help="Set to print verbose output")
def submit(filename, batch_file, model, verbose):
    """Submit a batch of papers using the OpenAI Batch API"""
    # override the config model
    if model:
        orig = config.llms.openai.model
        config.llms.openai.model = model

        # update the existing batch file with the new model
        if batch_file:
            batch_file = adjust_model(batch_file, orig, model)

    # get the list of files
    bibcodes = filename.read().splitlines() if filename else None

    oa = OpenAIHelper(verbose=verbose)
    oa.submit_batch(bibcodes=bibcodes, batch_file=str(batch_file))


@llmbatch.command("retrieve", help="Retrieve a batch run from the OpenAI Batch API")
@click.option("-b", "--batchid", default=None, help="The ID of the batch to retrieve")
@click.option("-v", "--verbose", is_flag=True, show_default=True, help="Set to print verbose output")
def retrieve(batchid, verbose):
    """Retrieve a batch run from the OpenAI Batch API"""
    oa = OpenAIHelper(verbose=verbose)
    oa.retrieve_batch(batchid)


@llmbatch.command("process", help="Process a large batch of papers with the OpenAI Batch API")
@click.option(
    "-p",
    "--filename",
    default=None,
    type=click.File("r"),
    show_default=True,
    help="The path to a file of bibcodes to read in",
)
@click.option(
    "-b",
    "--batch-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="The jsonl batch file for submission",
)
@click.option("-m", "--model", default=None, type=str, show_default=True, help="The model type to use")
@click.option("-t", "--test", is_flag=True, show_default=True, help="Set to test a dry-run submission")
@click.option("-c", "--check", is_flag=True, show_default=True, help="Set to check batch statues")
@click.option(
    "-r", "--retrieve-batch", is_flag=True, show_default=True, help="Set to retrieve results of completed batches"
)
@click.option("-e", "--eval-batch", is_flag=True, show_default=True, help="Evaulate individual chunk results")
@click.option("-g", "--merge", is_flag=True, show_default=True, help="Merge chunks into final single output file")
def process(filename, batch_file, model, test, retrieve_batch, check, eval_batch, merge):  # noqa: C901
    """Process a batch of papers using the OpenAI Batch API

    Process a large batch of papers, with proper file chunking, for
    submission to the OpenAI Batch API.  Handles chunking of large files
    to account for the OpenAI API limites.  Manage daily submissions,
    check batch status, and retrieve results.

    """
    # override the config model
    if model:
        orig = config.llms.openai.model
        config.llms.openai.model = model

        # update the existing batch file with the new model
        if batch_file:
            batch_file = adjust_model(batch_file, orig, model)

    # get the bibcodes
    if filename:
        bibcodes = filename.read().splitlines() if filename else None
        oa = OpenAIHelper()
        batch_file = oa.create_batch_file(bibcodes)

    planner = ChunkPlanner(batch_file)
    if not planner.has_been_planned:
        planner.prepare_all()
    sm = SubmissionManager(planner)

    if not sm.all_batches_submitted and not check and not retrieve_batch and not merge and not eval_batch:
        click.echo(f"{sm.remaining_chunks} chunks remaining. Submitting next batch.")
        click.echo(sm.submit_batch(dry_run=test))
    elif sm.all_batches_submitted:
        click.echo("All batches already submitted.")

    if check:
        click.echo(sm.check_batches_status())
        return

    if retrieve_batch:
        click.echo("Retrieving completed batch results.")
        click.echo(sm.retrieve_batch_results())

    if eval_batch:
        click.echo("Evaluating batch results.")
        click.echo(sm.evaluate_batch_results())

    completed = sm.all_batches_completed
    if merge and not completed:
        click.echo("All batches must be completed before merging.")
        return
    elif merge and completed:
        click.echo("Merging prompt and evaluation chunks.")
        sm.merge_outputs(kind="llm")
        sm.merge_outputs(kind="eval")


if __name__ == "__main__":
    cli()
