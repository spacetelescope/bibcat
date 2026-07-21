import json

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pylint: disable=all
from click.testing import CliRunner

import bibcat.main as main
from bibcat.main import cli


def test_bibcat_cli() -> None:
    """test the top level bibcat cli"""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Command-line tool for running the bibcat package" in result.output
    assert "LLM-based paper classification" in result.output


def test_dataset() -> None:
    """test the cli dataset help command"""

    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "--help"])
    assert "Build a combined dataset" in result.output


def test_llm() -> None:
    """test the cli llm help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "--help"])
    assert "CLI for classifying papers using LLMs, specifically OpenAI models." in result.output
    assert "Batch processing of papers with an LLM" in result.output


def test_evaluate_llm() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "evaluate", "--help"])
    assert "Evaluate the LLM output" in result.output


def test_evaluate_llm_batch() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "batch", "evaluate", "--help"])
    assert "Batch evaluate the LLM output" in result.output


def test_run_gpt() -> None:
    """test the cli run-gpt help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "run", "--help"])
    assert "Send a prompt to an OpenAI LLM model" in result.output


def test_run_gpt_batch() -> None:
    """test the cli run-gpt help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "batch", "run", "--help"])
    assert "Batch submit papers to an OpenAI LLM model" in result.output


def test_eval_plot() -> None:
    """test the cli eval-plot help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "plot", "--help"])
    assert "Create evaluation plots" in result.output
    assert "--run-index" in result.output


def test_cm() -> None:
    """test the cli cm help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "cm", "--help"])
    assert "Save Confusion Matrix metrics for llm performance" in result.output
    assert "-f, --filename" in result.output
    assert "-i, --run-index" in result.output


def test_roc() -> None:
    """test the cli roc help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "roc", "--help"])
    assert "Save ROC metrics for llm performance" in result.output
    assert "-f, --filename" in result.output
    assert "-i, --run-index" in result.output


def test_cm_requires_filename() -> None:
    """test cm requires a bibcode file"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "cm"])
    assert result.exit_code != 0
    assert "Missing option '-f'" in result.output


def test_roc_requires_filename() -> None:
    """test roc requires a bibcode file"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "roc"])
    assert result.exit_code != 0
    assert "Missing option '-f'" in result.output


def test_cm_rejects_run_index_with_aggregate(tmp_path) -> None:
    """test cm rejects run-index in aggregate mode"""
    bibcodes = tmp_path / "bibcodes.txt"
    bibcodes.write_text("B1\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "cm", "-a", "-i", "0", "-f", str(bibcodes)])
    assert result.exit_code != 0
    assert "--run-index cannot be used with -a/--aggregate." in result.output


def test_roc_rejects_run_index_with_aggregate(tmp_path) -> None:
    """test roc rejects run-index in aggregate mode"""
    bibcodes = tmp_path / "bibcodes.txt"
    bibcodes.write_text("B1\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "roc", "-a", "-i", "0", "-f", str(bibcodes)])
    assert result.exit_code != 0
    assert "--run-index cannot be used with -a/--aggregate." in result.output


def test_roc_reports_save_errors(tmp_path, mocker) -> None:
    """test roc wraps save errors in a ClickException"""
    bibcodes = tmp_path / "bibcodes.txt"
    bibcodes.write_text("B1\n", encoding="utf-8")

    mocker.patch("bibcat.main.read_output", return_value=[{"dummy": "data"}])
    mocker.patch("bibcat.main.extract_roc_metrics_for_run", return_value={"roc_auc": 0.5})
    mocker.patch("bibcat.main.save_json_file", side_effect=IOError("disk full"))

    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "roc", "-f", str(bibcodes)])

    assert result.exit_code != 0
    assert "Failed to save metrics to" in result.output
    assert "disk full" in result.output


def test_plot_cm_requires_saved_metrics_file(tmp_path, mocker) -> None:
    runner = CliRunner()
    mocker.patch.object(main.config.paths, "output", str(tmp_path))

    result = runner.invoke(cli, ["llm", "plot", "--cm"])

    assert result.exit_code != 0
    assert "CM metrics file not found" in result.output


def test_plot_cm_reads_saved_metrics_data(tmp_path, mocker) -> None:
    runner = CliRunner()
    mocker.patch.object(main.config.paths, "output", str(tmp_path))
    plot_mock = mocker.patch("bibcat.main.cm_plot")

    output_dir = tmp_path / f"llms/openai_{main.config.llms.openai.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = main._output_path("cm", "single_r2")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "human_labels": ["SCIENCE"],
        "llm_labels": ["SCIENCE"],
        "threshold": 0.5,
        "missions": ["JWST"],
        "human_llm_missions": ["JWST"],
    }
    metrics_path.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(cli, ["llm", "plot", "--cm", "--run-index", "2"])

    assert result.exit_code == 0
    plot_mock.assert_called_once_with(metrics_data=data, metrics_type="single_r2")


def test_plot_roc_reads_saved_metrics_data(tmp_path, mocker) -> None:
    runner = CliRunner()
    mocker.patch.object(main.config.paths, "output", str(tmp_path))
    plot_mock = mocker.patch("bibcat.main.roc_plot")

    output_dir = tmp_path / f"llms/openai_{main.config.llms.openai.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = main._output_path("roc", "single_r1")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fpr": [0.0, 1.0],
        "tpr": [0.0, 1.0],
        "thresholds": [1.0, 0.0],
        "roc_auc": 0.5,
        "n_verdicts": 2,
        "human_llm_missions": ["JWST"],
        "missions": ["JWST"],
    }
    metrics_path.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(cli, ["llm", "plot", "--roc", "--run-index", "1"])

    assert result.exit_code == 0
    plot_mock.assert_called_once_with(metrics_data=data, metrics_type="single_r1")


def test_plot_cm_rejects_legacy_metrics_without_missions(tmp_path, mocker) -> None:
    runner = CliRunner()
    mocker.patch.object(main.config.paths, "output", str(tmp_path))

    output_dir = tmp_path / f"llms/openai_{main.config.llms.openai.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = main._output_path("cm", "single_r0")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "human_labels": ["SCIENCE"],
        "llm_labels": ["SCIENCE"],
        "threshold": 0.5,
        "human_llm_missions": ["JWST"],
    }
    metrics_path.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(cli, ["llm", "plot", "--cm"])

    assert result.exit_code != 0
    assert "missing required field 'missions'" in result.output


def test_batch_submit() -> None:
    """test the cli batch submit help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "batch", "submit", "--help"])
    assert "Submit a batch of papers using the OpenAI Batch API" in result.output


def test_batch_retrieve() -> None:
    """test the cli batch retrieve help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "batch", "retrieve", "--help"])
    assert "Retrieve a batch run from the OpenAI Batch API" in result.output


def test_batch_process() -> None:
    """test the cli batch process help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "batch", "process", "--help"])
    assert "Process a large batch of papers with the OpenAI Batch API" in result.output
