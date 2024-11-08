# !/usr/bin/env python
# -*- coding: utf-8 -*-
#

# pylint: disable=all

from click.testing import CliRunner

from bibcat.main import cli


def test_bibcat_cli() -> None:
    """test the top level bibcat cli"""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Command-line tool for running the bibcat package" in result.output


def test_dataset() -> None:
    """test the cli dataset help command"""

    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "--help"])
    assert "Build a combined dataset" in result.output


def test_train() -> None:
    """test the cli train help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert "Build or train a classical NLP ML model" in result.output


def test_classify() -> None:
    """test the cli classify help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["classify", "--help"])
    assert "Classify a paper using a trained model" in result.output


def test_evaluate() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert "Evaluate a trained model on efficacy and performance" in result.output


def test_evaluate_llm() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate-llm", "--help"])
    assert "Evaluate the LLM output" in result.output


def test_evaluate_llm_batch() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate-llm-batch", "--help"])
    assert "Batch evaluate the LLM output" in result.output


def test_run_gpt() -> None:
    """test the cli run-gpt help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["run-gpt", "--help"])
    assert "Send a prompt to an OpenAI LLM model" in result.output


def test_run_gpt_batch() -> None:
    """test the cli run-gpt help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["run-gpt-batch", "--help"])
    assert "Batch submit papers to an OpenAI LLM model" in result.output


def test_eval_plot() -> None:
    """test the cli eval-plot help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["eval-plot", "--help"])
    assert "Create evaulation plots" in result.output
