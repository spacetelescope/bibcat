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
    assert "LLM-based paper classification" in result.output
    assert "Classical ML-based paper classification" in result.output


def test_dataset() -> None:
    """test the cli dataset help command"""

    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "--help"])
    assert "Build a combined dataset" in result.output


def test_ml() -> None:
    """test the cli ml help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "--help"])
    assert "ML-based paper classification using classical NLP models, e.g. BERT" in result.output
    assert "Classify a paper using a trained model" in result.output


def test_train() -> None:
    """test the cli train help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "train", "--help"])
    assert "Build or train a classical NLP ML model" in result.output


def test_classify() -> None:
    """test the cli classify help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "classify", "--help"])
    assert "Classify a paper using a trained model" in result.output


def test_evaluate() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "evaluate", "--help"])
    assert "Evaluate a trained model on efficacy and performance" in result.output


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


def test_stats() -> None:
    """test the cli stats help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "stats", "--help"])
    assert "Create a statisics table for classification" in result.output


def test_audit() -> None:
    """test the cli audit help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["llm", "audit", "--help"])
    assert "Create a JSON file to audit LLM classification" in result.output
