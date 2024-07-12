# !/usr/bin/env python
# -*- coding: utf-8 -*-
#

# pylint: disable=all

import pytest
from click.testing import CliRunner

from bibcat.main import cli


def test_bibcat_cli() -> None:
    """test the top level bibcat cli"""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Command-line tool for running the bibcat package" in result.output


def test_train() -> None:
    """test the cli train help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert "Build or train a classical NLP ML model" in result.output


def test_classify() -> None:
    """test the cli classify help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["classify", "--help"])
    assert "classify a paper using a trained model" in result.output


def test_evaluate() -> None:
    """test the cli evaluate help command"""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert "evaluate a trained model on efficacy and performance" in result.output
