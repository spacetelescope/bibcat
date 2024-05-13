# !/usr/bin/env python
# -*- coding: utf-8 -*-
#

# pylint: disable=all

import pytest
from click.testing import CliRunner

from bibcat.main import cli


def test_bibcat_cli():
    """ test the top level bibcat cli """
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Command-line tool for running the bibcat package' in result.output


def test_build():
    """ test the cli train help command """
    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--help'])
    assert 'train or retrain a classical ML model' in result.output

def test_classify():
    """ test the cli classify help command """
    runner = CliRunner()
    result = runner.invoke(cli, ['classify', '--help'])
    assert 'classify a paper using a trained model' in result.output
