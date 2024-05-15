
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Main entry point into bibcat
"""

import click

from bibcat.build_model import build_model
from bibcat.classify_papers import classify_papers


@click.group('bibcat')
def cli():
    """ Command-line tool for running the bibcat package """


@cli.command(help='train or retrain a classical ML model')
def train():
    """ Train or retrain a classical ML model

    Wraps the original build_model script.
    """
    build_model()


@cli.command(help='fine-tune a LLM model')
def finetune():
    pass


@cli.command(help='classify a paper using a trained model')
@click.option('-n', '--name', default='ML', type=click.Choice(['ML', 'RB']), show_default=True,
              help="The type of classifier to use.  Either machine-learning (ML) or rule-based (RB).")
def classify(name):
    """ Classify a paper using a trained model

    Wraps the original classify_papers script.
    """
    classify_papers(classifier_name=name)


@cli.command(help='update the training dataset JSON file')
def update():
    pass


@cli.command(help='evaluate a trained model on efficacy and performance')
def evaluate():
    pass


if __name__ == '__main__':
    cli()

