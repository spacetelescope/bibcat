# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Main entry point into bibcat
"""

import click

from bibcat.build_model import build_model
from bibcat.classify_papers import classify_papers
from bibcat.data.build_dataset import build_dataset
from bibcat.evaluate_basic_performance import evaluate_basic_performance


@click.group("bibcat")
def cli():
    """Command-line tool for running the bibcat package"""


@cli.command(help="build a combined dataset ")
def dataset():
    """build a combined dataset from the papertrack data and the ADS fulltext data

    Wraps the original build_dataset script.
    """
    build_dataset()


@cli.command(help="train or retrain a classical ML model")
def train():
    """Train or retrain a classical ML model

    Wraps the original build_model script.
    """
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


if __name__ == "__main__":
    cli()
