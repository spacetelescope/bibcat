Quick start
===========

There is a CLI interface to bibcat.  After installation with ``pip install -e .``, a `bibcat` cli will be available from the terminal.  Run ``bibcat --help`` from the terminal to display the available commands.  All commands also have their own help.  For example to see the options
for classifying papers, run ``bibcat train --help``.

- First, set the three user BIBCAT_XXX_DIR environment variables specified above, in particular `BIBCAT_DATA_DIR` points to the location of your input JSON files.

Build The Dataset
-----------------

- run ``bibcat dataset`` if you don't already have the source dataset combined from the papertrack data and the papertext data.

Using Pretrained Models (BERT flavors)
--------------------------------------

You can classify papers using the pretrained models like `BERT` or `RoBERTa`. Please see the :doc:`pretrained` to get started.

Using LLM Prompting Method
------------------------------

You can submit paper content to OpenAI's gpt models.  Please see the :doc:`llm` to get started.
