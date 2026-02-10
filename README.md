# BibCAT
BibCAT (Bibliography Classification Automation Tool) classifies astronomical journal papers into multiple categories. The primary categories are "science" and "mention." In our work, we focus on distinguishing between "science" and "nonscience" papers, where "nonscience" includes "mention" and other papers that are not relevant to the mission.

## Development Workflow
There are two main branches for bibcat work:

- The dev branch contains ongoing development work. All new features and changes should be developed in branches that are merged into `dev`.

- The main branch contains the latest stable release of `bibcat` (coming soon).

## Installation
### Required packages and versions
- See the required package dependencies found in the [pyproject.toml](https://github.com/spacetelescope/bibcat/blob/dev/pyproject.toml).

### Conda environment installation
Change `env_name` below with your preferred name for the environment.
- In the terminal, run these commands.


```shell
conda create -n env_name python=3.10
conda activate env_name
```

If you want to create a lightweight python environment, you can use `micromamba`, which is fast alternative to conda, written in C++, that implements the same CLI interface. Follow this [mamba instruction](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to install `micromamba` and the following step.

```shell
micromamba create -n env_name python=3.10
micromamba activate env_name
```

### BibCAT installation
The `bibcat` directory contains the python package itself, installable via pip. Move to the main bibcat root directory where `pyproject.toml` is located and run this command. This will only install the dependencies needed to run the LLM component of bibcat.  **Note:** you still need to manually run the `spacy download` command specified below.

```shell
pip install .
```
#### Installation for developers
If you are interested in developing and contributing to **BibCAT**, you should install this package with `-e`, it allows you to work on the package's source code and see changes reflected immediately without needing to reinstall.

```shell
pip install -e . # install editable mode
```


### Spacy model downloads
spaCy is a Python library that provides efficient NLP tools for text preprocessing, including tokenization, tagging, and named entity recognition.

*Note: Some core tests that use spaCy may fail if the version is not 3.7.2. If this happens, you can reinstall it with pip install spacy==3.7.2. This is a temporary workaround until we have the capacity to update the tests.*

This model is used for processing the input text, which is then analyzed further to identify the mission keywords:
```
python -m spacy download en_core_web_sm
```

## pre-commit for development

[pre-commit](https://pre-commit.com/) allows all collaborators push their commits compliant with the same set of lint and format rules in [pyproject.toml](https://github.com/spacetelescope/bibcat/blob/dev/pyproject.toml) by checking all files in the project at different stages of the git workflow. It runs commands specified in the [.pre-commit-config.yaml](https://github.com/spacetelescope/bibcat/blob/dev/.pre-commit-config.yaml) config file and runs checks before committing or pushing, to catch errors that would have caused a build failure before they reach CI.

### Install pre-commit
You will need to install `pre-commit` manually. `pre-commit` is included in `dev` dependencies in `pyproject.toml`.
```bash
pip install pre-commit # if you haven't already installed the package.
```

```bash
pre-commit install # install default hooks `pre-commit`, `pre-push`, and `commit-msg` as specified in the config file.
```

If this is your first time running, you should run the hooks against for all files and it will fix all files based on your setting.
```bash
pre-commit run --all-files
```
Finally, you will need to update `pre-commit` regularly by running
```bash
pre-commit autoupdate
```
For other configuration options and more detailed information, check out at the [pre-commit](https://pre-commit.com/) page.


## Setup
### Input JSON file

To build training models or create a combined full-text dataset for input, you’ll need to download several data files: the ADS full-text file and the papertrack file. These files are accessible only to authorized users and require single sign-on (SSO) for download.

> **Important:**
Save these files **outside** the `bibcat` folder on your local machine. You will later configure file paths to point to them.
For more on this setup, see [**User Configuration and Data Filepaths**](https://bibcat.readthedocs.io/en/latest/readme.html#user-configuration-and-data-filepaths).

We refer to the following files throughout this guide:

- **Source data**:
  [combined_dataset_2025_07_08.json](https://stsci.box.com/s/4xnzbgq9vw3lt34lyxumeo0nnil7x7lx) — a combined papers + classification JSON file.

- **Papertrack data**:
  [papertrack_export_papertext_2025-07-08.json](https://stsci.box.com/s/4jdvotw1hdz6d9i1l7uvj1o2u2a3ddow) — export from papertrack.
  _(Extract the `.tar.gz` file to access the JSON.)_

- **Papertext data**:
  [ST_Request2023_cleaned_2025_03_10.json](https://stsci.box.com/s/0a5uzmfsnokx1rth8m6wseybpz5bxne3) — full-text data from ADS.

For details on the input files and how to use them to build your own datasets, see the [Input Data Readme](https://bibcat.readthedocs.io/en/latest/data_readme.html).


### User Configuration and Data Filepaths

There are three user environment variables to set:

- **BIBCAT_CONFIG_DIR**: a local path to your user configuration yaml file
- **BIBCAT_OPSDATA_DIR** : a local path to the directory of operational data in JSON format.
- **BIBCAT_DATA_DIR**: a local path to the directory of input data, e.g the input JSON files and full text
- **BIBCAT_OUTPUT_DIR**: a local path to a directory where the output of bibcat will be written, e.g. the output model and QA plots

If not set, all envvars will default to the user's home directory.  You can set these environment variables in your shell terminal, or in your shell config file, i.e. `.bashrc` or `.zshrc` file. For example,
```bash
export BIBCAT_CONFIG_DIR=/my/local/path/to/custom/config
export BIBCAT_DATA_DIR=/my/local/path/to/input/data/dir
export BIBCAT_OPSDATA_DIR=/my/local/path/to/operational/data/dir
export BIBCAT_OUTPUT_DIR=/my/local/path/to/bibcat/output
```

All `bibcat` configuration is contained in a YAML configuration file, `bibcat_config.yaml` .  The default settings are located in `etc/bibcat_config.yaml`.  You don't modify this file directly.  To modify any of the settings, you do so through a custom user configuration file of the same name, placed in `$BIBCAT_CONFIG_DIR` or your home directory, mirroring the same default structure.  All user custom settings override the defaults.

For example, to change the name of the output model saved, within your user `$BIBCAT_CONFIG_DIR/bibcat_config.yaml`, set
```yaml
output:
  name_model: my_new_model
```

### When testing with pytest

The test suite is located in `tests/`. We can recommend using `pytest` for running tests.  Navigate to `/tests/` and run `pytest`, or for extra verbosity run `pytest -vs`. `pytest` can find and run tests written with pytest or unittests.

### Building the documentation
Sphinx will create the documentation automatically using the module docstrings.
Use `sphinx-apidoc` to automatically generate API documentation from your docstrings.

Run
```shell

sphinx-apidoc -o docs/api bibcat bibcat/tests/
```
The last pattern in the command indicates all test modules excluded from API Doc.

To build live-reload documentation, run

```shell
sphinx-autobuild docs docs/_build/html
```

For one time build,
```shell
make -C docs html
```

Then navigate to `docs/_build/html` and open `index.html` on your browser to see the built documentation.

However, you can build live API docs and htmls together with this one command,

```shell
cd docs
make live-docs
```

To remove existing output,

```shell
make clean
```

## Quick start

There is a CLI interface to bibcat.  After installation with `pip install .`, a `bibcat` cli will be available from the terminal.  Run `bibcat --help` from the terminal to display the available commands.  All commands also have their own help.  For example to see the options
for classifying papers, run `bibcat train --help`.

- First, set the three user BIBCAT_XXX_DIR environment variables specified above, in particular `BIBCAT_DATA_DIR` points to the location of your input JSON files.

### Build The Dataset

- run `bibcat dataset`if you don't already have the source dataset combined from the papertrack data and the papertext data.

### Using LLM Prompting Method

You can submit paper content to OpenAI's gpt models.  Please see the following [Quick Start Guide using LLM Prompting](https://bibcat.readthedocs.io/en/latest/llm.html) to get started.


## License

This project is Copyright (c) Mikulski Archive for Space Telescopes and is licensed under the terms of the BSD 3-Clause license. This package is based upon the [Openastronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide), which is licensed under the BSD 3-clause license. See the licenses folder for more information.

## Contributing

We love contributions! bibcat is open source, built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not ready to be an open source contributor; that your skills aren't nearly good enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at all, you can contribute code to open source. Contributing to open source projects is a fantastic way to advance one's coding skills. Writing perfect code isn't the measure of a good developer (that would disqualify all of us!); it's trying to create something, making mistakes, and learning from those mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can help out by writing documentation, tests, or even giving feedback about the project (and yes - that includes giving feedback about the contribution process). Some of these contributions may be the most valuable to the project as a whole, because you're coming to the project with fresh eyes, so you can see the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by [Adrienne Lowe](https://github.com/adriennefriend) for a [PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by bibcat based on its use in the README file for the [MetPy project](https://github.com/Unidata/MetPy).
