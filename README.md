# BibCat
Bibcat classifies astronomical journal papers into multiple categories. The primary categories are "science" and "mention." In our work, we focus on distinguishing between "science" and "nonscience" papers, where "nonscience" includes "mention" and other non-mission-relevant papers.

## Development Workflow
There are two main branches for bibcat work:

- The dev branch contains ongoing development work. All new features and changes should be developed in branches that are merged into `dev`.

- The main branch contains the latest stable release of `bibcat`.

## Installation
### Required packages and versions
- See the required package dependencies found in the [pyproject.toml](https://github.com/spacetelescope/bibcat/blob/dev/pyproject.toml).
- A few tensorflow packages required for Apple silicon chip computers should be installed manually; see below.

### Conda environment installation
Change `env_name` below with whatever you want to name the environment.
- In the terminal, run these commands.


```shell
conda create -n env_name python=3.10
conda activate env_name
```

If you want to create a lightweight python environment, you can use `micromamba`, which is fast alternative to conda, written in C++, that implements the same CLI interface. Follow this [instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to install `micromamba`.

```shell
micromamba create -n env_name python=3.10
micromamba activate env_name
```

### Bibcat installation
The `bibcat` directory contains the python package itself, installable via pip. Move to the main bibcat root directory where `pyproject.toml` is located and run this command. This will only install the dependencies needed to run the LLM component of bibcat.  **Note:** you still need to manually run the `spacy download` command specified below.

```shell
pip install .
```
#### Installation for developers
If you are interested in developing and contributing to **casper**, you should install this package with `-e`, it allows you to work on the package's source code and see changes reflected immediately without needing to reinstall.

```shell
pip install -e . # install editable mode
```

To install all dependencies for development except for the ML component, testing, and documentation, run `pip install -e ".[dev,test,docs]"` or `pip install -e .[all]`.


### Spacy model downloads
*Note that some core tests using `spacy` could fail if the version number is not `3.7.2`. You could reinstall `pip install spacy==3.7.2` if that happens. This is a work-around solution until we have the capacity to update the tests.

This model is used for the [Pretrained model method](https://github.com/spacetelescope/bibcat/blob/dev/docs/pretrained.md)
```
python -m spacy download en_core_web_sm-3.7.1
```

### Tensorflow package installation for `Pretrained` method
`tensorflow` packages are used for the [Pretrained model method](https://github.com/spacetelescope/bibcat/blob/dev/docs/pretrained.md)

#### For CPU computers (e.g., intel chips)
To install the Tensorflow dependencies for use of the ML component of bibcat, run `pip install -e ".[cpu_ml]"`.

#### For Apple silicon M1/M2/M3 chip computers
- If you have an Apple Silicon chip computer and want to utilize your GPU, you run `pip install -e ".[gpu_ml]"` and follow the tensorflow instructions below. If not, skip this part.

 To verify if tensorflow is set up to utilize your GPU, do the following:

  ```python
  import tensorflow as tf
  tf.config.list_physical_devices('GPU')
  ```
  You should see the following output: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.  If the output is an empty list, you are not setup for GPU use.

##### Install `tensorflow-text`

- For Apple silicon M1/M2/M3 chip, to install `tensorflow-text`, the command `pip install -U "tensorflow-text"` **does not work** due to some package version conflict (as of sometime 2024, need to revisit). You need to download the latest release library compatible with your system and the tensorflow version (2.15.0 in the example) from [the Tensorflow library link.](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases); For instance, if you have MacOSX with python 3.10, download [this library.](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.15/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl)
- Then `pip install /path-to-download/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl`


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
### Input JSON file <!--- THIS NEEDS A REVISION to remove the Box Links for PRODUCTION --->
Download several data files (the ADS full text file and the papertrack file) to create models for training or combined fulltext dataset files for the input text. These files can be accessed only by authorized users. Downloading the files requires a single sign-on.
Save the files outside the `bibcat` folder on your local computer, and you will set up the paths to the files. See more details in **User Configuration and Data Filepaths** below.
We refer
- the combined papers+classification JSON file ([combined_dataset_2025_07_08.json](https://stsci.box.com/s/4xnzbgq9vw3lt34lyxumeo0nnil7x7lx)) to `source data`,
- the papertrack export JSON file ([papertrack_export_papertext_2025-07-08.json](https://stsci.box.com/s/4jdvotw1hdz6d9i1l7uvj1o2u2a3ddow)) to `papertrack data` (extract the tar.gz file to use), and
- the ADS fulltext data file ([ST_Request2023_cleaned_2025_03_10.json](https://stsci.box.com/s/0a5uzmfsnokx1rth8m6wseybpz5bxne3)) to `papertext data`.

Check out [the readme about the input data](https://github.com/spacetelescope/bibcat/blob/dev/docs/data_readme.rst) for more details.

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

sphinx-apidoc -o docs/api bibcat bibcat/tests/* # the last pattern indicates all test modules excluded from API Doc
```
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

### Using Pretrained Models (BERT flavors)

You can classify papers using the pretrained models like `BERT` or `RoBERTa`. Please see the following [Quick Start Guide using Pretrained Models](https://github.com/spacetelescope/bibcat/blob/dev/docs/pretrained.md) to get started.

### Using LLM Prompting Method

You can submit paper content to OpenAI's gpt models.  Please see the following [Quick Start Guide using LLM Prompting](https://github.com/spacetelescope/bibcat/blob/dev/docs/llm.md) to get started.


## License

This project is Copyright (c) Jamila Pegues, Jinmi Yoon, and Brian Cherinka and is licensed under the terms of the BSD 3-Clause license. This package is based upon the [Openastronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide), which is licensed under the BSD 3-clause license. See the licenses folder for more information.
## Contributing

We love contributions! bibcat is open source, built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not ready to be an open source contributor; that your skills aren't nearly good enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at all, you can contribute code to open source. Contributing to open source projects is a fantastic way to advance one's coding skills. Writing perfect code isn't the measure of a good developer (that would disqualify all of us!); it's trying to create something, making mistakes, and learning from those mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can help out by writing documentation, tests, or even giving feedback about the project (and yes - that includes giving feedback about the contribution process). Some of these contributions may be the most valuable to the project as a whole, because you're coming to the project with fresh eyes, so you can see the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by [Adrienne Lowe](https://github.com/adriennefriend) for a [PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by bibcat based on its use in the README file for the [MetPy project](https://github.com/Unidata/MetPy).
