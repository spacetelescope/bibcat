# Bibcat
Bibcat classifies astronomical journal papers into multiple paper categories. The primary categories are "science", "mention", "data-influenced", and "ignore".

## Development Workflow
There are two main branches for bibcat work:

- The **dev** branch contains ongoing development work and all new work should be done in branches that are merged against **dev**.

- The **main** branch contains the latest stable release of `bibcat`.

## Installation
### Required packages and versions
- Python 3.10
- click
- deepmerge
- spacy
- nltk
- pytest
- pytest-doctestplus
- tensorflow 2.15.0
- tensorflow-hub
- tensorflow-text

### Conda env installation
Change `env_name` below with whatever you want to name the environment.
- Download conda installation yml file [here](envs/bibcat_py310.yml).
- In the terminal, run these commands.
```shell
conda env create -n `env_name` -f bibcat_py310.yml
conda activate `env_name`
python -m spacy download en_core_web_sm
```
#### Extra pacakge for Apple M1/M2 chip
For Apple Silicon chips, to utilize your GPU, you install `tensorflow-metal`.  You can run `pip install tensorflow-metal`.  To verify if tensorflow is set up to utilize your GPU, do the following:
```
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```
You should see the following output: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.  If the output is an empty list, you are not setup for GPU use.

#### Install `tensorflow-text`
- You need to install this package separately. Follow the instruction order below.

- To install `tensorflow-text`, the command `pip install -U "tensorflow-text"` does not work due to some package version conflict. You need to download the latest release library compatible with your system from [the Tensorflow library link.](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases); For instance, if you have MacOSX with python 3.10, download [this library.](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.15/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl)
- Then `pip install /path-to-download/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl`


### Bibcat installation
The `bibcat` directory contains the python package itself, installable via pip.
```shell
pip install -e .
```
## Setup
### Input JSON file
Download several data files (the ADS full text file and the papertrack file) to create models for training or combined fulltext dataset files for the input text. These files can be accessed only by authorized users. Downloading the files requires a single sign-on.
Save the files outside the `bibcat` folder on your local computer, and you will set up the paths to the files. See more details in **User Configuration and Data Filepaths** below.

- The combined papers+classification JSON file ([dataset_combined_all_2018-2023.json](https://stsci.box.com/s/q99xtyey1lgydt0jtonhot3b3rlv8rns))
- The papertrack export JSON file ([papertrack_export_2023-11-06.json](https://stsci.box.com/s/zadlr8dixw8706o9k9smlxdk99yohw4d))
- ADS fulltext data file ([ST_Request2018-2023.json](https://stsci.box.com/s/ym9pbt2iz7cqc8m1gbbd2slo0lwbwlr8))

Note that other JSON files (extracted from 2018-2023) include paper track data and full texts later than 2021.

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

### When testing with pytest or unittest

For testing, you need to install the extra test dependencies.  You do this with `pip install -e ".[test]"`.  The test suite is located in `tests/`. We can recommend using `pytest` for running tests.  Navigate to `/tests/` and run `pytest`, or for extra verbosity run `pytest -vs`. `pytest` can find and run tests written with pytest or unittests.


## Quick start

There is a CLI interface to bibcat.  After installation with `pip install -e .`, a `bibcat` cli will be available from the terminal.  Run `bibcat --help` from the terminal to display the available commands.  All commands also have their own help.  For example to see the options
for classifying papers, run `bibcat classify --help`.

- Set the three user BIBCAT_XXX_DIR environment variables specified above, in particular `BIBCAT_DATA_DIR` points to the location of your input JSON files.
- To create a training model, run `bibcat train`.
- To classify papers, run `bibcat classify`. Copy `etc/fakedata.json` to your local OPSDATA folder to test `bibcat classify`. Check out `etc/fakedata.json` to see the necessary contents for operational papers in JSON. 
- To evaluate the classifiers, run `bibcat evaluate`. It will produce some evaluation
  diagnostics such as a confusion matrix in the `output/` directory.



## License

This project is Copyright (c) Jamila Pegues, Jinmi Yoon, and Brian Cherinka and is licensed under the terms of the BSD 3-Clause license. This package is based upon the [Openastronomy packaging guide](https://github.com/OpenAstronomy/packaging-guide), which is licensed under the BSD 3-clause license. See the licenses folder for more information.
## Contributing

We love contributions! bibcat is open source, built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not ready to be an open source contributor; that your skills aren't nearly good enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at all, you can contribute code to open source. Contributing to open source projects is a fantastic way to advance one's coding skills. Writing perfect code isn't the measure of a good developer (that would disqualify all of us!); it's trying to create something, making mistakes, and learning from those mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can help out by writing documentation, tests, or even giving feedback about the project (and yes - that includes giving feedback about the contribution process). Some of these contributions may be the most valuable to the project as a whole, because you're coming to the project with fresh eyes, so you can see the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by [Adrienne Lowe](https://github.com/adriennefriend) for a [PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by bibcat based on its use in the README file for the [MetPy project](https://github.com/Unidata/MetPy).
