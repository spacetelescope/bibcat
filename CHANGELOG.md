# Change Log

### [Unreleased]
- Bibcat to do list
    - Adding tutorial notebooks back


### Removed
- [PP #48](https://github.com/spacetelescope/bibcat/pull/48)
    - Removed conda env file.
- [PR #9](https://github.com/spacetelescope/bibcat/pull/9)
    - Deleted test_bibcat.py
    - Deleted the same set of test gloabal variables assigned in multiple test scripts

- [PR #7](https://github.com/spacetelescope/bibcat/pull/7)
    - Deleted all previous codes and files for a fresh start

### Changed
- [PR #68](https://github.com/spacetelescope/bibcat/pull/68)
    -`Inconsistent_classifications.json` was revised and separated from `bibcat stats-llms`
    - Updated `metrics_summary.json` to include confusion matrix metrics
- [PR #66](https://github.com/spacetelescope/bibcat/pull/66)
    - Moved _process_database_ambig, _extract_core_from_phrase, _streamline_phrase, and _check_truematch from base.py to paper.py
    - Updated tests to read from Paper() object instead of Base() object
- [PR #64](https://github.com/spacetelescope/bibcat/pull/64) Update ROC input and docs
- [PR #63](https://github.com/spacetelescope/bibcat/pull/63) Refactored to use newer OpenAI Responses API and remove deprecated Assistants API.
- [PR #62](https://github.com/spacetelescope/bibcat/pull/62) Update `metrics.py` and its pytest
- [PR #61](https://github.com/spacetelescope/bibcat/pull/61) Update InfoModel response with enum
- [PR #56](https://github.com/spacetelescope/bibcat/pull/56)adds the MAST mission simple keyword text match to the user prompt
- [PR #54](https://github.com/spacetelescope/bibcat/pull/53) Sanitizing keywords
- [PR #53](https://github.com/spacetelescope/bibcat/pull/53) ROC curve bug fixes, add more evaluation metrics, etc
- [PR #47](https://github.com/spacetelescope/bibcat/pull/47) New calculations for evaluation confidence values for multiple GPT runs
- [PR #46](https://github.com/spacetelescope/bibcat/pull/46)
    - Grouping the BERT model method into the pretrained folder
    - Created PRETRAINED_README.md and updated the main README.md

- [PR #29](https://github.com/spacetelescope/bibcat/pull/29)
    - Refactored the ML classifier to allow for other `tensorflow` models, and for adding other libraries, e.g. `pytorch`, down the line.

- [PR #23](https://github.com/spacetelescope/bibcat/pull/23)
    - Setting a new config for the directory of papers for operational classification with a fake JSON file
    - Refactored `fetch_paper.py`
    - Other relevant updates and minor updates
- [PR #22](https://github.com/spacetelescope/bibcat/pull/22), [PR #23](https://github.com/spacetelescope/bibcat/pull/23)
    - The `is_keyword` method is replaced with the `identify_keyword` method.

- [PR # 21](https://github.com/spacetelescope/bibcat/pull/22)
    - `evaluate` and `classify` are now separate CLI options.

- [PR #19](https://github.com/spacetelescope/bibcat/pull/19) [PR #20](https://github.com/spacetelescope/bibcat/pull/20)
    - `get_config()` error fix
    - `_add_word()` temporary fix
    - `merger` erorr fix for config parameters

- [PR #18](https://github.com/spacetelescope/bibcat/pull/18)
    - Fix ddict type errors

- [PR #16](https://github.com/spacetelescope/bibcat/pull/16) [# 17](https://github.com/spacetelescope/bibcat/pull/17)
    - Consolidated all config into a single `bibcat_config.yaml` YAML file.
    - Moved `bibcat` output to outside the package directory
    - Added support for user custom configuration and settings
    - Migrated code to use new `config` object, a dottable dictionary to retain the old config syntax

- [PR #14](https://github.com/spacetelescope/bibcat/pull/14)
    - fixed various type annotation errors while refactoring `classify_papers.py` and other related modules such as `performance.py` or `operator.py`.
    - all output results will be saved under a subdirectory of the given model run in the `output` directory.
    - classify_papers.py will produce both evaluation results and classification results per method, rather than combined results of both the RB and ML methods. This way will allow users to choose a classification method using CLI once CLI is enabled.

- [PR #13](https://github.com/spacetelescope/bibcat/pull/13)
    - Enabling build_model.py to be both a module and a main script.

- [PR #12](https://github.com/spacetelescope/bibcat/pull/12)
    - Refactoring build_model.py has started, the first part includes to
        - extract generate_direcotry_TVT() from `core/classifiers/textdata.py` to create a stand-alone module, `split_dataset.py`
        - modify to store the training, validation, and test (TVT) data set directories under the `data/partitioned_datasets` directory
    - The second part in refactoring required some relevant changes to implement the new modules and updating build_module.py accordingly.
        - `build_model.py`,`base.py`, `operator.py`, `config.py`, etc.

- [PR #11](https://github.com/spacetelescope/bibcat/pull/11)
    - Renamed create_model.py to build_model.py
    - Updated README.md
    - Updated config.py to create variables to support the new script, build_dataset.py

- [PR #10](https://github.com/spacetelescope/bibcat/pull/10)
    - Renamed `test_core` to `core`
    - Renamed `test_data` to `data`

- [PR #9](https://github.com/spacetelescope/bibcat/pull/9)
    - The test global variables are called directly in the script rather than using redundantly reassigned to other variables.
    - Moved test Keyword-object lookup variables to parameters.py

- [PR #8](https://github.com/spacetelescope/bibcat/pull/8)
    - Refactored classes.py into several individual class files below, which were moved to the new folder names, `core` and `core/classifiers`.
        - `core`: base.py, grammar.py, keyword.py, operator.py, paper.py, performance.py
        - `core/classifiers`:
            - _Classfier(): ClassifierBase() in textdata.py,
            - Classifier_ML: MachineLearningClassifier() in ml.py,
            - Classifier_Rules: RuleBasedClassifier() in rules.py
    - Continued formatting and styling these class files using `Ruff` and the line length set to 120
    - Updated module updates according to the refactoring
    - Updated CHANGELOG.MD and pyproject.toml

- [PR #7](https://github.com/spacetelescope/bibcat/pull/7)
    - Updated the main README file
    - updated formatting and styling

### Added
- [PR #68](https://github.com/spacetelescope/bibcat/pull/68) Add a new CLI, `audit-llms` to create a json file to store failure modes stats and the breakdown information for failed bibcodes.
- [PR #48](https://github.com/spacetelescope/bibcat/pull/48)
    - Set up Sphinx autodoc build

- [PR #44](https://github.com/spacetelescope/bibcat/pull/44)
    - Updated LLM prompt to include its rationale and reasoning in the output
    - Switch to OpenAI Structured Response output, using pydantic models to control output

- [PR #43](https://github.com/spacetelescope/bibcat/pull/43)
    - pre-commit-hook setup
    - GitHub CI/CD action pipeline for linting/formatting and pytests
- [PR #40](https://github.com/spacetelescope/bibcat/pull/40)
    - Add `stats-llm.py` to output statistics results from the evaluation summary output and the operational gpt results
    - pytests (`test_stats_llm.py`) and llm `README.md` updated

- [PR #38](https://github.com/spacetelescope/bibcat/pull/38)
    - Add option to run gpt-batch multiple times
- [PR #35](https://github.com/spacetelescope/bibcat/pull/35)
    - Implement performance evaluation metrics and plots

- [PR #34](https://github.com/spacetelescope/bibcat/pull/34)
    - Added a summary output code for evaluation
- [PR #32](https://github.com/spacetelescope/bibcat/pull/32)
    - Added unit test for `build_dataset.py`
- [PR #31](https://github.com/spacetelescope/bibcat/pull/31)
    - Implemented ChatGPT agent prompt engineering approach to classify papers
    - Added a basic classification output
- [PR #27](https://github.com/spacetelescope/bibcat/pull/27)
    - Added a CLI option to build the combined dataset from the papertrack data and papertext (from ADS) data and refactored `build_dataset.py`.
    - Enabled dynamic version control
    - Readme update: clarify the workflow in Quick Start; the use of fetching papers using the `do_evaluation` keyword when `bibcat classify` and `bibcat evaluate`

- [PR #18](https://github.com/spacetelescope/bibcat/pull/18)
    - Added new `click` cli for `bibcat`

- [PR #14](https://github.com/spacetelescope/bibcat/pull/14)
    - Refactored `classify_papers.py` and created a few modules, which are called in `classify_papers.py`. These modules could be executed based on CLI options once they are employed.
        - `fetch_papers.py` : fetching papers from the `dir_test` data directory to the bibcat pipeline. This needs an update to fetch operational data using the `dir_datasets` argument in this module.
        - `operate_classifier.py`: the main purpose of this module is to use only one method, classify the input papers, and output classification results as a JSON file for operation.
        - `evaluate_basic_performance.py` : this module employes two performance functions to evaluate test paper classification and produce relevant files and a confusion matrix if a ML method is used.
    - created `fakedata.txt` in `/bibcat/data/operational_data/` to test operational classification with simple ascii text
    - created `fake_testdata.json`, which has paper classification with its associated simple text, for testing and performance evaluation.
    - included additional VS code ruff setting to `pyproject.toml`

- [PR #12](https://github.com/spacetelescope/bibcat/pull/12)
    - The second part of refactoring `build_model.py` includes
        - create a new module, `model_settings.py` to set up various model related variables. This eventually will relocating other model related variables in `config.py` to this module in the near future.
        - Created `streamline_dataset.py` to streamline the source data equipped to be ML input dataset. It does load the source dataset and streamline the dataset.
        - Created `partition_dataset.py` to split the streamlined dataset into the train, validation, and test dataset for DL models.
- [PR # 11](https://github.com/spacetelescope/bibcat/pull/11)
    - Create a new script to build the input dataset. It's called build_dataset.py
    - Added some information about the data folder in README.rst
    - Added __init__.py

- [PR #9](https://github.com/spacetelescope/bibcat/pull/9)
    - test_bibcat.py was refactored into several sub test scripts.
        - tests/test_core/test_base.py
        - tests/test_core/test_grammar.py
        - tests/test_core/test_keyword.py
        - tests/test_core/test_operator.py
        - tests/test_core/test_paper.py
        - tests/test_data/test_dataset.py

- [PR #8](https://github.com/spacetelescope/bibcat/pull/8)
    - Created a new folder named `core` to store all refactored class scripts
    - Added more description to each class script and other main scripts.

- [PR #7](https://github.com/spacetelescope/bibcat/pull/7)
    - Started with open astronomy cookiecutter template for bibcat
    - Re-organized the file structure (e.g., bibcat/bibcat/) and modified the file names
        - bibcat_classes.py to classes.py
        - bibcat_config.py to config.py
        - bibcat_parameters.py to parameters.py
        - bibcat_tests.py to test_bibcat.py
    - Refactor classes.py into several individual class script under the `core` directory
    - Created two main scripts
        - create_model.py :  this script can be run to create a new training model
        - classify_papers.py : this script will fetch input papers, classify them into the designated paper categories, and produce performance evaluation materials such as confusion matrix and plots
    - Created CHANGELOG.md

## [0.1.0] - 2024-01-29

Initial tag to preserve code before refactor
