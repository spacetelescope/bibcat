# Change Log

## [Unreleased]

- A few bug fixes
    - `_add_word()` temporary fix
    - `merger` erorr fix for config parameters
    - `get_config() error fix


Bibcat to do list
- Refactor performance.py
- Adding tutorial notebooks back


### Removed
- [PR #9]
    - Deleted test_bibcat.py
    - Deleted the same set of test gloabal variables assigned in multiple test scripts

- [PR #7](https://github.com/spacetelescope/bibcat/pull/7)
    - Deleted all previous codes and files for a fresh start

### Changed
- [PR #17 # 18]
    - Consolidated all config into a single `bibcat_config.yaml` YAML file.
    - Moved `bibcat` output to outside the package directory
    - Added support for user custom configuration and settings
    - Migrated code to use new `config` object, a dottable dictionary to retain the old config syntax

- [PR #14]
    - fixed various type annotation errors while refactoring `classify_papers.py` and other related modules such as `performance.py` or `operator.py`.
    - all output results will be saved under a subdirectory of the given model run in the `output` directory.
    - classify_papers.py will produce both evaluation results and classification results per method, rather than combined results of both the RB and ML methods. This way will allow users to choose a classification method using CLI once CLI is enabled.

- [PR #13]
    - Enabling build_model.py to be both a module and a main script.

- [PR #12]
    - Refactoring build_model.py has started, the first part includes to
        - extract generate_direcotry_TVT() from `core/classifiers/textdata.py` to create a stand-alone module, `split_dataset.py`
        - modify to store the training, validation, and test (TVT) data set directories under the `data/partitioned_datasets` directory
    - The second part in refactoring required some relevant changes to implement the new modules and updating build_module.py accordingly.
        - `build_model.py`,`base.py`, `operator.py`, `config.py`, etc.

- [PR #11]
    - Renamed create_model.py to build_model.py
    - Updated README.md
    - Updated config.py to create variables to support the new script, build_dataset.py

- [PR #10]
    - Renamed `test_core` to `core`
    - Renamed `test_data` to `data`

- [PR #9]
    - The test global variables are called directly in the script rather than using redundantly reassigned to other variables.
    - Moved test Keyword-object lookup variables to parameters.py

- [PR #8]
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
- [PR #18] 
    - Added new `click` cli for `bibcat`

- [PR #14]
    - Refactored `classify_papers.py` and created a few modules, which are called in `classify_papers.py`. These modules could be executed based on CLI options once they are employed.
        - `fetch_papers.py` : fetching papers from the `dir_test` data directory to the bibcat pipeline. This needs an update to fetch operational data using the `dir_datasets` argument in this module.
        - `operate_classifier.py`: the main purpose of this module is to use only one method, classify the input papers, and output classification results as a JSON file for operation.
        - `evaluate_basic_performance.py` : this module employes two performance functions to evaluate test paper classification and produce relevant files and a confusion matrix if a ML method is used.
    - created `fakedata.txt` in `/bibcat/data/operational_data/` to test operational classification with simple ascii text
    - created `fake_testdata.json`, which has paper classification with its associated simple text, for testing and performance evaluation.
    - included additional VS code ruff setting to `pyproject.toml`

- [PR #12]
    - The second part of refactoring `build_model.py` includes
        - create a new module, `model_settings.py` to set up various model related variables. This eventually will relocating other model related variables in `config.py` to this module in the near future.
        - Created `streamline_dataset.py` to streamline the source data equipped to be ML input dataset. It does load the source dataset and streamline the dataset.
        - Created `partition_dataset.py` to split the streamlined dataset into the train, validation, and test dataset for DL models.
- [PR # 11]
    - Create a new script to build the input dataset. It's called build_dataset.py
    - Added some information about the data folder in README.rst
    - Added __init__.py

- [PR #9]
    - test_bibcat.py was refactored into several sub test scripts.
        - tests/test_core/test_base.py
        - tests/test_core/test_grammar.py
        - tests/test_core/test_keyword.py
        - tests/test_core/test_operator.py
        - tests/test_core/test_paper.py
        - tests/test_data/test_dataset.py

- [PR #8]
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