# Change Log

## [Unreleased]

Refactoring bibcat has started.
- Refactoring test_bibcat.py
- Refactoring create_model.py and classify_papers.py
- Adding tutorial notebooks back

### Removed
- PR #9 
    - Deleted test_bibcat.py 

- [PR #7](https://github.com/spacetelescope/bibcat/pull/7)
    - Deleted all previous codes and files for a fresh start

### Changed

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

- PR #9
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