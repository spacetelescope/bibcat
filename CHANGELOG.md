# Change Log

## [Unreleased]

Refactoring bibcat has started.
- Refactoring classes.py, parameters.py, and config.py 
- Refactoring create_model.py and classify_papers.py
- Adding tutorial notebooks back

### Removed
- Deleted all previous codes and files for a fresh start

### Changed
- Updated the main README file
- updated formatting and styling

### Added 
- Started with open astronomy cookiecutter template for bibcat 
- Re-organized the file structure (e.g., bibcat/bibcat/) and modified the file names
    - bibcat_classes.py to classes.py
    - bibcat_config.py to config.py
    - bibcat_parameters.py to parameters.py
    - bibcat_tests.py to test_bibcat.py
- Created two main scripts
    - create_model.py :  this script can be run to create a new training model
    - classify_papers.py : this script will fetch input papers, classify them into the designated paper categories, and produce performance evaluation materials such as confusion matrix and plots
- Created CHANGELOG.md

## [0.1.0] - 2024-01-29

Initial tag to preserve code before refactor