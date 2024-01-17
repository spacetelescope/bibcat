# bibcat

This repository contains the bibcat codebase, used to classify texts based on the categories of the journal papers using MAST data. The main category is "science", "data-influence", and "mention".

## Installation

### Required packages and versions
- Python 3.10
- spacy
- nltk
- tensorflow
- tf-models-official
### Conda installation
Change `env_name` below with whatever you want to name the environment.
- Download conda installation yml file [here](bibcat_py310.yml)
- conda env create -n `env_name` -f bibcat_py310.yml
- conda activate `env_name`
- pip install -U "tensorflow-text"
- python -m spacy download en_core_web_sm


### Input JSON file
Download several data files for training models or create combined fulltext dataset files using the ADS full text files and the papertrack file. These files are accessed only by authorized users. Downloading the files requires a single sign-on.
Save the files outside the bibcat folder on your local computer, and you will set up the paths to the files. See more details in Set data file paths below.

- The combined papers+classification JSON file ([dataset_combined_all_2018-2021.json](https://stsci.app.box.com/file/1380606268376))
- The papertrack export JSON file ([papertrack_export_2023-11-06.json])(https://stsci.box.com/s/zadlr8dixw8706o9k9smlxdk99yohw4d))
- ADS fulltext data file ([ST_Request2018-2021.json])(https://stsci.box.com/s/cl3yg5mxqcz484w0iptwbl9b43h4tk1g)

Note that other JSON files (extracted from 2018-2023) include paper track data and full texts later than 2021. If you like to test them for your work, feel free to do so.

### Set data file paths
In `bibcat_config.py`, you will need to set several paths as follow.

#### The combined data set used for training models.
Set the combined data file path outside the `bibcat` folder, which should not be git-committed to the repo.
- `path_json = path/to/dataset_combined_all_2018-2021.json`

#### `bibcat_tests.py` testing purposes only
Set the input file path outside the `bibcat` folder, which should not be git-committed to the repo.
- `filepath_input = "/path/to/the/dataset"`
- `path_papertrack = os.path.join(filepath_input, "papertrack_export_2023-11-06.json")`
- `path_papertext = os.path.join(filepath_input, "ST_Request2018-2021.json")`

## Quick start

- First, set the variables `path_json` (JSON file location path)  and `name_model` (Model name of your choice to save or load) in src/`bibcat_config.py`.
- Next, run `bibcat_tutorial_trainingML.ipynb` to create a training model.  
- Then, run `bibcat_tutorial_workflow.ipynb` to see the bibcat workflow overview.
- Finally, run `bibcat_tutorial_performance.ipynb` to see some output (output/) below.
    - confusion matrix plot: `confmatr.png`
    - mis-classification lists: `test_misclassif_Operator_1.txt` and `test_misclassif_Operator_2.txt`
    - `test_eval.npy`
