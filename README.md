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
Download the papers+classification JSON file ([dataset_combined_all_2018-2021.json](https://stsci.app.box.com/file/1380606268376)) from the Box folder. This file is accessed only by the authorized users. The downloading the file requires single sign-on.
Save the file outside the bibcat folder on your local computer and you will set up the path to the file as a variable called `path_json` in `bibcat_config.py`.


## Quick start

- First, set the variables `path_json` (JSON file location path)  and `name_model` (Model name of your choice to save or load) in src/`bibcat_config.py`.
- Next, run `bibcat_tutorial_trainingML.ipynb` to create a training model.  
- Then, run `bibcat_tutorial_workflow.ipynb` to see the bibcat workflow overview.
- Finally, run `bibcat_tutorial_performance.ipynb` to see some output (output/) below.
    - confusion matrix plot: `confmatr.png`
    - mis-classification lists: `test_misclassif_Operator_1.txt` and `test_misclassif_Operator_2.txt`
    - `test_eval.npy`
