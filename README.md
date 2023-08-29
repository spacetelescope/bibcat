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
Download the papers+classification JSON file ([dataset_combined_all.json](https://stsci.app.box.com/file/1284375342540)) from the Box folder. This file is accessed only by the authorized users. The downloading the file requires single sign-on. 
Save the file outside the bibcat folder on your local computer and you will set up the path to the file as a variable called `path_json` in `bibcat_config.py`.

