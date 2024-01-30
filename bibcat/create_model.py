"""
:title: create_model.py

This module creates a new training ML model.

- The input full text JSON file (papertrack + ADS full texts) is called via
  `PATH_INPUT` configured in `bibcat/config.py` and is used for training,
  validating, and testing the trained model.
  
- Once the model is trained, the `models` folder and its subdirectories for T/V/T
  are created along with various model related files. 

:Run example: python create_model.py
"""

import os
import time
import json

from bibcat import classes
from bibcat import config
from bibcat import parameters as params

# do_check_truematch: If any papers in dataset encountered within
# the codebase that have unknown ambiguous phrases, then a note will be
# printed and those papers will not be used for training-validation-testing.
# Add the identified ambiguous phrase to the external ambiguous phrase
# database and rerun to include those papers.
do_check_truematch = True

# num_papers: Set None to use all available papers in external dataset
# Note: If set to integer, final paper count might be a little more than
# target num_papers given
num_papers = 500
do_verbose_text_summary = True  # print out the text info summary per paper.

# For external data; classifications to include
allowed_classifications = params.allowed_classifications
# For masking of classes (e.g., masking 'supermention' as 'mention')
mapper = params.map_papertypes

# Fetch filepath for model
name_model = config.name_model
dir_model = os.path.join(config.dir_allmodels, name_model)

# Fetch filepaths for input and output
filepath_input = config.PATH_INPUT
filepath_output = config.PATH_OUTPUT

# filepath to save processing errors
filesave_error = os.path.join(dir_model,
                              "{0}_processing_errors.txt".format(name_model))

# Set values for generating ML model below

# do_reuse_run: Whether or not to reuse any existing output from previous
# training+validation+testing runs
do_reuse_run = True
# do_shuffle: Whether or not to shuffle contents of training vs. validation
# vs. testing datasets
do_shuffle = True
# Fractional breakdown of training vs. validation vs. testing dataset
fraction_TVT = [0.8, 0.1, 0.1]

# mode_TVT: "uniform" = all training datasets will have the same number of
# entries from fraction_TVT
# "available" = all training datasets will use full fraction
mode_TVT = "uniform"

seed_TVT = 10  # Random seed for generating train vs valid. vs test datasets
seed_ML = 8  # Random seed for ML model
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = "skim_anon"

# buffer: the number of sentences to include within paragraph around
# each sentence with target terms
buffer = 0
all_kobjs = params.all_kobjs  # mission keywords to use

# Initialize an empty ML classifier
classifier_ML = classes.Classifier_ML(
    filepath_model=None, fileloc_ML=None, do_verbose=True)

# Initialize an Operator
tabby_ML = classes.Operator(classifier=classifier_ML,
                            mode=mode_modif, keyword_objs=all_kobjs,
                            do_verbose=True,
                            load_check_truematch=do_check_truematch,
                            do_verbose_deep=False)

# Load the original data
with open(filepath_input, 'r') as openfile:
    dataset = json.load(openfile)
    len(dataset)

# keep track of used bibcodes avoiding duplicate dataset entries
list_bibcodes = []

# Organize a new version of the data with: key:text,class,id,mission structure
i_track = 0  # Track number of papers kept from original dataset
dict_texts = {}
for ii in range(0, len(dataset)):
    # Extract mission classifications for current text
    curr_data = dataset[ii]

    # Skip if no valid text at all for this text
    if ("body" not in curr_data):
        continue

    # Skip if no valid missions at all for this text
    if ("class_missions" not in curr_data):
        continue

    # Otherwise, extract the bibcodes and missions
    curr_bibcode = curr_data["bibcode"]
    curr_missions = curr_data["class_missions"]
    print(curr_bibcode)
    print(curr_missions)

    # Skip if bibcode already encountered (and so duplicate entry)
    if (curr_bibcode in list_bibcodes):
        print("Duplicate bibcode encountered: {0}. Skipping.".format(
            curr_bibcode))
        continue

    # Iterate through missions for this text
    i_mission = 0
    for curr_key in curr_missions:
        # If this is not an allowed mission, skip
        if (curr_missions[curr_key]["papertype"] not in allowed_classifications):
            continue

        # Otherwise, check if this mission is a target mission
        fetched_kobj = tabby_ML._fetch_keyword_object(lookup=curr_key,
                                                      do_verbose=False,
                                                      do_raise_emptyerror=False)
        # Skip if not a target
        if (fetched_kobj is None):
            continue

        # Otherwise, store classification info for this entry
        curr_class = curr_missions[curr_key]["papertype"]
        new_dict = {"text": curr_data["body"],
                    "bibcode": curr_data["bibcode"],
                    "class": curr_class,  # Classification for this mission
                    "mission": curr_key,
                    "id": ("paper{0}_mission{1}_{2}_{3}".format(ii, i_mission,
                                                                curr_key,
                                                                curr_class))
                    }
        dict_texts[str(i_track)] = new_dict

        # Increment counters
        i_mission += 1  # Count of kept missions for this paper
        i_track += 1  # Count of kept classifications overall
    #

    # Record this bibcode as stored
    list_bibcodes.append(curr_bibcode)

    # Terminate early if requested number of papers reached
    if ((num_papers is not None) and (i_track >= num_papers)):
        break


# Throw error if not enough text entries collected

if ((num_papers is not None) and (len(dict_texts) < num_papers)):
    raise ValueError("Err: Something went wrong during initial processing. "
                     + "Insufficient number of texts extracted."
                     + "\nRequested number of texts:"
                     + " {0}\nActual number of texts: {1}"
                     .format(num_papers, len(dict_texts)))

# print a snippet of each of the entries in the dataset.
print("Number of processed texts: {0}={1}\n".format(i_track, len(dict_texts)))
if do_verbose_text_summary:
    for curr_key in dict_texts:
        print("Text #{0}:".format(curr_key))
        print("Classification: {0}".format(dict_texts[curr_key]["class"]))
        print("Mission: {0}".format(dict_texts[curr_key]["mission"]))
        print("ID: {0}".format(dict_texts[curr_key]["id"]))
        print("Bibcode: {0}".format(dict_texts[curr_key]["bibcode"]))
        print("Text snippet:")
        print(dict_texts[curr_key]["text"][0:500])
        print("---\n\n")

# Print number of texts that fell under given parameters
print("Target missions:")
for curr_kobj in all_kobjs:
    print(curr_kobj)
    print("")
print("")
print("Number of valid text entries:")
print(len(dict_texts))

# Use the Operator instance to train an ML model
start = time.time()
str_err = tabby_ML.train_model_ML(dir_model=dir_model, name_model=name_model,
                                  do_reuse_run=do_reuse_run,
                                  do_check_truematch=do_check_truematch,
                                  seed_ML=seed_ML, seed_TVT=seed_TVT,
                                  dict_texts=dict_texts, mapper=mapper,
                                  buffer=buffer, fraction_TVT=fraction_TVT,
                                  mode_TVT=mode_TVT, do_shuffle=do_shuffle,
                                  do_verbose=True, do_verbose_deep=False)

print(f'Time to train the model with run = {time.time()-start} seconds.')

# Save the output error string to a file
with open(filesave_error, 'x') as openfile:
    openfile.write(str_err)
