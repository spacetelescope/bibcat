###FILE: bibcat_config.py
###PURPOSE: Container for all user inputs for the bibcat package.
###DATA CREATED: 2023-08-17
###DEVELOPERS: (Jamila Pegues, Others)


##Import external packages
import os
SRC_ROOT = os.path.dirname(__file__)
_parent = os.path.dirname(SRC_ROOT)
print(f"Root directory ={SRC_ROOT}, parent directory={_parent}")

PATH_MODELS = os.path.join(SRC_ROOT, "models")
PATH_CONFIG = os.path.join(SRC_ROOT, "config")
PATH_DOCS = os.path.join(_parent, "docs")
PATH_OUTPUT = os.path.join(_parent,"output")

if not os.path.isdir(PATH_MODELS):
    os.makedirs(PATH_MODELS)
    print("created folder : ", PATH_MODELS)
else:
    print(PATH_MODELS, "folder already exists.")

if not os.path.isdir(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)
    print("created folder : ", PATH_OUTPUT)
else:
    print(PATH_OUTPUT, "folder already exists.")



#
KW_AMBIG = os.path.join(PATH_CONFIG, "keywords_ambig.txt")
PHR_AMBIG = os.path.join(PATH_CONFIG, "phrases_ambig.txt")

##Global filepaths
#path_json = "path/to/dataset_combined_all.json" # set the path to the location of the JSON file you downloaded from Box
#path_json ="/Users/jyoon/asb/bibliography_automation/bibcat_datasets/dataset_combined_all.json"
path_json = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/BibTracking/scratchwork/dataset_combined_all.json")
#"path/to/file.json" #Path+Filename for .json file containing pre-classified texts. Each entry in the .json file should be a dictionary-style entry, containing the following key:value structure: {"text":<the text as a string>, "class":<the class name as a string>, "id":<None or <an identifier for this item>}

#Below are for bibcat_tests.py testing purposes only
filepath_input = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/BibTracking/Datasets")
path_papertrack = os.path.join(filepath_input, "papertrack_export_2023-11-06.json")
path_papertext = os.path.join(filepath_input, "ST_Request2021_use.json")
#

dir_allmodels = PATH_MODELS #Path to directory for saving or loading a model

name_model = "my_test_run" #Name of model run to save or load
#name_model ="lamb_real_run"
#
path_TVTinfo = os.path.join(dir_allmodels, name_model, "dict_TVTinfo.npy")
#

#Classification parameters
allowed_classifications = ["SCIENCE", "DATA_INFLUENCED", "MENTION"]
#

##Machine learning (ML) parameters
ML_label_model = "categorical"
ML_activation_dense = "softmax"
ML_batch_size = 32 #32 #5 #32 #10 #5 #32
ML_model_key = "small_bert/bert_en_uncased_L-4_H-512_A-8"
ML_type_optimizer = "lamb" #"adamw"
ML_name_optimizer = "LAMB" #"AdamWeightDecay"
ML_frac_dropout = 0.2 #0.1 #0.2 instead?
ML_frac_steps_warmup = 0.1
ML_num_epochs = 5 ###leave at 5; x-10 #20 #1 #8 #8 #5 #20 #5
ML_init_lr = 3E-5
#





#
