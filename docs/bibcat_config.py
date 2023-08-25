###FILE: bibcat_config.py
###PURPOSE: Container for all user inputs for the bibcat package.
###DATA CREATED: 2023-08-17
###DEVELOPERS: (Jamila Pegues, Others)


##Import external packages
import os
#

##Global filepaths
path_json = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/BibTracking/scratchwork/dataset_combined_all.json")
#"path/to/file.json" #Path+Filename for .json file containing pre-classified texts. Each entry in the .json file should be a dictionary-style entry, containing the following key:value structure: {"text":<the text as a string>, "class":<the class name as a string>, "id":<None or <an identifier for this item>}
dir_allmodels = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/BibTracking/scratchwork") #Path to directory for saving or loading a model
name_model = "test_model_2023_08_25a" #Name of model run to save or load
#

#Classification parameters
allowed_classifications = ["SCIENCE", "DATA_INFLUENCED", "MENTION"]
#

##Machine learning (ML) parameters
ML_label_model = "categorical"
ML_activation_dense = "softmax"
ML_batch_size = 32 #32 #5 #32 #10 #5 #32
ML_model_key = "small_bert/bert_en_uncased_L-4_H-512_A-8"
ML_type_optimizer = "adamw"
ML_name_optimizer = "AdamWeightDecay"
ML_frac_dropout = 0.2 #0.1 #0.2 instead?
ML_frac_steps_warmup = 0.1
ML_num_epochs = 5 ###leave at 5; x-10 #20 #1 #8 #8 #5 #20 #5
ML_init_lr = 3E-5
#





#
