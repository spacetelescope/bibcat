"""
This module fetches text input data for classification, classifies the texts into three categories; science, mention, data-influenced. Finally, it produces performance results such as a confusion matrix.
"""
import os
import json
import numpy as np

import classes
import config
import parameters as params 

#Fetch filepaths for model and data
name_model = config.name_model
filepath_json = config.path_json
dir_model = os.path.join(config.dir_allmodels, name_model)

# Fetch filepath for output
filepath_output = config.PATH_OUTPUT

#Set directories for fetching text
dir_info = dir_model
folder_test = config.folders_TVT["test"]
dir_test = os.path.join(dir_model, folder_test)

#Set parameters for each operator and its internal classifier
#Global parameters
do_verify_truematch = True #A very important parameter - discuss with J.P. first!!!  Set it to either True or False.
do_raise_innererror = False #If True, will stop if exception encountered; if False, will print error and continue
do_verbose_text_summary = False # print input text summary

list_threshold_arrays = [np.linspace(0.5, 0.95, 20)]*2 #For uncertainty test
class_mapper = params.map_papertypes #Mapper for class types; None for no mapper

#Set some overarching global variables
seed_test = 10 #Random seed for shuffling text dataset
np.random.seed(seed_test)
do_shuffle = True #Whether or not to shuffle the text dataset
do_real_testdata = True #If True, will use real papers to test performance; if False, will use fake texts below
#
max_tests = 30 #Number of text entries to test the performance for; None for all tests available
mode_modif = "anon" #"skim_trim_anon" #None #We are using preprocessed data in this tutorial, so we do not need a processing mode at all
#
#Prepare some Keyword objects
all_kobjs = params.all_kobjs
lookup = "HST"


## Set two operators: machine learning (ML) and rule-based (RL)
#For operator ML
mapper_ML = class_mapper #Mapper to mask classifications; None if no masking
threshold_ML = 0.70 #Uncertainty threshold for this classifier
buffer_ML = 0

#For operator RL
mapper_RL = class_mapper #Mapper to mask classifications; None if no masking
threshold_RL = 0.70 #Uncertainty threshold for this classifier
buffer_RL = 0

#Gather parameters into lists
list_mappers = [mapper_ML, mapper_RL]
list_thresholds = [threshold_ML, threshold_RL]
list_buffers = [buffer_ML, buffer_RL]

## perpare papers to test on

#For use of real papers from test dataset to test on
if (do_real_testdata):
    #Load information for processed bibcodes reserved for testing
    dict_TVTinfo = np.load(os.path.join(dir_info, "dict_TVTinfo.npy"), allow_pickle=True).item()
    list_test_bibcodes = [key for key in dict_TVTinfo if (dict_TVTinfo[key]["folder_TVT"] == folder_test)]
    
    #Load the original data
    with open(filepath_json, 'r') as openfile:
        dataset = json.load(openfile)
    #
    
    #Extract text information for the bibcodes reserved for testing
    list_test_indanddata_raw = [(ii, dataset[ii]) for ii in range(0, len(dataset))
                                if (dataset[ii]["bibcode"] in list_test_bibcodes)] #Data for test set
    #
    
    #Shuffle, if requested
    if do_shuffle:
        np.random.shuffle(list_test_indanddata_raw)
    #
    
    #Extract target number of test papers from the test bibcodes
    if (max_tests is not None): #Fetch subset of tests
        list_test_indanddata = list_test_indanddata_raw[0:max_tests]
    else: #Use all tests
        list_test_indanddata = list_test_indanddata_raw
    #
    
    #Process the text input into dictionary format for inputting into the codebase
    dict_texts = {} #To hold formatted text entries
    for ii in range(0, len(list_test_indanddata)):
        curr_ind = list_test_indanddata[ii][0]
        curr_data = list_test_indanddata[ii][1]
        #
        #Convert this data entry into dictionary with: key:text,id,bibcode,mission structure
        curr_info = {"text":curr_data["body"], "id":str(curr_ind), "bibcode":curr_data["bibcode"],
                    "missions":{}}
        for curr_mission in curr_data["class_missions"]: #Iterate through missions for this paper
            for curr_kobj in all_kobjs: #Iterate through declared Keyword objects
                curr_name = curr_kobj.get_name()
                #Store mission data under keyword name, if applicable
                if (curr_kobj.is_keyword(curr_mission)):
                    curr_info["missions"][curr_name] = {"mission":curr_name,
                                                    "class":curr_data["class_missions"][curr_mission]["papertype"]}
                #
                #Otherwise, store that this mission was not detected for this text
                else:
                    curr_info["missions"][curr_name] = {"mission":curr_name, "class":config.verdict_rejection}                    
            #
        #
        #Store this data entry
        dict_texts[str(curr_ind)] = curr_info
    #
    
    #Print some notes about the testing data
    print("Number of texts in text set: {0}".format(len(dict_texts)))
    print("")
    if do_verbose_text_summary:
        for key in dict_texts:
            print("Entry {0}:".format(key))
            print("ID: {0}".format(dict_texts[key]["id"]))
            print("Bibcode: {0}".format(dict_texts[key]["bibcode"]))
            print("Missions: {0}".format(dict_texts[key]["missions"]))
            print("Start of text:\n{0}".format(dict_texts[key]["text"][0:500]))
            print("-\n")
    #
#

#Store texts for each operator and its internal classifier
#For operator ML
dict_texts_ML = dict_texts #Dictionary of texts to classify
#For operator RL
dict_texts_RL = dict_texts #Dictionary of texts to classify
#Gather into list
list_dict_texts = [dict_texts_ML, dict_texts_RL]

#Create a list of classifiers
#This can be modified to use whatever classifiers you'd like.
#initialize classifiers by loading a previously trained ML model
filepath_model = os.path.join(dir_model, (name_model+".npy"))
fileloc_ML = os.path.join(dir_model, (config.tfoutput_prefix+name_model))
classifier_ML = classes.Classifier_ML(filepath_model=filepath_model, fileloc_ML=fileloc_ML, do_verbose=True)

#Load a rule-based classifier
classifier_rules = classes.Classifier_Rules()


##Initialize operators by loading models into instances of the Operator class

#The machine learning operator
operator_ML = classes.Operator(classifier=classifier_ML, mode=mode_modif, keyword_objs=all_kobjs,
                            name="Operator_ML", do_verbose=True, load_check_truematch=True, do_verbose_deep=False)
#The rule-based operator
operator_RL = classes.Operator(classifier=classifier_rules,
                            name="Operator_RB", mode=mode_modif, keyword_objs=all_kobjs,
                            do_verbose=True, do_verbose_deep=False)
list_operators = [operator_ML, operator_RL] #Feel free to add more/less operators here.


##Run the operators
#The machine learning operator
results_ML = operator_ML.classify(text=dict_texts_ML,
                                    lookup=lookup, buffer=buffer_ML,
                                    modif=mode_modif,
                                    threshold=threshold_ML,
                                    do_raise_innererror=False,
                                    do_check_truematch=True,do_verbose_deep=False)

#The rule-based operator
results_RL = operator_RL.classify(text=dict_texts_RL,
                                    lookup=lookup, buffer=buffer_RL,
                                    modif=mode_modif,
                                    threshold=threshold_RL,
                                    do_raise_innererror=False,
                                    do_check_truematch=True, do_verbose_deep=False)

##Print the results
#Print the text and classes as a reminder
#print("Text:\n\"\n{0}\n\"\n".format(dict_texts))
print("Lookup: {0}\n".format(lookup))

#The machine learning results
print("> Machine learning results:")
print("Paragraph:\n\"\n{0}\n\"".format(results_ML["modif"]))
print("Verdict: {0}".format(results_ML["verdict"]))
print("Probabilities: {0}".format(results_ML["uncertainty"]))
print("-\n")

#The rule-based results
print("> Rule-based results:")
print("Paragraph:\n\"\n{0}\n\"".format(results_RL["modif"]))
print("Verdict: {0}".format(results_RL["verdict"]))
print("Probabilities: {0}".format(results_RL["uncertainty"]))
print("-\n")


# We will consider these performance tests:
# * Basic: We generate confusion matrices for the set of Operators (containing the different classifiers).
# * Uncertainty: We plot performance as a function of uncertainty level for the set of Operators.

#Create an instance of the Performance class
performer = classes.Performance()

#Parameters for this evaluation
fileroot_evaluation = "test_eval_basic" #Root name of the file within which to store the performance evaluation output
fileroot_misclassif = "test_misclassif_basic" #Root name of the file within which to store misclassified text information
figsize = (20, 12)

#Run the pipeline for a basic evaluation of model performance
performer.evaluate_performance_basic(operators=list_operators, dicts_texts=list_dict_texts, mappers=list_mappers,
                                     thresholds=list_thresholds, buffers=list_buffers, is_text_processed=False,
                                     do_verify_truematch=do_verify_truematch, do_raise_innererror=do_raise_innererror,
                                     do_save_evaluation=True, do_save_misclassif=True, filepath_output=filepath_output,
                                     fileroot_evaluation=fileroot_evaluation, fileroot_misclassif=fileroot_misclassif,
                                     print_freq=1, do_verbose=True, do_verbose_deep=False, figsize=figsize)

#Parameters for this evaluation
fileroot_evaluation = "test_eval_uncertainty" #Root name of the file within which to store the performance evaluation output
fileroot_misclassif = "test_misclassif_uncertainty" #Root name of the file within which to store misclassified text information
figsize = (40, 12)

#Run the pipeline for an evaluation of model performance as a function of uncertainty
performer.evaluate_performance_uncertainty(operators=list_operators, dicts_texts=list_dict_texts, mappers=list_mappers,
                                     threshold_arrays=list_threshold_arrays, buffers=list_buffers,
                                     is_text_processed=False,
                                     do_verify_truematch=do_verify_truematch, do_raise_innererror=do_raise_innererror,
                                     do_save_evaluation=True, do_save_misclassif=True, filepath_output=filepath_output,
                                     fileroot_evaluation=fileroot_evaluation, fileroot_misclassif=fileroot_misclassif,
                                     print_freq=25, do_verbose=True, do_verbose_deep=False, figsize=figsize)

