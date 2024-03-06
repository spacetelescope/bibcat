###FILE: bibcat_config.py
###PURPOSE: Container for all user inputs for the bibcat package.
###DATA CREATED: 2023-08-17
###DEVELOPERS: (Jamila Pegues, Others)


##Import external packages
import os
import nltk
from nltk.corpus import wordnet
#

##Set global user paths
#path_json = "path/to/dataset_combined_all.json" # set the path to the location of the JSON file you downloaded from Box
path_json = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/Bibtracking/datasets/dataset_combined_all.json") # set the path to the location of the JSON file you downloaded from Box
#name_model = "run_full_none_uniform" #Name of model run to save or load
#name_model = "run_full_anon_uniform" #Name of model run to save or load
#name_model = "test_run_rule" #"run_full_skim_trim_anon_uniform" #! "run_class2_full_none_uniform" #Name of model run to save or load
#


#!!!
mode_modif = "none" #"skim_anon" #"skim_trim_anon" #None
#name_model = "perf_run_ML_{0}".format(mode_modif)
name_model = "perf_run_MLandRB_{0}".format(mode_modif)
#name_model = "perf_run_RB_{0}".format(mode_modif)
#!!!

##Set global fixed paths
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

KW_AMBIG = os.path.join(PATH_CONFIG, "keywords_ambig.txt")
PHR_AMBIG = os.path.join(PATH_CONFIG, "phrases_ambig.txt")
#

##Set global input and output paths
dir_allmodels = PATH_MODELS #Path to directory for saving or loading a model
path_modiferrors = os.path.join(dir_allmodels, name_model, "dict_modiferrors.npy")
path_TVTinfo = os.path.join(dir_allmodels, name_model, "dict_TVTinfo.npy")
tfoutput_prefix = "tfoutput_"
folders_TVT = {"train":"dir_train", "validate":"dir_validate", "test":"dir_test"}

#Below are for bibcat_tests.py testing purposes only
filepath_input = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/Bibtracking/datasets") #"/path/to/the/dataset"
path_papertrack = os.path.join(filepath_input, "papertrack_export_2023-11-06.json")
path_papertext = os.path.join(filepath_input, "ST_Request_2018_to_2023_use.json")
#

##Set text processing and regular expression variables
#Placeholders for replacing various substrings in text
placeholder_anon = "OBJ"
placeholder_numeric = "numeric"
placeholder_number = "000"
placeholder_website = "website"
placeholder_author = "Authorsetal"
#
#Regular expressions: acronym matching
exp_acronym_midwords = r" +(?:(?:(?:for )|(?:the )|(?:in )|(?:of )|(?:from ))*)\b"
#
#Regular expressions: punctuation
exp_nopunct = "[^\w\s]" #For removing punctuation from strings
exp_punctuation = ["\.",",",";","\:","\?","\!"] #For matching open brackets in string
set_punctuation = [".",",",";",":","?","!"] #For matching open brackets in string
set_apostrophe = ["'"] #For matching apostrophes
set_openbrackets = ["(","[","{","<"] #For matching open brackets in string
set_closebrackets = [")","]","}",">"] #For matching close brackets in string
#
#Regular expressions: split text
exp_splittext = "(?<=.[.?!]) +(?=[A-Z])" #Splits text into rough sentences; based on stackoverflow post, heh: Avinash Raj answer from Sep 9 2014 from https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
exp_splitbracketstarts = "(?<=.[.?!]) +(?=\\"+"|\\".join(set_openbrackets)+")" #Splits bracketed sentences into separate sentences
exp_splitbracketends = "(?<=.[.?!]["+("\\"+"\\".join(set_closebrackets))+"]) +(?=[A-Z])" #Splits bracketed sentences into separate sentences
#
#Regular expressions: synsets
exp_synset = "[A-Z|a-z]+\.[a-z]\.[0-9]{2,2}" #Regular expression to match to wordnet synsets
#
#Regular expressions: text cleansing
exp_website = (r"\b(http(s?):(//)?)?(www\.)?[A-Z|a-z|-]*"
                +r"(\.[A-Z|a-z|0-9]+)*" #More of domain
                +r"(\.[A-Z|a-z]{2,})" #End of domain
                +r"((/|\?|=)[A-Z|a-z|0-9]+)*\b\/?") #Slash, etc, after domain
exp_etal_cleansed = r"\b[A-Z][A-Z|a-z]+etal\b"
dict_exp_abbrev = {r"\bFig\b\.":"Figure", r"\bTab\b\.":"Table", r"\bvs\b\.":"vs"}
#

##Set Grammar variables
#For important pos flags
trail_pos_main = ["VERB"]
special_pos_main = ["SUBJECT", "PREPOSITION_SUBJECT", "VERB", "DIRECT_OBJECT", "PREPOSITION", "PREPOSITION_OBJECT", "AUX"]
ignore_pos_main = ["USELESS", "PUNCTUATION", "ADJECTIVE", "MARKER", "CONJUNCTION"]
#
#For part-of-speech (pos) identification
#Conjoined
dep_conjoined = ["conj"]
#Conjunction
pos_conjunction = ["CCONJ"]
tag_conjunction = ["CC"]
#Roots
dep_root = ["ROOT"]
#Useless words
dep_useless = ["det", "amod", "advmod"]
pos_useless = ["DET", "ADJ", "ADV"]
tag_useless = ["CD", "JJ", "JJR", "JJS", "LS", "PDT", "RB", "RBR", "RBS", "UH", "WRB", "RP"]
#Verbs
dep_verb = ["verb", "advcl", "relcl", "acl"]
dep_verb_passive = ["pass", "auxpass"]
pos_verb = ["VERB"]
tag_verb_past = ["VBD", "VBN"]
tag_verb_present = ["VB", "VBP", "VBZ", "VBG"]
tag_verb_future = ["MD"]
tag_verb_purpose = ["TO"]
tag_verb_any = tag_verb_past + tag_verb_present + tag_verb_future
#Subjects
dep_subject = ["nsubj", "nsubjpass", "expl"]
#Objects
#dep_object = ["dobj", "pobj", "attr"] #, "appos"]
dep_object = ["dobj", "pobj", "attr", "compound"] #, "appos"]
#Appositional Modifiers
dep_appos = ["appos"]
#Clausal Complements
dep_ccomp = ["ccomp"]
#Prepositions
dep_preposition = ["prep", "agent", "dative"]
pos_preposition = ["ADP"]
tag_preposition = ["IN"]
#Aux
dep_aux = ["aux", "auxpass"]
pos_aux = ["AUX", "PART"] #, "PART"]
#Determinants
pos_determinant = ["DET", "PRON"]
tag_determinant = ["PDT", "DT", "WDT", "EX"]
#Nouns
pos_noun = ["PROPN", "NOUN", "PRON"]
#Pronouns
pos_pronoun = ["PRON"]
tag_pronoun = ["PRP", "PRP$", "WP", "WP$"]
#Adjectives
dep_adjective = ["amod"]
pos_adjective = ["ADJ"]
tag_adjective = ["JJ", "JJR", "JJS"]
#Markers
dep_marker = ["mark"]
tag_marker = ["WRB", "WDT"]
#Negatives
dep_negative = ["neg"]
#Numbers
pos_number = ["NUM"]
tag_number = ["CD"]
#Unidentified words
dep_xpos = ["X"]
pos_xpos = ["X"]
#Brackets
tag_brackets = ["-LRB-", "-RRB-"]
#Possessive
tag_possessive = ["POS"]
#Punctuation
dep_punctuation = ["punct"]
#
#
dict_conv_pos = {"prep_objects":"NOUN", "dir_objects":"NOUN", "subjects":"NOUN", "auxs":"AUX"}
#

##Set natural language processing (NLP) variables
spacy_language_model = "en_core_web_sm" #Simpler language model
#
#For natural language processing
nlp_lookup_person = "Person" #To look up e.g. 1st,3rd person status of pronoun
#
#For ambiguous phrase processing
string_anymatch_ambig = "anymission" #Mission marker in ambig. phrase database to match to any mission
string_numeral_ambig = "000" #String representation of given numeral word
#Spacy language model
#
#Special synsets
special_synsets_fig = ["table.n.01", "tab.n.04", "figure.n.01", "section.n.01", "chapter.n.01", "appendix.n.01"] #Includes synset for 'Tab', which can be 'Table' abbreviated
#

##Set machine learning variables
ML_label_model = "categorical"
ML_activation_dense = "softmax"
ML_batch_size = 32
ML_model_key = "small_bert/bert_en_uncased_L-4_H-512_A-8" #Simpler language model
#ML_model_key = "bert_en_uncased_L-12_H-768_A-12" #Fancier language model
ML_type_optimizer = "lamb" #"adamw"
ML_name_optimizer = "LAMB" #"AdamWeightDecay"
ML_frac_dropout = 0.2
ML_frac_steps_warmup = 0.1
ML_num_epochs = 20 #50 #15 #10 #5
ML_init_lr = 3E-5
#
"""
dict_ml_model_encoders = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}
dict_ml_model_preprocessors = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}
#"""
#
dict_ml_model_encoders = {
"small_bert/bert_en_uncased_L-4_H-512_A-8":"https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/bert-en-uncased-l-4-h-512-a-8/versions/1"
}
dict_ml_model_preprocessors = {
"small_bert/bert_en_uncased_L-4_H-512_A-8":"https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3"
}
#

##Classification
#For custom classification verdicts
verdict_error = "z_error"
verdict_lowprob = "z_lowprob"
verdict_rejection = "z_notmatch"
list_other_verdicts = [verdict_error, verdict_lowprob, verdict_rejection]
#
#For preset custom verdict outputs
dictverdict_lowprob = {
        "verdict":verdict_lowprob, "scores_comb":None,
        "scores_indiv":None, "uncertainty":None}
dictverdict_rejection = {
        "verdict":verdict_rejection, "scores_comb":None,
        "scores_indiv":None, "uncertainty":None}
dictverdict_error = {
        "verdict":verdict_error, "scores_comb":None,
        "scores_indiv":None, "uncertainty":None}
#
#For rule-based classification
list_default_verdicts_decisiontree = ["science", "data_influenced", "mention"]
#dict_tree_possible_values = {"subjectmatter":["is_pron_1st", "is_etal", "is_keyword", "is_term_fig", "is_pron_3rd"], "objectmatter":["is_pron_1st", "is_etal", "is_keyword", "is_term_fig", "is_pron_3rd"], "verbclass":["be", "has", "know", "plot", "root_nonverb", "science", "datainfluenced"], "verbtypes":["FUTURE", "PASSIVE", "PAST", "POTENTIAL", "PURPOSE", "PRESENT"]}
#dict_tree_possible_values = {"subjectmatter":["is_pron_1st", "is_etal", "is_keyword", "is_term_fig", "is_pron_3rd"], "objectmatter":["is_pron_1st", "is_etal", "is_keyword", "is_term_fig", "is_pron_3rd"], "verbclass":["be", "has", "know", "plot", "science", "datainfluenced"], "verbtypes":["FUTURE", "PAST", "PURPOSE", "PRESENT"]}
dict_tree_possible_values = {"allmatter":["is_pron_1st", "is_etal", "is_keyword", "is_term_fig", "is_pron_3rd"], "verbclass":["be", "has", "know", "plot", "science", "datainfluenced"], "verbtypes":["FUTURE", "PAST", "PURPOSE", "PRESENT"]}
#

##Set rule-based processing variables
#Rule-based thresholds
thres_category_fracdiff = 0.1
thres_verbsimilaritymain = 0.75 #25 #Threshold of similarity to say two verbs are similar
thres_verbsimilarityhigh = 0.75 #Threshold of similarity to say two verbs are similar
#
#Grammar nest generation
conv_pos_fromtreetonest = {"PREPOSITION_SUBJECT":"subjectmatter", "SUBJECT":"subjectmatter", "DIRECT_OBJECT":"objectmatter", "PREPOSITION_OBJECT":"objectmatter"} #Convert from tree to nest terminology
nest_important_treebools = ["is_pron_1st", "is_keyword", "is_etal", "is_pron_3rd", "is_term_fig"] #List of important booleans to record into a post-grammar nest
nest_important_treebools_superior = ["is_pron_1st", "is_term_fig", "is_etal"] #List of important booleans that get merged into main component from linked terms in a nest
nest_unimportant_pos = [None, "PUNCTUATION"] #Any parts-of-speech (pos) that will be ignored+skipped during nest-building
#nest_keys_main = ["subjectmatter", "objectmatter", "verbclass", "verbtypes"] #List of main keywords that define a nest
#nest_keys_matter = ["subjectmatter", "objectmatter"] #List of matter keywords that define a nest
nest_keys_main = ["allmatter", "verbclass", "verbtypes"] #List of main keywords that define a nest
nest_keys_matter = ["allmatter"] #List of matter keywords that define a nest
nest_key_verbtype = "verbtypes" #verbtype keyword
nest_prefix_link = "link_" #Prefix for linked keys within nest
max_num_hypernyms = 3
#
#Verb synsets
synsets_verbs_be = ["be.v.01"]
synsets_verbs_be = [wordnet.synset(s) for s in synsets_verbs_be]
synsets_verbs_has = ["have.v.01"]
synsets_verbs_has = [wordnet.synset(s) for s in synsets_verbs_has]
synsets_verbs_science = ["use.v.01", "diagram.v.01", "get.v.01", "analyze.v.01", "detect.v.01", "determine.v.03", "classify.v.01", "correct.v.01", "target.v.01", "estimate.v.01", "perform.v.01"]
synsets_verbs_science = [wordnet.synset(s) for s in synsets_verbs_science]
synsets_verbs_influenced = ["imitate.v.01", "model.v.05", "simulate.v.03", "compare.v.01", "discuss.v.01", "predict.v.01", "estimate.v.01"]
synsets_verbs_influenced = [wordnet.synset(s) for s in synsets_verbs_influenced]
synsets_verbs_plot = ["plot.v.01", "show.v.01", "exemplify.v.02", "highlight.v.02", "correlate.v.01", "indicate.v.02", "choose.v.01", "represent.v.01"]
synsets_verbs_plot = [wordnet.synset(s) for s in synsets_verbs_plot]
synsets_verbs_know = ["know.v.01", "understand.v.01", "want.v.02", "acknowledge.v.01"]
synsets_verbs_know = [wordnet.synset(s) for s in synsets_verbs_know]
#
#Verb categorization
#category_nonverb_root = "root_nonverb"
list_category_names = ["science", "datainfluenced", "plot", "know"]
list_category_synsets = [synsets_verbs_science, synsets_verbs_influenced, synsets_verbs_plot, synsets_verbs_know]
list_category_threses = [thres_verbsimilaritymain, thres_verbsimilaritymain, thres_verbsimilaritymain, thres_verbsimilarityhigh]
#
