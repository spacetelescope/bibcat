###FILE: bibcat_text.py
###PURPOSE: Container for all functions/classes used to process/prepare text for testing/using the BibCat package.
###DATE CREATED: 2022-03-20
###DEVELOPERS: (Jamila Pegues, Others)


##Below Section: Imports necessary functions
import numpy as np
from scipy import stats as scipystats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close()
import os
import re
import json
import cmasher
import copy as copier
import tensorflow as tf
from official.nlp import optimization as tf_opt
seed_gen = 8
seed_ML = 9
seed_TVT = 10
np.random.seed(seed_gen)
#
import bibcat_classes as bibcat
import bibcat_constants as preset
#
keys_classes = ["bibcode", "name_search", "is_falsesearch", "papertype", "mission", "year_entry"] #preset.dataset_ordered_keys_classes
classifys_textforms = ["none", "anon", "skim", "skim_anon", "trim_anon", "skim_trim_anon"]
classifys_TVTs = ["train", "validate", "test"]
main_textform = "none" #"full_exact" #"FULL"
full_textform = "none" #"full_exact" #"FULL"
#

##Below Section: Sets global variables
#Global booleans
do_prepare_dataset = False #If True, prepares+saves dataset for classification
do_prepare_paragraphs = False #If True, prepares+saves paragraphs for classification
#
do_gen_dir_TVT = False #If True, generate copied directory of TVT statements
do_train_ML_model = False #If True, will train machine learning model based on global parameters
#
do_extract_keyword_phrases = False #If True, will extract unique list of keyword phrases from collection of papers
do_extract_missing_branches = False #If True, will determine and extract missing branches from internal bibcat decision tree
#
if True in [do_prepare_dataset, do_prepare_paragraphs, do_gen_dir_TVT, do_train_ML_model, do_extract_keyword_phrases]:
    do_save = False
#
#ADD BUFFER ONCE TESTS DONE; Fetch old code, fix just-test-statement issue, run to see if something else changed to affect performance; try different ML parameters and better ML BERT model; fix text processing issues finally; refresh/reread up on BERT
do_evaluate_pipeline_performance = True #If True, will evaluate full pipeline from raw text to rejection/verdict
do_evaluate_pipeline_probability = False #If True, will evaluate full pipeline from raw text to rejection/verdict over different probability values (to generate, e.g., ROC curve)
do_evaluate_pipeline_falsehit = False #If True, will evaluate full pipeline with focus on false-positives
#
if True in [do_evaluate_pipeline_performance, do_evaluate_pipeline_probability, do_evaluate_pipeline_falsehit, do_extract_keyword_phrases]:
    max_papers = 500 #00 #None #500 #100 #500 #200 #0 #None #200 #100#0 #None
    textform_new = "trim" #"none"
    textform_ML = "trim_vague" #"trim_vague", "skim_anon" ###! #"skim_anon" #"trim_exact" #"trim_anon" #"skim_exact" #"trim_vague" #"full_exact" #"full_anon" #"skim_anon" #"trim_anon" #"FULL" #preset.anon_textform
#
spec_id = None
which_map = "map_3exact" #"map_2exact" "map_2incl" "map_3exact"
which_set = "set8a" #"ACS" #"set6a" "set8a" #"orig" "HSTandTESS" #"HST"
which_scheme_TVT = "minimum" #"minimum", "individual"
#


#
if (which_map == "map_3exact"): #Distinct: science, data_infl., mention
    map_papertypes = {"SCIENCE":"SCIENCE", "MENTION":"MENTION", "SUPERMENTION":"MENTION", "DATA_INFLUENCED":"DATA_INFLUENCED", "UNRESOLVED_GREY":None, "ENGINEERING":None, "INSTRUMENT":None}
elif (which_map == "map_2exact"): #Distinct: science, mention
    map_papertypes = {"SCIENCE":"SCIENCE", "MENTION":"MENTION", "SUPERMENTION":"MENTION", "DATA_INFLUENCED":None, "UNRESOLVED_GREY":None, "ENGINEERING":None, "INSTRUMENT":None}
elif (which_map == "map_2incl"): #Inclusive: science, (mention+data_infl.)
    map_papertypes = {"SCIENCE":"SCIENCE", "MENTION":"MENTION", "SUPERMENTION":"MENTION", "DATA_INFLUENCED":"MENTION", "UNRESOLVED_GREY":None, "ENGINEERING":None, "INSTRUMENT":None}
else:
    raise ValueError("Whoa! Map scheme {0} not recognized!".format(which_map))
#

#Global files
filepath_base_global = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Functional/Library/BibTracking")
filepath_tests_global = os.path.join(filepath_base_global, "tests")
filepath_models_global = os.path.join(filepath_base_global, "models")
#
filename_classifs = "./classifs.json"
filename_bibcodes = "./bibcodes.txt"
filename_json = "./../../datasets/dataset_combined_all.json"
filename_bibnotintrack = "./../../datasets/bibcodes_notin_papertrack.txt"
dir_forsave_root = "./../../trials/statements_labeled_all_{0}".format(which_map)
dir_forTVT_root = "./../../trials/statements_labeled_all_TVTseed{1}_{0}".format(which_map, seed_TVT)
base_forsave = "statement"
exp_statement = "{0}_.*\.txt".format(base_forsave)
#
#keyword_obj_HST = bibcat.Keyword(
#            keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
#            acronyms=["HST", "HT"], type_acronyms=None,
#            mismatches=["Edwin Hubble", "Hubble Constant", "Hubble Time", "Hubble Parameter", "Hubble Parameters", "Hubble Expansion", "Hubble Flow", "Hubble Distance", "Hubble Fellowship", "grant HST", "Hubble Scale", "Hubble Type", "Hubble Types", "Hubble Function", "Hubble Scalar", "Hubble Tension", "Hubble Diagram", "Hubble Rate", "Hubble Friction", "Hubble Factor", "Hubble Equation", "Hubble Sequence", "Hubble Law", "Hubble's Constant", "Hubble's Tuning Fork", "Hubble 1926", "Hubble Radius", "Hubble Tuning", "Hubble (Hubble 1936)"])
keyword_obj_HST = bibcat.Keyword(
            keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
            acronyms=["HST", "HT"])#,
            #mismatches=["Edwin Hubble", "Hubble Constant", "Hubble Time", "Hubble Parameter", "Hubble Expansion", "Hubble Flow", "Hubble Distance", "Hubble Fellowship", "grant HST", "Hubble Scale", "Hubble Type", "Hubble Function", "Hubble Scalar", "Hubble Tension", "Hubble Diagram", "Hubble Rate", "Hubble Friction", "Hubble Factor", "Hubble Equation", "Hubble Sequence", "Hubble Law", "Hubble's Constant", "Hubble's Tuning Fork", "Hubble 1926", "Hubble Radius", "Hubble Tuning", "Hubble (Hubble 1936)"])
keyword_obj_ACS = bibcat.Keyword(
            keywords=["Advanced Camera for Surveys"],
            acronyms=["ACS"])
keyword_obj_TESS = bibcat.Keyword(
            keywords=["Transiting Exoplanet Survey Satellite"],
            acronyms=["TESS"])
keyword_obj_JWST = bibcat.Keyword(
            keywords=["James Webb Space Telescope", "James Webb Telescope", "Webb Space Telescope", "Webb Telescope"],
            acronyms=["JWST", "JST", "JT"])
keyword_obj_Kepler = bibcat.Keyword(
            keywords=["KEPLER"],
            acronyms=[])
keyword_obj_PanSTARRS = bibcat.Keyword(
            keywords=["Panoramic Survey Telescope and Rapid Response System", "Pan-STARRS", "Pan-STARRS1"],
            acronyms=["PanSTARRS", "PanSTARRS1", "PS1"])
keyword_obj_GALEX = bibcat.Keyword(
            keywords=["Galaxy Evolution Explorer"],
            acronyms=["GALEX"])
keyword_obj_K2 = bibcat.Keyword(
            keywords=["K2"],
            #acronyms=["K2"], type_acronyms=None,
            acronyms=[])
keyword_obj_HLA = bibcat.Keyword(
            keywords=["Hubble Legacy Archive"],
            acronyms=["HLA"])
#
skip_bibcodes = ["2021MNRAS.500..942D", #No HST keywords, just COSMOS and CANDELS
                "2020PASJ...72...86H", #HST keyword only in footnote (which didn't appear in its text), just COSMOS
                "2020MNRAS.497.3273F", #No HST keywords, but COSMOS and CANDELS
                "2020MNRAS.499.5107R", #HST keyword in table - table not in delivered .json?
                "2020MNRAS.495.3252S", #No HST keywords, but COSMOS
                "2020SSRv..216...32F", #Super-annoying table to deal with later
                "2020SSRv..216..139S", #Strange case to deal with later
                "2020SPIE11439E..1SZ", #Strange quotations resisting cleaning
                "2021MNRAS.500.5142F", #Hubble constant mentioned, but HST MENTION; not sure why
                #"2021MNRAS.500.4849C", #Marked as Kepler and K2, but only K2 keyword seen in text?
                "2020Sci...367..577V", #Bibcode not in papertrack
                #"2020PASJ...72...26M", #Stated as PanSTARRS, but internal keyword unclear?
                #"2020PhRvD.101h3008W", #Stated as Kepler, but internal keyword unclear?
                #"2020PASA...37...53S", #Stated as HLA, but internal keyword unclear?
                "2020NatAs...4..377N", #Bibcode not in papertrack
                "2019ChA&A..43..143L", #Bibcode not in papertrack
                "2019ChA&A..43..342C", #Bibcode not in papertrack
                "2019ChA&A..43..199Z", #Bibcode not in papertrack
                "2019ChA&A..43..217W", #Bibcode not in papertrack
                "2019ChA&A..43..444P", #Bibcode not in papertrack
                "2019A&A...621A..90G", #Bibcode not in papertrack
                "2018ChA&A..42..487Z", #Bibcode not in papertrack
                "2018ChA&A..42..609Y", #Bibcode not in papertrack
                "2020Galax...8...71P", #No text for this paper?
                "2020Galax...8...49K", #No text for this paper?
                "2020Galax...8...51L", #No text for this paper?
                "2019Sci...365.1418H", #No text for this paper?
                "2019Sci...364..480G", #No text for this paper?
                "2019Sci...363.1258M", #No text for this paper?
                "2019Sci...364..438M", #No text for this paper?
                "2019RMxAC..51..131E", #No text for this paper?
                "2019RMxAA..55...73Y", #No text for this paper?
                "2019RMxAA..55..177E", #No text for this paper?
                "2019Sci...363.1041R", #No text for this paper?
                "2019Sci...363S1052P", #No text for this paper?
                "2019Sci...366..302O", #No text for this paper?
                "2019RMxAA..55..117R", #No text for this paper?
                "2019Sci...364.1020C", #No text for this paper?
                "2019ARep...63..998B", #No text for this paper?
                "2019AstL...45...81B", #No text for this paper?
                "2019BAAS...51g..32R", #No text for this paper?
                "2019BAAS...51c.413R", #No text for this paper?
                "2019BAAS...51c..44C", #No text for this paper?
                "2019BAAS...51g..83H", #No text for this paper?
                "2019BAAS...51c.418E", #No text for this paper?
                "2019BAAS...51c..22H", #No text for this paper?
                "2019BAAS...51c..45R", #No text for this paper?
                "2019BAAS...51c.201R", #No text for this paper?
                "2019BAAS...51c.269B", #No text for this paper?
                "2019BAAS...51c..11R", #No text for this paper?
                "2019BAAS...51c..58B", #No text for this paper?
                "2019Ap.....62..513S", #No text for this paper?
                "2019Ap.....62..177S", #No text for this paper?
                "2019Ap.....62..518A", #No text for this paper?
                "2018JAI.....740005H", #No text for this paper?
                "2018JAI.....740012E", #No text for this paper?
                "2018JAI.....740006P", #No text for this paper?
                "2018Sci...362..895Y", #No text for this paper?
                "2018Sci...362.1230C", #No text for this paper?
                "2018Sci...362.1354W", #No text for this paper?
                "2018AstL...44..735M", #No text for this paper?
                "2020ApJ...894..115P" #Weird text R.A. processing error
                #"2020PASA...37...26D", #Stated as GALEX, but internal keyword unclear?
                #"2020MNRAS.498..235H", #Stated as HST, but internal keyword unclear
                #"2020Natur.577...39W", #Stated as HLA, but internal keyword unclear
                #"2020MNRAS.498.2030C" #Stated as PanSTARRS, but internal keyword unclear
                ]
#Figure names
figpath_pipeline = "./../../plots/performance_pipeline"
#
if which_set == "orig":
    target_missions = ["HST"]
    keyword_obj_list = [keyword_obj_HST]
    which_specificacronym = None
    buffer = 0 #None
#
elif which_set == "HST":
    target_missions = ["HST"]
    keyword_obj_list = [keyword_obj_HST]
    which_specificacronym = None
    buffer = 0
#
elif which_set == "ACS":
    target_missions = ["HST"] # , ""]
    keyword_obj_list = [keyword_obj_ACS]
    buffer = 0
    which_specificacronym = "ACS"
#
elif which_set == "KeplerandK2":
    target_missions = ["KEPLER", "K2"] # , ""]
    keyword_obj_list = [keyword_obj_Kepler, keyword_obj_K2]
    buffer = 0
    which_specificacronym = None
#
elif which_set == "HSTandTESS":
    target_missions = ["HST", "TESS"]
    keyword_obj_list = [keyword_obj_HST, keyword_obj_TESS]
    which_specificacronym = None
    buffer = 0
#
elif which_set == "set4a":
    target_missions = ["HST", "TESS", "JWST", "PanSTARRS"]
    keyword_obj_list = [keyword_obj_HST, keyword_obj_TESS, keyword_obj_JWST, keyword_obj_PanSTARRS]
    which_specificacronym = None
    buffer = 0
#
elif which_set == "set6a":
    target_missions = ["HST", "TESS", "JWST", "PanSTARRS", "GALEX", "HLA"]
    keyword_obj_list = [keyword_obj_HST, keyword_obj_TESS, keyword_obj_JWST, keyword_obj_PanSTARRS, keyword_obj_GALEX, keyword_obj_HLA]
    which_specificacronym = None
    buffer = 0
#
elif which_set == "set8a":
    target_missions = ["HST", "TESS", "JWST", "PanSTARRS", "GALEX", "HLA", "KEPLER", "K2"]
    keyword_obj_list = [keyword_obj_HST, keyword_obj_TESS, keyword_obj_JWST, keyword_obj_PanSTARRS, keyword_obj_GALEX, keyword_obj_HLA, keyword_obj_Kepler, keyword_obj_K2]
    which_specificacronym = None
    buffer = 0
#
else:
    raise ValueError("Whoa! Set {0} not allowed!".format(which_set))
#
dir_ext = "buffer_{0}".format(buffer)
dir_forsave = "{0}_{1}/".format(dir_forsave_root, dir_ext)
dir_fortext = dir_forsave
dir_forTVT_global = "{0}_{1}_{2}/".format(dir_forTVT_root, dir_ext, which_scheme_TVT)
#
additional_verdicts_meas = [item.upper() for item in [preset.verdict_error, preset.verdict_lowprob, preset.verdict_rejection]]
additional_verdicts_act = [item.upper() for item in [preset.verdict_rejection]]
set_verdicts_errors_global = [item.upper() for item in [preset.verdict_error, preset.verdict_lowprob]]
#

###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###QUICK CHECKS
if True:
    #Verify that target missions and keywords match in number
    if (len(target_missions) != len(keyword_obj_list)):
        raise ValueError("Whoa! Mismatch # of missions vs keyobjs!: {0} vs {1}"
                        .format(target_missions, keyword_obj_list))
    #
    #Verify that target missions and keywords match in order
    for ii in range(0, len(target_missions)):
        if ((target_missions[ii] not in keyword_obj_list[ii]._get_info("keywords"))
                and (target_missions[ii] not in
                        keyword_obj_list[ii]._get_info("acronyms"))):
            raise ValueError("Whoa! Mismatched missions vs keyobjs!: {0} vs {1}"
                        .format(target_missions, keyword_obj_list))
    #
    ii = None
#
###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###FUNCTIONS: TEXT PROCESSING
#Procedure to generate dictionary dataset from raw papertrack and papertext data
def generate_dataset(filename_papertrack, filename_papertext, filesave_json, filesave_notinpapertrack, keys_papertrack, keys_papertext, do_save=False, do_verbose=False):
    #Load paper texts
    with open(filename_papertext) as openfile:
        dataset_papers_orig = json.load(openfile)
    #Load paper classes
    dataset_classes_orig = np.genfromtxt(filename_papertrack, delimiter=",",
                                        skip_header=1, dtype=str)
    #Extract bibcodes from each set
    bibcodes_papers = [item["bibcode"] for item in dataset_papers_orig]
    dates_papers = [item["pubdate"] for item in dataset_papers_orig]
    min_year_papers = min(dates_papers)
    max_year_papers = max(dates_papers)
    bibcodes_classes = dataset_classes_orig[:,keys_classes.index("bibcode")]
    missions_classes = dataset_classes_orig[:,keys_classes.index("mission")]
    papertypes_classes = dataset_classes_orig[:,keys_classes.index("papertype")]
    is_falsepos_classes = dataset_classes_orig[:,
                                        keys_classes.index("is_falsesearch")]

    #Print some notes
    if do_verbose:
        print("Min. date of papers within text database: {0}."
                .format(min_year_papers))
        print("Max. date of papers within text database: {0}."
                .format(max_year_papers))
    #

    #Trim papertrack dictionary down to only columns to include
    try:
        storage = [{key:value for key,value in thisdict.items()
                                if (key in keys_papertext)}
                                for thisdict in dataset_papers_orig]
    except AttributeError: #If this error raised, probably earlier Python vers.
        storage = [{key:value for key,value in thisdict.iteritems()
                                if (key in keys_papertext)}
                                for thisdict in dataset_papers_orig]
    #

    #Verify that all papers within papertrack are within the papertext database
    if any([val not in bibcodes_papers for val in np.unique(bibcodes_classes)]):
        tmp_vals = [val for val in np.unique(bibcodes_classes)
                        if (val not in bibcodes_papers)]
        errstr = ("Note! Papers in papertrack not in text database!"
                    +"\n{0}\n{1} of {2} in all."
                    .format(tmp_vals, len(tmp_vals), len(bibcodes_papers)))
        #raise ValueError(errstr)
        print(errstr)
    #

    #Iterate through paper dictionary
    num_notin_papertrack = 0
    bibcodes_notin_papertrack = []
    for ii in range(0, len(storage)):
        #Extract information for current paper within text database
        curr_dict = storage[ii] #Current dictionary
        curr_bibcode = curr_dict["bibcode"]

        #Extract index for current paper within papertrack
        curr_inds = [jj for jj,x in enumerate(bibcodes_classes)
                        if (x == curr_bibcode)]
        if len(curr_inds) == 0:
            print("Bibcode ({0}, {1}) not in papertrack database. Continuing..."
                    .format(ii, curr_bibcode))
            bibcodes_notin_papertrack.append(curr_bibcode)
            num_notin_papertrack += 1
            continue
        #

        #Copy over data from papertrack into text database
        curr_dict["class_missions"] = {}
        for jj in range(0, len(curr_inds)):
            #Prepare inner dictionary for current mission
            inner_dict = {}
            curr_dict["class_missions"][missions_classes[
                                                    curr_inds[jj]]] = inner_dict
            #Store papertrack information into the inner dictionary
            inner_dict["bibcode"] = bibcodes_classes[curr_inds[jj]]
            inner_dict["papertype"] = papertypes_classes[curr_inds[jj]]
            tmp_falsepos = is_falsepos_classes[curr_inds[jj]]
            if tmp_falsepos in ["True", "TRUE"]:
                inner_dict["is_falsepos"] = True
            elif tmp_falsepos in ["False", "FALSE"]:
                inner_dict["is_falsepos"] = False
            else:
                raise ValueError("Whoa! Diff. false-pos flag {0} at {1}!\n{2}"
                    .format(tmp_falsepos, curr_inds[jj]))
            #
    #

    #Print some notes
    if do_verbose:
        print("Done generating dictionaries of combined papertrack+text data.")
        print("NOTE: {0} papers in text data that were not in papertrack."
                .format(num_notin_papertrack))
    #

    #Save the file, if so desired, and exit the function
    if do_save:
        #Save the combined dataset
        with open(filesave_json, 'w') as openfile:
            json.dump(storage, openfile, indent=4)
        #Also save the paper-texts not found in papertrack
        np.savetxt(filesave_notinpapertrack,
                    np.asarray(bibcodes_notin_papertrack).astype(str),
                    delimiter="\n", fmt='%s')
    #
    return storage
#

#Procedure to extract+save papers from dataset
def extract_all_paper_missions():
    ##Set global variables
    #Load the dataset of texts
    dataset_orig = load_data(filename_json=filename_json,
                                i_truncate=None)["dataset"]
    #

    ##Establish reference keys for all available missions within papers
    list_paper_missions = np.unique([item2
                        for item1 in dataset_orig
                        if ("class_missions" in item1)
                        for item2 in item1["class_missions"].keys()])
    #

    ##Return the extracted mission set
    print("Done extracting all paper missions from original dataset!")
    print("Missions: {0}".format(list_paper_missions))
    return list_paper_missions
#

#Procedure to extract+save papers from dataset
def extract_papers_from_dataset(map_classifs_orig, which_missions, which_acronym, dir_test, max_papers, do_test_only, do_falsehits):
    ##Throw an error if tests only requested and false-hits requested
    #Not permitted, because otherwise keyobj matches not in tests can be...
    #...flagged as z_NOTMATCH, even though code technically matches to them
    if (do_test_only and do_falsehits):
        raise ValueError("Whoa! Test-only and false-hits cannot both be True!")
    #


    ##Set global variables
    #Load the dataset of texts
    dataset_orig = load_data(filename_json=filename_json,
                                i_truncate=None)["dataset"]
    num_dataorig = len(dataset_orig)
    #
    #Load the allowed and converted classifs
    if (map_classifs_orig is not None):
        map_classifs = copier.deepcopy(map_classifs_orig) #Independent copy
        #Tack on falsehit classif, if requested
        if do_falsehits:
            map_classifs[""] = preset.verdict_rejection #falsehit
        else:
            map_classifs[""] = None
        #
        #Extract allowed and converted classes
        tmp_res = fetch_classifs(mapper=map_classifs)
        allowed_classifs = tmp_res["allowed"]
        converted_classifs = tmp_res["converted"]
    else:
        map_classifs = None
        allowed_classifs = None
        converted_classifs = None
    #
    #Prepare containers for papers to keep
    dataset_keep = {}
    #
    #Prepare shuffled indices for accessing papers
    tmp_inds_shuffled = list(range(0, num_dataorig))
    np.random.shuffle(tmp_inds_shuffled)
    #

    ##Perform quick checks
    #Check that target papertypes is blank if target missions is blank
    if (which_missions is None) and (map_classifs is not None):
        raise ValueError("Whoa! map_classifs should be None"
                            +" if which_missions is None!")
    #
    #Verify that the papertypes in the dataset are all recognized by mapper
    if (map_classifs is not None):
        tmp_list = np.unique(
            [dataset_orig[ind]["class_missions"][key]["papertype"]
            for ind in range(0, num_dataorig)
                if ("class_missions" in dataset_orig[ind])
            for key in dataset_orig[ind]["class_missions"]]
            ) #Unique list of classifs within original dataset
        #
        tmp_check1 = np.sort(list(map_classifs.keys()))
        tmp_check2 = np.sort(tmp_list)
        if not np.array_equal(tmp_check1, tmp_check2):
            raise ValueError("Whoa! Diff. classifs in map vs data?!\n{0} vs {1}"
                            .format(tmp_check1, tmp_check2))
    #

    ##If evaluating only on testing data, then extract indices of tests
    if do_test_only:
        list_testfiles_hi = os.listdir(dir_test)
        list_testfiles_raw = [os.listdir(os.path.join(dir_test, item))
                                for item in list_testfiles_hi
                                if os.path.isdir(os.path.join(dir_test, item))]
        #list_testfiles_raw = [item2 for item1 in list_testfiles_raw
        list_testfiles = [item2 for item1 in list_testfiles_raw
                                for item2 in item1]
        #list_testfiles = [item.replace(".txt", "")
        #                    for item in list_testfiles_raw] #Remove extensions
        #list_inds_test = [item.split("_")[1]
        #                    for item in list_testfiles] #Indices of each file
        #list_keys_lookup = [item.split("_")[3]
        #                    for item in list_testfiles] #Keyword of each file
    #

    ##Iterate through shuffled papers
    i_track = 0
    for ii in tmp_inds_shuffled:
        paper_orig = dataset_orig[ii]

        if do_verbose:
            print("-"*30)
            print("\nConsidering paper {0}:\n".format(ii))
        #


        ##Skip papers that do not fulfill given criteria
        #Skip if not data from test dir., if requested
        #if do_test_only:
            #Skip if not test data
        #    if (str(ii) not in list_inds_test):
        #        if do_verbose:
        #            print("Paper {0} is not test paper. Skipping.".format(ii))
        #        continue
            #Otherwise, store the lookup-keywords for this file
            #else:
            #    if do_verbose:
            #        print("Paper {0} is test paper. Keeping so far.".format(ii))
            #    #
            #    print(paper.keys())
            #    print(dothisbetter)
            #    paper["lookup"] =list_keys_lookup[list_inds_test.index(str(ii))]
        #

        #Skip if no text for this paper
        if "body" not in paper_orig:
            if do_verbose:
                print("Skipping paper {0} since no text available.".format(ii))
            #Skip ahead
            continue
        #

        #Skip if no missions for this paper (not PaperTrack) and targets given
        if (which_missions is not None) and ("class_missions" not in paper_orig):
            if do_verbose:
                print("No missions for paper {0}. Skipping.".format(ii))
            #Skip ahead
            continue
        #

        #Skip if no target missions for this paper
        if (which_missions is not None):
            is_mission = any([item in which_missions
                            for item in paper_orig["class_missions"]])
            if not is_mission:
                if do_verbose:
                    print("Skipping this paper since no target missions ({0})."
                            .format(paper_orig["class_missions"]))
                #Skip ahead
                continue
            #
            #Otherwise, print some notes
            else:
                if do_verbose:
                    print("Paper {0} has missions: {1}"
                            .format(ii, paper_orig["class_missions"].keys()))
        #

        #Skip if no target papertypes for this paper and targets given
        if (map_classifs is not None):
            is_papertype = any([paper_orig["class_missions"][item]["papertype"]
                                in allowed_classifs for item in which_missions
                                if item in paper_orig["class_missions"]])
            if not is_papertype:
                if do_verbose:
                    print("Skipping this paper since no target papertypes ({0})."
                        .format([paper_orig["class_missions"][item]["papertype"]
                                    for item in which_missions
                                    if item in paper_orig["class_missions"]]))
                #Skip ahead
                continue
            #
            #Otherwise, print some notes
            if do_verbose:
                print("Paper {0} has mission:papertypes:".format(ii))
                for item in paper_orig["class_missions"]:
                    if item in which_missions:
                        print("- '{0}': '{1}'".format(item,
                            paper_orig["class_missions"][item]["papertype"]))
        #

        #Skip this paper if does not contain specific acronym, if requested
        if (which_acronym is not None):
            tmp_text = paper_orig["body"].replace(".", "")
            tmp_exp = r"\b" + which_acronym.replace(".", "") + r"\b"
            if not bool(re.search(tmp_exp, tmp_text, flags=re.IGNORECASE)):
                #Print some notes
                if do_verbose:
                    print("Requested acronym {0} not in paper {1}. Skipping."
                            .format(which_acronym, ii))
                #Skip ahead
                continue
        #


        ##Make a deep copy of this paper, if passed earlier skipping tests
        paper_new = copier.deepcopy(paper_orig)
        paper_new["ignored_missions"] = {}#For ignored not-test missions, etc


        ##Fill in blank placeholder if no missions for this paper
        if ("class_missions" not in paper_new):
            paper_new["class_missions"] = {}
        #


        ##Modify any false-hits to have standard false-hit and other verdicts
        is_any_kept = (not do_test_only) #Initialize boolean
        tmp_keys = list(paper_new["class_missions"].keys())
        for curr_mission in tmp_keys:
            if do_verbose:
                print("---")
                print("Considering mission {0}.".format(curr_mission))
            #
            curr_papertype_init = paper_new["class_missions"][
                                                curr_mission]["papertype"]
            #
            #Verify that this statement is a test statement
            if do_test_only:
                tmp_name = build_filename_statement(
                                base_forsave=base_forsave, id_dataset=ii,
                                papertype=curr_papertype_init,
                                mission=curr_mission)
                if (tmp_name not in list_testfiles): #Skip if not test statement
                    if do_verbose:
                        print("Test-only, and {0} not test. Removing."
                                .format(curr_mission))
                    #
                    paper_new["ignored_missions"][curr_mission] = (
                                    paper_new["class_missions"][curr_mission])
                    del paper_new["class_missions"][curr_mission]
                    continue
                #
                else: #At least one kept, so mark to keep
                    is_any_kept = True
            #

            #Convert the papertype to final version as needed
            if (map_classifs is not None):
                #Extract and convert the papertype for this mission
                curr_papertype_conv = map_classifs[curr_papertype_init]
                #Store the original and converted papertypes for paper
                paper_new["class_missions"][curr_mission]["papertype_init"
                            ] = curr_papertype_init #Initial papertype
                paper_new["class_missions"][curr_mission]["papertype_fin"
                            ] = curr_papertype_conv #Converted papertype
                paper_new["class_missions"][curr_mission]["papertype"
                            ] = None #Erase old value to avoid confusion later
            #
            else: #Otherwise, just use original values
                paper_new["class_missions"][curr_mission]["papertype_init"
                            ] = None
                paper_new["class_missions"][curr_mission]["papertype_fin"
                            ] = curr_papertype #Original papertype
                paper_new["class_missions"][curr_mission]["papertype"
                            ] = None #Erase old value to avoid confusion later
        #


        ##Skip ahead if no missions kept for this paper
        if (not is_any_kept):
            if do_verbose:
                print("Test-only, and nothing kept for this paper. Skipping.")
            continue
        #


        ##Store the data for this paper
        dataset_keep[str(ii)] = paper_new
        i_track += 1
        #


        ##Terminate early if enough papers stored
        if (max_papers is not None) and (i_track == max_papers):
            break
    #


    ##Return the extracted database
    print("Done extracting papers from original dataset!")
    print("Original number of papers: {0}".format(num_dataorig))
    print("Kept number of papers: {0}".format(len(dataset_keep)))
    return dataset_keep
#

#Procedure to extract+save sentences with target keywords from given text
def script_process_text(filename_json, keyword_objs, map_classifs, target_missions, buffer, filename_bibnotintrack, do_save=False, dir_forsave=None, base_forsave=None, i_truncate=None, do_treeerrors=True, do_verbose=False, filename_missingkeys="missingkeys.txt", filename_bibcodes="bibcodes.txt", filename_classifs="classifs.json", filename_duplicates="duplicates.txt", filename_innererrors="innererrors.txt"):
    #Prepare overarching variables
    textforms = classifys_textforms
    #main_textform = preset.main_textform
    #
    #Truncate textforms if buffer > 0
    if (buffer is None) or (buffer != 0):
        textforms = [main_textform]

    #Load the dataset of texts
    stuff = load_data(filename_json=filename_json, i_truncate=i_truncate)
    bibcodes_notin_papertrack = load_text(filename_bibnotintrack)
    dataset = stuff["dataset"]
    num_texts = stuff["num_texts"]
    num_kobjs = len(keyword_objs)
    #

    #Prepare directory within which to save statements, if so desired
    if do_save:
        dir_texts = os.path.dirname(dir_forsave)
        filepath_bibcodes = os.path.join(dir_texts, filename_bibcodes)
        filepath_classifs = os.path.join(dir_texts, filename_classifs)
        filepath_missingkeys = os.path.join(dir_texts, filename_missingkeys)
        filepath_duplicates = os.path.join(dir_texts, filename_duplicates)
        filepath_innererrors = os.path.join(dir_texts, filename_innererrors)
        is_newdir = prep_directory(dir_name=dir_forsave, textforms=textforms,
                                    map_classifs=map_classifs,
                                    filepath_bibcodes=filepath_bibcodes,
                                    filepath_classifs=filepath_classifs,
                                    filepath_missingkeys=filepath_missingkeys,
                                    filepath_duplicates=filepath_duplicates,
                                    filepath_innererrors=filepath_innererrors)
        str_bibcodes = load_text(filepath=filepath_bibcodes)
        str_missingkeys = load_text(filepath=filepath_missingkeys)
        str_duplicates = load_text(filepath=filepath_duplicates)
        str_innererrors = load_text(filepath=filepath_innererrors)
        counter_classifs = load_data(filename_json=filepath_classifs,
                                    i_truncate=None)["dataset"]
        if str_bibcodes != "":
            startinglist_bibcodes_raw = [item.split(" : ")
                                        for item in str_bibcodes.split("\n")]
            startinglist_bibcodes = [item[1] for item in
                                        startinglist_bibcodes_raw
                                        if (item != [""])]
            startinglist_ids = [item[0] for item in
                                        startinglist_bibcodes_raw
                                        if (item != [""])]
        else:
            startinglist_bibcodes = []
            startinglist_ids = []
    #Otherwise, set empty strings
    else:
        str_bibcodes = ""
        startinglist_bibcodes = []
        startinglist_ids = []
    #

    #Print some notes
    if do_verbose:
        print("Iterating through papers next!")
    #

    #Iterate through all texts ('papers')
    count_errors = 0
    count_kept = 0
    for ii in range(0, num_texts):
        #Print some notes
        if do_verbose:
            print(("--"*30)+"\n")
            print("Starting Paper #{0} of {1}.".format(ii, num_texts))
        #

        #Extract current paper information
        paper = dataset[ii]

        #Note and skip this paper if no text given at all
        """
        if "body" not in paper:
            if do_verbose:
                print("Skipping this paper because no text found: '{0}'."
                        .format(paper["bibcode"]))
            if do_save and (curr_bibcode not in str_innererrors):
                print("Recording...")
                str_innererrors += "{0} : {1}\n".format(ii, paper["bibcode"])
                #Save keyword-empty bibcodes (SO FAR) to file
                write_text(text=str_innererrors,
                            filepath=filepath_innererrors)
            continue
        #"""

        #Extract current text information
        curr_bibcode = paper["bibcode"]
        #Skip this paper if an approved exception
        if curr_bibcode in skip_bibcodes:
            if do_verbose:
                print("Skipping this paper because skip-marked bibcode: '{0}'."
                        .format(curr_bibcode))
            continue
        #
        curr_text = paper["body"]
        #

        #Skip this paper if bibcode not within papertrack
        """
        if paper["bibcode"] in bibcodes_notin_papertrack:
            if do_verbose:
                print("Skipping this paper because not in papertrack: '{0}'."
                        .format(paper["bibcode"]))
            continue
        #"""

        #Extract more information specific to this mission
        curr_missions = [item for item in paper["class_missions"]]
        is_missions = [(item in curr_missions) for item in target_missions]

        #Skip this paper if target mission not covered in this paper
        if not any(is_missions):
            if do_verbose:
                print("Skipping because non-target missions: '{0}'."
                        .format(curr_missions))
            continue
        #

        #Extract remaining information specific to existing target missions
        curr_papertypes_raw = [None]*len(target_missions)
        curr_papertypes_all = [None]*len(target_missions)
        for jj in range(0, len(target_missions)):
            if is_missions[jj]:
                tmp_papertype = paper["class_missions"][
                                    target_missions[jj]]["papertype"]
                #Map and store raw papertype to substitute papertype as needed
                curr_papertypes_raw[jj] = tmp_papertype
                curr_papertypes_all[jj] = map_classifs[tmp_papertype]
                if do_verbose:
                    print("Initial papertype '{0}' mapped to '{1}' for {2}."
                            .format(curr_papertypes_raw[jj],
                                    curr_papertypes_all[jj],
                                    target_missions[jj]))
        #

        #Print some notes
        if do_verbose:
            print("")
            #print("--"*30)
            #print("Considering Paper #{0} of {1}:".format(ii, num_texts))
            print("Paper missions: {0}".format(curr_missions))
            print("Target missions: {0}".format([item for item in curr_missions
                                                if (item in target_missions)]))
            print("Initial paper classes: {0}".format(curr_papertypes_raw))
            print("Mapped paper classes: {0}".format(curr_papertypes_all))
        #

        #Skip this paper if type not within allowed list
        curr_skips = [(item is None) for item in curr_papertypes_all]
        curr_empties = [False for item in curr_skips]
        if all(curr_skips):
            if do_verbose:
                print("Skipping this paper because non-target classes: '{0}'."
                        .format(curr_papertypes_all))
            continue
        #

        #Extract statements from current text
        statements_perkobj = [None]*num_kobjs
        #try:
        for jj in range(0, num_kobjs):
            if (not is_missions[jj]) or curr_skips[jj]:
                continue
            statements_perkobj[jj] = parse_text(text=curr_text,
                                        keyword_objs=[keyword_objs[jj]],
                                        do_verbose=False)
        """ #BLOCKED 2023-05-30 SINCE SEEMED REDUNDANT WITH INTERNAL EXCEPTIONS
        except (KeyError, IndexError, ValueError, UnboundLocalError) as err:
            count_errors += 1
            #Throw error if surplus of errors, for now
            #if count_errors > 200:
            #    raise ValueError("Whoa! Too many errors!: {0}"
            #                    .format(count_errors))
            #
            #
            if do_treeerrors:
                raise ValueError("Whoa! Internal error for {0}, {1}."
                                .format(ii, curr_bibcode))
            else:
                print("Internal error found for {0}, {1}:\n{2}\nNoted, skipped."
                                .format(ii, curr_bibcode, err)
                        +"\n{0} errors skipped so far.".format(count_errors))
                if do_save and (curr_bibcode not in str_innererrors):
                    print("Recording...")
                    str_innererrors += "{0} : {1}\n".format(ii, curr_bibcode)
                    #Save keyword-empty bibcodes (SO FAR) to file
                    write_text(text=str_innererrors,
                                filepath=filepath_innererrors)
                continue
        #"""
        #

        ##Iterate through keyword objects
        curr_paragraph_dict_perkobj = [None]*num_kobjs
        for jj in range(0, num_kobjs):
            #Skip if no papertype for this keyword object for this paper
            if (not is_missions[jj]) or curr_skips[jj]:
                print("Skipping {0}, {1}, {2}: <not mission> or <skip>..."
                    .format(jj, target_missions[jj], curr_papertypes_raw[jj]))
                continue
            #

            #Throw an error if unequal number of statements per text form
            tmp_counts = [len(statements_perkobj[jj][textforms[kk].lower()])
                            for kk in range(0, len(textforms))]
            if (tmp_counts.count(tmp_counts[0]) != len(tmp_counts)):
                errstr = ""
                for key in statements_perkobj[jj]:
                    errstr += "{0}\n".format(len(statements_perkobj[jj][key]))
                    errstr += "{0}\n".format(statements_perkobj[jj][key])
                raise ValueError("Whoa! Unequal # statements per textform?!\n"
                                +errstr)
            num_statements = tmp_counts[0] #Number of statements per text form
            #Throw an error if no statements for this paper with target mission
            if num_statements == 0:
                #raise ValueError("Whoa! No statements for target paper?! {0}, {1}"
                #                    .format(ii, curr_bibcode))
                print("No statements for paper {0}:{1}:{2}!"
                                .format(ii, curr_bibcode, target_missions[jj])
                                +"\nPaper missions:\n{0}\n{1}\n{2}"
                                .format(curr_papertypes_raw,
                                        curr_papertypes_all, curr_missions))
                curr_empties[jj] = True
                if do_save and (curr_bibcode not in str_missingkeys):
                    print("Recording...")
                    str_missingkeys += ("{0} : {1} : {2}\n"
                                        .format(ii, curr_bibcode, jj))
                    #Save keyword-empty bibcodes (SO FAR) to file
                    write_text(text=str_missingkeys, filepath=filepath_missingkeys)
                continue
            #

            #Combine the statements into one paragraph
            curr_paragraph_dict_perkobj[jj] = {}
            if buffer is None: #If full text wanted
                curr_paragraph_dict_perkobj[jj][main_textform] = curr_text#Full
            elif isinstance(buffer, int): #If buffered-sentences wanted
                for subdir in textforms:
                    #Join statements
                    curr_paragraph_dict_perkobj[jj][subdir
                        ] = "\n\n".join(statements_perkobj[jj][subdir.lower()])
            else:
                raise ValueError("Whoa! Unrecognized buffer {0} given!"
                                    .format(buffer))
            #

            #Print some notes
            if do_verbose:
                print("Extracted statements ({0} in total):".format(num_statements))
                #for key in statements:
                for key in textforms:
                    print("--"*20)
                    print("{0}: {1}\n".format(key,
                                        statements_perkobj[jj][key.lower()]))
                    print("Fin. Paragraph:\n{0}\n"
                            .format(curr_paragraph_dict_perkobj[jj][key]))
        #

        #If this paper has been stored before, check if text is the same
        if bool(re.search(re.escape(curr_bibcode), str_bibcodes)):
            #Print some notes
            if do_verbose:
                print("Paper {0} ({1}) has already been catalogued."
                        .format(ii, curr_bibcode))
                print("Verifying that bibcode indices are identical...")

            #Throw an error if bibcode not at expected index
            #tmp_ind = startinglist_ids.index(str(ii))
            old_ind = startinglist_bibcodes.index(curr_bibcode)
            old_id = startinglist_ids[old_ind]
            #Print some notes
            if do_verbose:
                print("Old ind, id: {0}, {1}".format(old_id, old_ind))
            #
            if curr_bibcode != startinglist_bibcodes[old_ind]:
                raise ValueError("Whoa! Bibcodes out of order? {0} != {1}."
                        .format(curr_bibcode, startinglist_bibcodes[old_ind]))
            #Take note of and skip this paper if duplicate
            if old_ind != ii:
                print("Paper {0} appears to be a duplicate to {1}. Will skip."
                            .format(ii, old_ind))
                if do_save:
                    print("Recording, then skipping...")
                    str_duplicates += ("{0} : {1}\n"
                                        .format(ii, curr_bibcode))
                    #Save keyword-empty bibcodes (SO FAR) to file
                    write_text(text=str_duplicates,
                                filepath=filepath_duplicates)
                continue

            #Mark this paper as complete
            is_newpaper = False #Not a new paper, ultimately

            #Print some notes
            if do_verbose:
                print("Paragraph has been checked.")
        #
        #Otherwise, note that paper has not been stored before
        else:
            #Print some notes
            if do_verbose:
                print("Paper {0} ({1}) is new.".format(ii, curr_bibcode))
            is_newpaper = True #New paper encountered
        #

        #Do saving of statements for this text, if so desired
        if (do_save and is_newpaper):
            #Print some notes
            if do_verbose:
                print("Saving this paragraph for this paper...")

                #Save the statements for each text form under the correct subdir.
                for jj in range(0, num_kobjs):
                    #Skip if no target mission or empty here
                    if ((not is_missions[jj]) or curr_skips[jj]
                                        or curr_empties[jj]):
                        #print("Skipping {0}: <not mission> or <skip>..."
                        #        .format(jj))
                        continue
                    #
                    print("Processing for mission {0}:{1}..."
                            .format(jj, target_missions[jj]))
                    for subdir in textforms:
                        tmp_name = build_filename_statement(
                                        base_forsave=base_forsave,id_dataset=ii,
                                        papertype=curr_papertypes_raw[jj],
                                        mission=target_missions[jj])
                        tmp_path = os.path.join(dir_texts, subdir.upper(),
                                            curr_papertypes_all[jj], tmp_name)
                        write_text(curr_paragraph_dict_perkobj[jj][subdir],
                                            filepath=tmp_path)
        #

        #Otherwise, maintain same classification and replace old statements
        elif (do_save and (not is_newpaper)):
            #Print some notes
            if do_verbose:
                print("Overwriting latest statements for Paper {0}, old={1}..."
                        .format(ii, old_ind))

            #Save the statements for each text form under the correct subdir.
            #Iterate through keyword objects
            for jj in range(0, num_kobjs):
                if ((not is_missions[jj]) or curr_skips[jj]
                                        or curr_empties[jj]):
                    print("Skipping {0}: <not mission> or <skip>..."
                                .format(jj))
                    continue
                #
                #Iterate through textforms
                for subdir in textforms:
                    #Prepare filepath for current statement
                    tmp_name = build_filename_statement(
                                        base_forsave=base_forsave,id_dataset=ii,
                                        papertype=curr_papertypes_raw[jj],
                                        mission=target_missions[jj])
                    tmp_path = os.path.join(dir_texts, subdir.upper(),
                                            curr_papertypes_all[jj], tmp_name)
                    #Load the previous statement for comparison
                    pre_paragraph = load_text(tmp_path)
                    write_text(curr_paragraph_dict_perkobj[jj][subdir],
                                            filepath=tmp_path)
                    #Print some notes
                    if do_verbose:
                        print("Text form (classif.): {0} ({1})"
                                .format(subdir, curr_papertypes_all[jj]))
                        print("Previous text: {0}".format(pre_paragraph))
                        print("New text: {0}\n".format(
                                curr_paragraph_dict_perkobj[jj][subdir]))
        #

        #If this is a new paper, then record the bibcode and classif count
        if (do_save and is_newpaper):
            if do_verbose:
                print("Updating bibcodes...")
            str_bibcodes += "{0} : {1}\n".format(ii, curr_bibcode)
            #Save bibcodes (SO FAR) to file
            write_text(text=str_bibcodes, filepath=filepath_bibcodes)

            #Count up the raw and converted classifs
            for jj in range(0, len(curr_papertypes_all)):
                #Count if classif here
                if ((curr_papertypes_all[jj] is not None)
                                    and (not curr_empties[jj])):
                    counter_classifs["init_"+curr_papertypes_raw[jj]] += 1
                    counter_classifs["fin_"+curr_papertypes_all[jj]] += 1
            #
            #Update the saved counter
            with open(filepath_classifs, 'w') as openfile:
                json.dump(counter_classifs, openfile, indent=4)
        #

        #Otherwise, do nothing here
        else:
            pass
        #

        #Store the bibcode in local list
        startinglist_bibcodes.append(curr_bibcode)
        startinglist_ids.append(ii)

        #Increment count of kept papers
        count_kept += 1

        #Print some notes
        if do_verbose:
            print("Paper #{0} of {1} complete.".format(ii, num_texts))
            print("{0} papers kept so far.".format(count_kept))
            print("--"*30)
            print("\n\n")
        #
    #
    return
    #
#

#Helper function to prepare directory for saved files, if not already created
def prep_directory(dir_name, textforms, map_classifs, filepath_bibcodes, filepath_classifs, filepath_missingkeys, filepath_innererrors, filepath_duplicates, do_bibcodefile=True, do_verbose=True):
    dir_texts = os.path.dirname(dir_name)
    #Extract allowed and converted classtypes
    tmp_res = fetch_classifs(mapper=map_classifs)
    allowed_classtypes = tmp_res["allowed"]
    converted_classtypes = tmp_res["converted"]
    #Print some notes
    if do_verbose:
        print("Running the 'make_directory' function.")

    #If desired directory already exists, exit function
    if os.path.exists(dir_texts):
        if do_verbose:
            print("Given directory ({0}) already exists. Exiting function..."
                    .format(dir_texts))
        return False

    #Otherwise, create the desired directory, subdirectories, and bibcode file
    #Print some notes
    if do_verbose:
        print("Creating directory set ({0}) for text files...".format(dir_texts))
    #Create the directory
    os.makedirs(dir_texts)
    #Create the bibcode file
    if do_bibcodefile:
        write_text(text="", filepath=filepath_bibcodes)
        write_text(text="", filepath=filepath_missingkeys)
        write_text(text="", filepath=filepath_duplicates)
        write_text(text="", filepath=filepath_innererrors)
    #
    #Create the dictionary counter of initial and converted classif counts
    dict_counter = {("init_"+key):0 for key in allowed_classtypes}
    dict_counter.update({("fin_"+map_classifs[key]):0
                            for key in allowed_classtypes})
    with open(filepath_classifs, 'w') as openfile:
        json.dump(dict_counter, openfile, indent=4)
    #
    #Create the subdirectories for each textform
    for subdir in textforms:
        path1 = os.path.join(dir_texts, subdir.upper())
        os.makedirs(path1) #For subdirectory
        for subsubdir in converted_classtypes:
            path2 = os.path.join(path1, subsubdir.upper())
            os.makedirs(path2) #For subsubdirectory
    #

    #Exit the function
    #Print some notes
    if do_verbose:
        print("Directory, subdirectories, and opt. bibcode text file complete.")
    return True
#

#Helper function to extract statements from a single text block
def parse_text(text, keyword_objs, do_verbose=False):
    ##Extract and return statements from this text
    #Iterate through keyword objects
    for ii in range(0, len(keyword_objs)):
        curr_obj = keyword_objs[ii]
        grammar = bibcat.Grammar(text=text, keyword_obj=curr_obj,
                                do_verbose=do_verbose, buffer=0)
        curr_statements = grammar.get_statements()
        #Tack these statements onto the total statements
        if ii == 0:
            statements_all = curr_statements
        else:
            for key in curr_statements:
                statements_all[key] += curr_statements[key]
    #
    return statements_all
#

#Build filename for a statement (saved keyword paragraph)
def build_filename_statement(base_forsave, id_dataset, papertype, mission):
    #Build and return the appropriate file name for this statement
    return ("{0}_{1}_{2}_{3}.txt"
            .format(base_forsave, id_dataset, papertype, mission))
#

#Helper function to load data from .json file
def load_data(filename_json, i_truncate):
    with open(filename_json, 'r') as openfile:
        dataset = json.load(openfile)
    if i_truncate is not None: #Truncate dataset if so desired
        dataset = dataset[0:i_truncate]
    num_texts = len(dataset)

    #Return loaded data
    return {"dataset":dataset, "num_texts":num_texts}
#

##Helper function to write a text (or text form) to a file
def load_text(filepath):
    #Load bibcodes text file
    with open(filepath, 'r') as openfile:
        text = openfile.read()
    #Return the loaded bibcodes
    return text
#

##Helper function to write a text (or text form) to a file
def write_text(text, filepath):
    with open(filepath, 'w') as openfile:
        openfile.write(text)
    return
#

##Copy statements into dir. of training+validation+test (TVT) statements for ML
def gen_dir_TVT(dir_forbase, dir_forTVT, which_scheme, filename_bibcodes, filename_classifs, map_classifs, fraction_TVT, randseed, exp_statement, buffer, do_save=False, do_verbose=False):
    #Print some notes
    if do_verbose:
        print("Starting function for generating directory of TVT statements.")
        print("Fraction to use for TVT: {0}".format(fraction_TVT))
        print("Statements will be copied from:\n{0}".format(dir_forbase))
        print("TVT directory will take place in:\n{0}".format(dir_forTVT))
        print("Loading variables and bibcodes...")
    #

    #Load global variables
    textforms = classifys_textforms
    #main_textform = preset.main_textform
    #Extract converted classtypes
    converted_classtypes = fetch_classifs(mapper=map_classifs)["converted"]
    num_classtypes_conv = len(converted_classtypes)
    #
    #Truncate textforms if buffer > 0
    if (buffer is None) or (buffer != 0):
        raise ValueError("Whoa! Not really built to handle buffer != 0?")
        #textforms = [main_textform]
    #
    list_TVT = classifys_TVTs
    num_textforms = len(textforms)
    np.random.seed(randseed)

    #Load bibcodes and paper count
    dir_texts = os.path.dirname(dir_forbase)
    filepath_bibcodes = os.path.join(dir_texts, filename_bibcodes)
    filepath_classifs = os.path.join(dir_texts, filename_classifs)
    str_bibcodes = load_text(filepath=filepath_bibcodes)
    if str_bibcodes == "": #Raise an error if no bibcodes here
        raise ValueError("Whoa! Bibcodes file is empty!")
    str_split_bibcodes = [item.split(" : ")for item in str_bibcodes.split("\n")]
    paper_bibcodes = [item[1] for item in str_split_bibcodes if (item != [""])]
    paper_ids = [item[0] for item in str_split_bibcodes if (item != [""])]
    num_papers = len(paper_bibcodes) #Total number of papers
    #
    #Load counter of paper classifs
    dict_counter = load_data(filename_json=filepath_classifs,
                            i_truncate=None)["dataset"]
    #

    #Print some notes
    if do_verbose:
        print("Random seed set to: {0}".format(randseed))
        print("Extracting names of statement files...")
    #

    #Extract statement file names and count statements
    listfiles_perclassif = [None]*num_classtypes_conv
    num_statements_perclassif = [None]*num_classtypes_conv
    for ii in range(0, num_classtypes_conv): #Iterate through classifs
        curr_classif = converted_classtypes[ii]
        tmp_filepath = os.path.join(dir_texts, main_textform,
                                    curr_classif) #Representative path
        tmp_allfiles_raw = os.listdir(tmp_filepath)
        tmp_allfiles = [item for item in tmp_allfiles_raw
                            if (bool(re.search(exp_statement, item)))]
        listfiles_perclassif[ii] = np.asarray(tmp_allfiles) #Store stmnt files
        num_statements_perclassif[ii] = len(listfiles_perclassif[ii])
    #
    num_statements_tot = np.sum(num_statements_perclassif)
    #

    #Throw an error if counts do not match stored counter dictionary
    tmp_init_count = np.sum([dict_counter["init_"+key] for key in map_classifs
                            if (map_classifs[key] is not None)])
    tmp_fin_count = np.sum([dict_counter["fin_"+key]
                            for key in converted_classtypes])
    #
    #For individual counts
    if not all([(num_statements_perclassif[ii]
                            == dict_counter["fin_"+converted_classtypes[ii]])
                for ii in range(0, num_classtypes_conv)]):
        raise ValueError("Whoa! Mismatched statement count: {0} vs {1}\n{2}"
                            .format(num_statements_perclassif, dict_counter,
                                    converted_classtypes))
    #
    #For total counts
    if ((num_statements_tot != tmp_fin_count)
                or (num_statements_tot != tmp_init_count)):
        raise ValueError("Whoa! Incorrect statement counts: {0}, {1}, {2}, {3}"
                            .format(num_statements_perclassif, tmp_fin_count,
                                    tmp_init_count, dict_counter))
    #

    #Determine UNIFORM proportion of statements for training-validation-tests
    total_TVT_comb = np.floor(np.asarray(fraction_TVT)
                            * num_statements_tot).astype(int)#Num. stmnt per TVT
    amount_TVT_comb = (total_TVT_comb
                        // num_classtypes_conv) #Num. per cl. for Tr., V, Te.
    #

    #Determine NON-UNIFORM proportion of stmnts for training-validation-tests
    amount_TVT_perclassif = np.floor(np.array(
                [[(fraction_TVT[ind2] * num_statements_perclassif[ind1])
                    for ind2 in range(0, 3)]
                    for ind1 in range(0, num_classtypes_conv)]
                )).astype(int) #Num. per cl. for Tr., V, Te.
    #

    #Set final count of statements per classif, per TVT based on scheme
    amount_TVT_fin = [[None for jj in range(0, 3)]
                        for ii in range(0, num_classtypes_conv)]
    for ii in range(0, num_classtypes_conv):
        for jj in range(0, 3):
            if (jj == 2): #Testing set takes remainder of files
                amount_TVT_fin[ii][jj] = (num_statements_perclassif[ii]
                                - amount_TVT_fin[ii][0] - amount_TVT_fin[ii][1])
            elif (which_scheme == "individual"):
                amount_TVT_fin[ii][jj] = amount_TVT_perclassif[ii,jj]
            elif (which_scheme == "minimum"):
                amount_TVT_fin[ii][jj] = np.min(amount_TVT_perclassif[:,jj])
            else:
                raise ValueError("Whoa! Scheme {0} not recognized!"
                                .format(which_scheme))
    #
    amount_TVT_fin = np.asarray(amount_TVT_fin)

    #Print some notes
    if do_verbose:
        print("-")
        print("Total number of statements: {0}".format(num_statements_tot))
        print("Total #stmnt per classif: {0}".format(num_statements_perclassif))
        print("Tot #stmnt per TVT, combined clssf.: {0}".format(total_TVT_comb))
        print("")
        print("# per cl. for Tr,V,Te, uniform: {0}".format(amount_TVT_comb))
        print("")
        print("Non-uniform counts per classif:")
        for ii in range(0, num_classtypes_conv):
            print("# for {0} for Tr,V,Te: {1}".format(converted_classtypes[ii],
                                                amount_TVT_perclassif[ii,:]))
        print("")
        print("Final counts per classif:")
        for ii in range(0, num_classtypes_conv):
            print("# for {0} for Tr,V,Te: {1}".format(converted_classtypes[ii],
                                                        amount_TVT_fin[ii,:]))
        print("-\n\n")
        print("Shuffling indices...")
    #

    #Prepare shuffled indices for TVT extraction (same for each textform)
    inds_random_perclassif = [[None for jj in range(0, 3)]
                                for ii in range(0, num_classtypes_conv)]
    for ii in range(0, num_classtypes_conv):
        #Shuffle current indices
        curr_inds_random_raw = np.arange(0, num_statements_perclassif[ii])
        np.random.shuffle(curr_inds_random_raw)
        #Extract training indices
        starti = 0
        endi = amount_TVT_fin[ii,0]
        inds_random_perclassif[ii][0] = curr_inds_random_raw[starti:endi]
        #Extract validation files
        starti = endi
        endi += amount_TVT_fin[ii,1]
        inds_random_perclassif[ii][1] = curr_inds_random_raw[starti:endi]
        #Extract testing files
        starti = endi
        endi = len(curr_inds_random_raw) #+= amount_testing
        inds_random_perclassif[ii][2] = curr_inds_random_raw[starti:endi]
    #

    #Print some notes
    if do_verbose:
        print("Done shuffling indices.")
        for ii in range(0, num_classtypes_conv):
            for jj in range(0, 3):
                print("Number of indices for classif {0}, TVT={1}: {2}."
                        .format(converted_classtypes[ii], jj,
                                len(inds_random_perclassif[ii][jj])))
                print("First subset of indices for classif {0}, TVT={1}:\n{2}."
                        .format(converted_classtypes[ii], jj,
                                inds_random_perclassif[ii][jj][0:10]))
        print("-")
        print("Extracting files for TVT for each classif...")
    #

    #Extract files for TVT within each classif.
    movefiles_TVT = [[None for jj in range(0, 3)]
                                for ii in range(0, num_classtypes_conv)]
    for ii in range(0, num_classtypes_conv): #Iterate through classifs.
        for jj in range(0, 3): #Iterate through TVT (training,valid.,testing)
            #Concatenate in training files for this classif and TVT component
            movefiles_TVT[ii][jj] = listfiles_perclassif[ii][
                                                inds_random_perclassif[ii][jj]]
    #

    #Print some notes
    if do_verbose:
        print("Done extracting files for TVT.")
        for ii in range(0, num_classtypes_conv):
            for jj in range(0, 3):
                print("Number of files for classif {0}, TVT {1}: {2}."
                        .format(converted_classtypes[ii], jj,
                                len(movefiles_TVT[ii][jj])))
    #

    #Copy over files for each textform
    if do_save:
        #Print some notes
        print("Copying over files for TVT for each classif...")
        #Ensure that TVT directory does not already exist
        if os.path.exists(dir_forTVT):
            raise ValueError("Whoa! Given TVT directory ({0}) already exists!"
                            .format(dir_forTVT))
        #Iterate through heirarchy of files
        for aa in range(0, num_textforms): #Iterate through textforms
            curr_textform = textforms[aa].upper()
            os.makedirs(os.path.join(dir_forTVT, curr_textform))
            for jj in range(0, 3): #Iterate through TVT
                curr_TVT = list_TVT[jj]
                os.makedirs(os.path.join(dir_forTVT, curr_textform, curr_TVT))
                for ii in range(0, num_classtypes_conv): #Iterate through clfs.
                    curr_classif = converted_classtypes[ii]
                    curr_base = os.path.join(dir_forTVT, curr_textform,
                                                curr_TVT, curr_classif)
                    #
                    #Make directory for textform
                    os.makedirs(curr_base)
                    #Copy over statements
                    for zz in range(0, len(movefiles_TVT[ii][jj])): #Stmnt files
                        curr_file = movefiles_TVT[ii][jj][zz]
                        #Load the current statement text
                        old_filepath = os.path.join(dir_texts,
                                                    curr_textform, curr_classif,
                                                    curr_file)
                        curr_text = load_text(filepath=old_filepath)
                        #Write the current statement text to new location
                        new_filepath = os.path.join(curr_base, curr_file)
                        write_text(curr_text, filepath=new_filepath)
        #
        #Print some notes
        if do_verbose:
            print("Done copying over files.")
    #

    #Verify for each textform that count of files is as expected
    if do_save:
        #Print some notes
        print("Verify correct number of files for each classif...")
        #Iterate through heirarchy of files
        for aa in range(0, num_textforms): #Iterate through textforms
            curr_textform = textforms[aa].upper()
            for jj in range(0, 3): #Iterate through TVT
                curr_TVT = list_TVT[jj]
                for ii in range(0, num_classtypes_conv): #Iterate through clfs.
                    curr_classif = converted_classtypes[ii]
                    curr_base = os.path.join(dir_forTVT, curr_textform,
                                                curr_TVT, curr_classif)
                    #
                    #Verify the count of statements
                    curr_len = len([item for item in os.listdir(curr_base)
                                    if (bool(re.search(exp_statement, item)))])
                    if (curr_len != amount_TVT_fin[ii][jj]):
                        raise ValueError("Whoa! Wrong number of TVT files!"
                                    +"\nCount: {0}; Predicted: {1}"
                                    .format(curr_len, amount_TVT_fin[ii][jj])
                                    +"\nClasstype={0}, TVT={1}."
                                    .format(ii, jj))
            #
        #
        #Print some notes
        if do_verbose:
            print("Done copying over files.")
    #

    #Print some notes
    if do_verbose:
        print("Done copying over files (if do_save=True).")
        print("Note: do_save={0}.".format(do_save))
        print("Function run complete!")
    #

    #Exit the function
    return
#
###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###FUNCTIONS: MODELS AND HELPERS
#Function to fetch allowed and converted classifs based on specified mapper
def fetch_classifs(mapper):
    #Extract the allowed and converted classifs
    allowed = [key for key in mapper if (mapper[key] is not None)]
    converted = np.unique([mapper[key] for key in allowed]).tolist()
    #Return the classifs
    return {"allowed":allowed, "converted":converted}
#

#Function to fetch output of real and surface actual and measured classifs based on specified mappers
def fetch_outputs(actual_mapper, surface_mapper):
    tmp_res = fetch_classifs(mapper=actual_mapper)
    real_act_classifs = np.sort(tmp_res["allowed"] + additional_verdicts_act
                                ).tolist()
    real_meas_classifs = np.sort(tmp_res["converted"] + additional_verdicts_meas
                                ).tolist()
    tmp_res = None
    #
    #Map to surface-level classifs, if requested
    if (surface_mapper is not None):
        #surf_act_classifs =[surface_mapper[item] for item in real_act_classifs]
        #surf_meas_classifs = [surface_mapper[item]
        #                        for item in real_meas_classifs]
        surf_act_classifs = np.unique([surface_mapper[item]
                                    for item in real_act_classifs]).tolist()
        surf_meas_classifs = np.unique([surface_mapper[item]
                                    for item in real_meas_classifs]).tolist()
    #Otherwise, maintain underlying classifs
    else:
        surf_act_classifs = None
        surf_meas_classifs = None
    #
    #Return the outputs
    return {"real_act":real_act_classifs, "real_meas":real_meas_classifs,
            "surf_act":surf_act_classifs, "surf_meas":surf_meas_classifs}
#

#Function to fetch allowed and converted classifs based on specified mapper
def fetch_name_MLmodel(name_map, name_set, name_TVT, seed, name_textform):
    #Assemble name for machine learning model based on current global parameters
    namestr = ("dict_model_ML_{0}_{1}_{2}_{3}_TVTseed{4}"
                    .format(name_map, name_set, name_textform, name_TVT, seed))
    #Return assembled name
    return namestr
#

#Function to train an ML model on given TVT data
def train_model_ML(do_save, class_names, dir_TVT, filename_model, randseed):
    #Set global variables
    list_TVT = classifys_TVTs
    dir_train = os.path.join(dir_TVT, list_TVT[0])
    dir_validation = os.path.join(dir_TVT, list_TVT[1])
    dir_test = os.path.join(dir_TVT, list_TVT[2])
    #
    filesave_model = os.path.join(filepath_tests_global, filename_model)
    #

    #Initialize new classifier instance
    classifier = bibcat.Classifier_ML(filepath_model=None,
                                        class_names=class_names)
    #Create ML model
    dict_model_ML = classifier._train_BERT(dir_train=dir_train,
                    dir_validation=dir_validation, dir_test=dir_test,
                    do_save=do_save, filesave_model=filesave_model,
                    seed=randseed)

    #Exit the function
    return
#
###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###FUNCTIONS: PERFORMANCE EVALUATION
#Function to calculate performance from given classifier results
def _calculate_performance(dict_results, actual_mapper, surface_mapper, do_savemisclass, do_verbose=False, fileroot_savemisclass=None):
    ##Prepare global variables
    num_results = len(dict_results)
    tmp_res = fetch_outputs(actual_mapper=actual_mapper,
                            surface_mapper=surface_mapper)
    real_act_classifs = tmp_res["real_act"]
    real_meas_classifs = tmp_res["real_meas"]
    surf_act_classifs = tmp_res["surf_act"]
    surf_meas_classifs = tmp_res["surf_meas"]
    #

    ##Prepare string to hold misclassified cases, if so requested
    if do_savemisclass:
        str_misclass = "{0}\n---\n\n".format(fileroot_savemisclass)
        dict_misclass = {}
    else:
        dict_misclass = None
    #


    ##Compare statement verdicts and overall paper verdicts to actual classif.
    #Print some notes
    if do_verbose:
        print("Preparing containers to hold actual vs. measured verdicts...")
    #Prepare verdict counters
    dict_counters_real = {key1:{key2:0 for key2 in real_meas_classifs}
                            for key1 in real_act_classifs}
    if (surface_mapper is not None):
        dict_counters_surf = {key1:{key2:0 for key2 in surf_meas_classifs}
                            for key1 in surf_act_classifs}
    else:
        dict_counters_surf = None
    #

    #Fill in verdict counters
    list_incorrect_istr = []
    #Print some notes
    if do_verbose:
        print("Recording actual vs. measured verdict for each paper...")
    #Iterate through papers
    for curr_istr in np.sort(list(dict_results.keys())):
        curr_results = dict_results[curr_istr]
        #

        #Extract actual and measured classifs
        curr_act_real = curr_results["act_classif_real"].upper()
        curr_meas_real = curr_results["meas_classif_real"].upper()
        if (surface_mapper is not None):
            curr_act_surf = curr_results["act_classif_surf"].upper()
            curr_meas_surf = curr_results["meas_classif_surf"].upper()
        else:
            curr_act_surf = None
            curr_meas_surf = None
        #

        #Print some notes
        if do_verbose:
            print("---\nProcessed Verdict {0}:".format(curr_istr))
            print("Keyword paragraph:\n{0}.".format(curr_results["modif"]))
            print("Mission: {0}".format(curr_results["mission"]))
            print("Actual verdict: {0} ({1})."
                    .format(curr_act_surf, curr_act_real))
            print("Measured verdict: {0} ({1})."
                    .format(curr_meas_surf, curr_meas_real))
        #

        #Increment count of actual and measured surface verdicts
        dict_counters_real[curr_act_real][curr_meas_real] += 1
        if (surface_mapper is not None):
            dict_counters_surf[curr_act_surf][curr_meas_surf] += 1
        #

        #Record this paragraph id if correctly classified
        is_correct = True
        if (surface_mapper is not None):
            if (curr_act_surf != curr_meas_surf):
                list_incorrect_istr.append(curr_istr)
                is_correct = False
        elif (curr_act_real in additional_verdicts_act):
            if (curr_act_real != curr_meas_real):
                list_incorrect_istr.append(curr_istr)
                is_correct = False
        else:
            if (actual_mapper[curr_act_real] != curr_meas_real):
                list_incorrect_istr.append(curr_istr)
                is_correct = False
        #

        #Save this result to .txt file of misclassifications, if so requested
        if do_savemisclass and (not is_correct):
            str_misclass += ( #For rule-based classification
                ("Paper Internal IDs: istr={0}, ID={1}\n"
                    +"Bibcode: {2}\nMission: {3}\n"
                    +"Actual Classification: {4} ({5})\n"
                    +"Measured Classification: {6} ({7})\nText:\n'''\n{8}\n'''"
                    )
                    .format(curr_istr, curr_results["id_paper"],
                            curr_results["bibcode"], curr_results["mission"],
                            curr_act_surf, curr_act_real,
                            curr_meas_surf, curr_meas_real,
                            curr_results["modif"])
            )
            str_misclass += ("\n" + ("-"*20) + "\n")
            """
            str_misclass += (
                "Mission,istr={0}:{5}\nActual: {1} ({2})\nMeasured: {3} ({4})\n"
                    .format(curr_results["mission"], curr_act_surf,
                            curr_act_real, curr_meas_surf,
                            curr_meas_real, curr_istr)
                + "Text (ID={1}, Bibcode={2}): {0}\n"
                    .format(curr_results["modif"], curr_results["id_paper"],
                            curr_results["bibcode"]))
            str_misclass += (("-"*20) + "\n")
            """
            #Store this misclass for later use
            dict_misclass[curr_istr] = {"act_surf":curr_act_surf,
                        "meas_surf":curr_meas_surf, "meas_real":curr_meas_real,
                        "act_real":curr_act_real, "i_str":curr_istr,
                        "id_paper":curr_results["id_paper"],
                        "modif":curr_results["modif"],
                        "bibcode":curr_results["bibcode"],
                        "mission":curr_results["mission"]}
        #
    #

    #Compute totals: case of real output classifs
    for key_1 in dict_counters_real:
        dict_counters_real[key_1]["total"] = sum([
                                        dict_counters_real[key_1][key2]
                                        for key2 in dict_counters_real[key_1]])
    #
    dict_counters_real["total"] = sum([dict_counters_real[key1]["total"]
                                        for key1 in dict_counters_real])
    #

    #Compute totals: case of surface output classifs
    if (surface_mapper is not None):
        for key_1 in dict_counters_surf:
            dict_counters_surf[key_1]["total"] = sum([
                                        dict_counters_surf[key_1][key2]
                                        for key2 in dict_counters_surf[key_1]])
    #
        dict_counters_surf["total"] = sum([dict_counters_surf[key1]["total"]
                                        for key1 in dict_counters_surf])
    #

    ##Save the misclassification .txt file, if necessary
    if do_savemisclass:
        write_text(text=str_misclass, filepath=(fileroot_savemisclass+".txt"))
    #

    ##Return counters
    return {"_dict_results":dict_results, "counters_real":dict_counters_real,
            "counters_surf":dict_counters_surf, "dict_misclass":dict_misclass,
            "_incorrect_istr":list_incorrect_istr}
#

#Function to evaluate performance of full pipeline (text to rejection/verdict)
def evaluate_pipeline(dict_papers, all_keyobjs, surface_mapper, actual_mapper, which_textform, threshold_ML, threshold_rules, do_allkeyobjs, do_verify_truematch, do_raise_innererror, fileloc_ML=None, filepath_model=None, do_verbose=False, do_verbose_deep=False, do_rules=True, do_ML=True, do_savemisclass=False, fileroot_savemisclass=None):
    ##Prepare global variables
    all_paper_missions = extract_all_paper_missions()#All missions within papers
    tmp_res = fetch_classifs(mapper=actual_mapper)
    allowed_classifs = tmp_res["allowed"]
    converted_classifs = tmp_res["converted"]
    tmp_res = None
    #
    paper_ids = list(dict_papers.keys())
    num_papers = len(paper_ids)
    num_keyobjs = len(all_keyobjs)
    #

    ##Initialize full pipeline for each approach
    #For rule-based approach
    if do_rules:
        classifier_rules = bibcat.Classifier_Rules()
        tabby_rules = bibcat.Operator(classifier=classifier_rules,
                            #mode=preset.full_textform.lower(),
                            mode=full_textform.lower(),
                            keyword_objs=all_keyobjs, do_verbose=do_verbose_deep, do_verbose_deep=(spec_id is not None)) #True)
    #For ML-based approach
    if do_ML:
        classifier_ML = bibcat.Classifier_ML(filepath_model=filepath_model,
                            fileloc_ML=fileloc_ML,
                            class_names=converted_classifs)
        tabby_ML = bibcat.Operator(classifier=classifier_ML,
                            mode=which_textform.lower(),
                            keyword_objs=all_keyobjs, do_verbose=do_verbose_deep)
    #

    ##Classify each paper using each approach
    dict_results_ML = {}
    dict_results_rules = {}
    i_track = 0 #Keep track of keyword paragraphs to use as ids
    #Iterate through papers
    for ii in range(0, num_papers):
        #Print some notes
        if do_verbose:
            print("---"*20)
            print("Starting classification of paper {0}.".format(ii))
        #

        #Extract the current text
        curr_paperid = paper_ids[ii]
        curr_text = dict_papers[curr_paperid]["body"]
        curr_bibcode = dict_papers[curr_paperid]["bibcode"]
        if "class_missions" in dict_papers[curr_paperid]:
            curr_missiondict = dict_papers[curr_paperid]["class_missions"]
            curr_ignored = dict_papers[curr_paperid]["ignored_missions"]
        else:
            curr_missiondict = {}
            curr_ignored = {}
        #

        #Iterate through target keyword objects
        for jj in range(0, num_keyobjs):
            curr_istr = str(i_track)
            #Match this keyword to one of missions in papers
            tmp_kw = all_keyobjs[jj]._get_info("keywords")
            tmp_acr = all_keyobjs[jj]._get_info("acronyms")
            curr_lookup = [item for item in (tmp_acr+tmp_kw)
                            if (item in all_paper_missions)]
            if (len(curr_lookup) == 1): #Should just be 1 match
                curr_lookup = curr_lookup[0]
            else:
                raise ValueError("Whoa! Not just 1 lookup for {0}?\n{1}\n{2}"
                    .format(curr_lookup, all_paper_missions, (tmp_kw+tmp_acr)))
            #

            #Throw serious error if id already in dictionaries
            if (curr_istr in dict_results_rules):
                raise ValueError("Whoa! Duplicate id for rules?!\n{0}: {1}\n{2}"
                            .format(ii, curr_paperid,
                                    dict_results_rules[curr_istr]))
            if (curr_istr in dict_results_ML):
                raise ValueError("Whoa! Duplicate id for ML?!\n{0}: {1}\n{2}"
                            .format(ii, curr_paperid,
                                    dict_results_ML[curr_istr]))
            #

            #Print some notes
            if do_verbose:
                print("\nCurrent paper id: {0}".format(curr_paperid))
                print("Current keyobj lookup: {0}".format(curr_lookup))
                print("Current mission dict: {0}".format(curr_missiondict))
            #

            #If not considering all keyobjs (e.g., not do_falsehits), check
            if (not do_allkeyobjs) and (curr_lookup not in curr_missiondict):
                if do_verbose:
                    print("Skipping {0} since not in missions and not allkobj."
                            .format(curr_lookup))
                continue
            #

            #Extract actual results for keyobj for current text
            curr_res_act = {"mission":curr_lookup, "id_paper":curr_paperid}
            if ((curr_lookup in curr_missiondict)
                                and (curr_lookup not in curr_ignored)):
                curr_res_act["act_classif_real"] = curr_missiondict[curr_lookup
                                                            ]["papertype_init"]
            #Otherwise, store rejection verdict
            elif ((curr_lookup not in curr_missiondict)
                                and (curr_lookup not in curr_ignored)):
                curr_res_act["act_classif_real"] = preset.verdict_rejection
            #Otherwise, store that this was ignored (e.g., test-only)
            elif ((curr_lookup not in curr_missiondict)
                                and (curr_lookup in curr_ignored)):
                curr_res_act["act_classif_real"] = preset.verdict_ignored
            else:
                raise ValueError("Whoa! Unrecognized scenario for {0}!?"
                                .format(curr_missiondict))
            #

            #If classif not allowed, skip
            if (curr_res_act["act_classif_real"] in actual_mapper):
                if (actual_mapper[curr_res_act["act_classif_real"]] is None):
                    if do_verbose:
                        print("Classif in {0} not allowed:\n{1}\nSkipping."
                                .format(curr_res_act, actual_mapper.keys()))
                    #
                    continue
                #
            #

            #Set surface transformation of actual verdict, if requested
            if (surface_mapper is not None):
                curr_res_act["act_classif_surf"] = surface_mapper[
                                    curr_res_act["act_classif_real"].upper()]
            #Otherwise, just set actual value
            else:
                curr_res_act["act_classif_surf"] = None
            #

            #Run each pipeline on the current text and keyobj and store results
            if do_rules:
                #Extract results vs actual for current text+keyobj
                curr_res_rules = tabby_rules.classify(text=curr_text,
                                    lookup=curr_lookup, buffer=buffer,
                                    threshold=threshold_rules,
                                    do_raise_innererror=do_raise_innererror,
                                    do_check_truematch=do_verify_truematch).copy()
                curr_res_rules["meas_classif_real"
                                            ] = curr_res_rules["verdict"]
                #Store surface transformation, if requested
                if (surface_mapper is not None):
                    curr_res_rules["meas_classif_surf"
                        ] = surface_mapper[
                                curr_res_rules["meas_classif_real"].upper()]
                else:
                    curr_res_rules["meas_classif_surf"] = None
                #
                #Fold in actual classif and bibcode info
                curr_res_rules.update(curr_res_act)
                curr_res_rules["bibcode"] = curr_bibcode
                #Store the full dictionary of actual and measured results
                dict_results_rules[curr_istr] = curr_res_rules
                curr_res_rules = None
            #
            if do_ML:
                curr_res_ML = tabby_ML.classify(text=curr_text,
                                    lookup=curr_lookup, buffer=buffer,
                                    do_check_truematch=do_verify_truematch,
                                    do_raise_innererror=do_raise_innererror,
                                    threshold=threshold_ML).copy()
                curr_res_ML["meas_classif_real"
                                            ] = curr_res_ML["verdict"]
                #Store surface transformation, if requested
                if (surface_mapper is not None):
                    curr_res_ML["meas_classif_surf"
                        ] = surface_mapper[
                                curr_res_ML["meas_classif_real"].upper()]
                else:
                    curr_res_ML["meas_classif_surf"] = None
                #
                #Fold in actual classif and bibcode info
                curr_res_ML.update(curr_res_act)
                curr_res_ML["bibcode"] = curr_bibcode
                #Store the full dictionary of actual and measured results
                dict_results_ML[curr_istr] = curr_res_ML
                curr_res_ML = None
            #

            #Print some notes
            if do_verbose:
                print("Done with classification of paper {0} ({1}: {2})."
                        .format(ii, curr_istr, curr_bibcode))
                #
                if do_rules:
                    #if ("modif" in curr_res_rules):
                    print("Current keyword paragraph for rule approach:\n{0}."
                    .format(dict_results_rules[curr_istr]["modif"]))
                    print("Actual verdict (real): {0}."
                    .format(dict_results_rules[curr_istr]["act_classif_real"]))
                    print("Measured verdict (real): {0}."
                    .format(dict_results_rules[curr_istr]["meas_classif_real"]))
                    print("Actual verdict (surf.): {0}."
                    .format(dict_results_rules[curr_istr]["act_classif_surf"]))
                    print("Measured verdict (surf.): {0}."
                    .format(dict_results_rules[curr_istr]["meas_classif_surf"]))
                #
                if do_ML:
                    print("\nCurrent keyword paragraph for ML approach:\n{0}."
                    .format(dict_results_ML[curr_istr]["modif"]))
                    print("Actual verdict (real): {0}."
                    .format(dict_results_ML[curr_istr]["act_classif_real"]))
                    print("Measured verdict (real): {0}."
                    .format(dict_results_ML[curr_istr]["meas_classif_real"]))
                    print("Actual verdict (surf.): {0}."
                    .format(dict_results_ML[curr_istr]["act_classif_surf"]))
                    print("Measured verdict (surf.): {0}."
                    .format(dict_results_ML[curr_istr]["meas_classif_surf"]))
                #
                print("---"*20)
            #

            #For error checks and whatnot
            if (spec_id is not None) and (curr_paperid == spec_id):
                print(woo)
            #

            #Increment tracker of keyword paragraphs
            i_track += 1
    #

    ##Calculate performance for each classifier
    dict_perf = {}
    if do_rules:
        dict_perf["rule"] = _calculate_performance(actual_mapper=actual_mapper,
            dict_results=dict_results_rules, surface_mapper=surface_mapper,
            do_verbose=do_verbose, do_savemisclass=do_savemisclass,
            fileroot_savemisclass=fileroot_savemisclass+"_rule")
        dict_perf["rule"]["threshold"] = threshold_rules #Store prob. thres
    else:
        dict_perf["rule"] = None
    #
    if do_ML:
        dict_perf["ML"] = _calculate_performance(actual_mapper=actual_mapper,
            dict_results=dict_results_ML, surface_mapper=surface_mapper,
            do_verbose=do_verbose, do_savemisclass=do_savemisclass,
            fileroot_savemisclass=fileroot_savemisclass+"_ML")
        dict_perf["ML"]["threshold"] = threshold_ML #Store prob. thres
    else:
        dict_perf["ML"] = None
    #

    ##Record mutual error, if requested
    if do_savemisclass and (do_rules and do_ML):
        #Extract misclassification info
        dict_misclass_rule = dict_perf["rule"]["dict_misclass"]
        dict_misclass_ML = dict_perf["ML"]["dict_misclass"]

        #Extract common ids
        common_misclass_istrs = list(set(list(dict_misclass_rule.keys())
                                ).intersection(list(dict_misclass_ML.keys())))
        #

        #Save info for all common misclassifs
        str_misclass = "{0}\n---\n\n".format(fileroot_savemisclass)
        for ii in range(0, len(common_misclass_istrs)):
            #Extract current information
            curr_rules = dict_misclass_rule[common_misclass_istrs[ii]]
            curr_ML = dict_misclass_ML[common_misclass_istrs[ii]]

            #Record information including None/error output
            str_misclass += ("\n" + ("-"*20) + "\n")
            str_misclass += ( #For rule-based classification
                ("RULES:\nPaper Internal IDs: istr={0}, ID={1}\n"
                    +"Bibcode: {2}\nMission: {3}\n"
                    +"Actual Classification: {4} ({5})\n"
                    +"Measured Classification: {6} ({7})\nText:\n'''\n{8}\n'''"
                    )
                    .format(curr_rules["i_str"], curr_rules["id_paper"],
                            curr_rules["bibcode"], curr_rules["mission"],
                            curr_rules["act_surf"], curr_rules["act_real"],
                            curr_rules["meas_surf"], curr_rules["meas_real"],
                            curr_rules["modif"])
            )
            str_misclass += "\n\n"
            str_misclass += ( #For ML classification
                ("ML:\nPaper Internal IDs: istr={0}, ID={1}\n"
                    +"Bibcode: {2}\nMission: {3}\n"
                    +"Actual Classification: {4} ({5})\n"
                    +"Measured Classification: {6} ({7})\nText:\n'''\n{8}\n'''"
                    )
                    .format(curr_ML["i_str"], curr_ML["id_paper"],
                            curr_ML["bibcode"], curr_ML["mission"],
                            curr_ML["act_surf"], curr_ML["act_real"],
                            curr_ML["meas_surf"], curr_ML["meas_real"],
                            curr_ML["modif"])
            )
            str_misclass += ("\n-----\n")
            #
            """
            str_misclass += ("\n-----\n")
            str_misclass += ("RULES:\n")
            str_misclass += (
                "Mission,istr={0}:{5}\nActual: {1} ({2})\nMeasured: {3} ({4})\n"
                    .format(curr_rules["mission"], curr_rules["act_surf"],
                            curr_rules["act_real"], curr_rules["meas_surf"],
                            curr_rules["meas_real"], curr_rules["i_str"])
                + "Text (ID={1}): {0}\n"
                    .format(curr_rules["modif"], curr_rules["id_paper"]))
            str_misclass += ("\n\n")
            str_misclass += ("ML:\n")
            str_misclass += (
                "Mission,istr={0}:{5}\nActual: {1} ({2})\nMeasured: {3} ({4})\n"
                    .format(curr_ML["mission"], curr_ML["act_surf"],
                            curr_ML["act_real"], curr_ML["meas_surf"],
                            curr_ML["meas_real"], curr_ML["i_str"])
                + "Text (ID={1}): {0}\n"
                    .format(curr_ML["modif"], curr_ML["id_paper"]))
            str_misclass += (("-"*20) + "\n")
            #"""
        #

        ##Save the misclassification .txt file
        write_text(text=str_misclass,
                    filepath=(fileroot_savemisclass+"_both.txt"))
        #
    #

    ##Return the calculated performance
    return dict_perf
#

#Function to plot rectangular confusion matrix for given data and labels
def ax_confusion_matrix(matr, ax, x_labels, y_labels, x_title, y_title, cbar_title, ax_title, is_norm, cmap=plt.cm.BuPu, fontsize=16, ticksize=16, valsize=14, y_rotation=30, x_rotation=30):
    #Set global variables
    if is_norm:
        vmin = 0
        vmax = 1
    else:
        vmin = 0 #None
        #vmax = matr.sum() #None
        #Ignore nonmatch verdict to avoid spikes in color scaling, if present
        tmpmatr = matr.copy()
        if (preset.verdict_rejection.upper() in x_labels):
            tmpmatr[:,x_labels.index(preset.verdict_rejection.upper())] = -1
        if (preset.verdict_rejection.upper() in y_labels):
            tmpmatr[y_labels.index(preset.verdict_rejection.upper()),:] = -1
        vmax = tmpmatr.max() #None
    #

    #Plot the confusion matrix and colorbar
    image = ax.imshow(matr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    #Fetch the matrix dimensions
    ydim = matr.shape[0]
    xdim = matr.shape[1]

    #Write in the values
    for yy in range(0, ydim):
        for xx in range(0, xdim):
            #Set current text color based on background grayscale value
            if is_norm:
                curr_gray = np.mean(cmap(matr[yy,xx])[0:3])
            else:
                curr_gray = np.mean(cmap(matr[yy,xx] / vmax)[0:3])
            #
            if curr_gray <= 0.6:
                curr_color = "white"
            else:
                curr_color = "black"
            #

            #Write current text
            if is_norm:
                plt.text(xx, yy, "{0:.3f}".format(matr[yy,xx]), color=curr_color, horizontalalignment="center", verticalalignment="center", fontsize=valsize)
            else:
                plt.text(xx, yy, "{0:.0f}".format(matr[yy,xx]), color=curr_color, horizontalalignment="center", verticalalignment="center", fontsize=valsize)
        #
    #

    #Generate the colorbar
    cbarax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(image, cax=cbarax, extend="max")
    #cbar = plt.colorbar(image, label=cbar_title, cax=cbarax)
    cbar.ax.tick_params(labelsize=valsize)

    #Set the tick and axis labels
    ax.tick_params(axis="both", which="both", labelsize=ticksize)
    ax.set_xticks(np.arange(0, xdim, 1))
    ax.set_xticklabels([item.title() for item in x_labels], rotation=x_rotation)
    ax.set_yticks(np.arange(0, ydim, 1))
    ax.set_yticklabels([item.title() for item in y_labels], rotation=y_rotation)
    ax.set_xlabel(x_title, fontsize=fontsize)
    ax.set_ylabel(y_title, fontsize=fontsize)

    #Set the subplot title
    ax.set_title("{0}\n{1}".format(ax_title, cbar_title), fontsize=fontsize)

    #Exit the function
    return
#

#Function to plot bar plot of performance of counters extracted from some approach
def plot_performance_bars(evaluation, actual_mapper, surface_mapper, figpath, figname, figsize=(10, 6), barwidth=1, alpha=0.75, colors=["cyan", "dodgerblue", "blue", "black", "gray", "purple", "orange", "red", "goldenrod", "green", "pink"], do_norm=False, title=""):
    ##Prepare global variables
    tmp_res = fetch_outputs(actual_mapper=actual_mapper,
                            surface_mapper=surface_mapper)
    real_act_classifs = tmp_res["real_act"]
    real_meas_classifs = tmp_res["real_meas"]
    surf_act_classifs = tmp_res["surf_act"]
    surf_meas_classifs = tmp_res["surf_meas"]
    #
    num_real_act = len(real_act_classifs)
    num_real_meas = len(real_meas_classifs)
    evaluation_real = evaluation["counters_real"]
    groupwidth_real = (num_real_meas * barwidth)
    #
    if (surface_mapper is not None):
        num_surf_act = len(surf_act_classifs)
        num_surf_meas = len(surf_meas_classifs)
        evaluation_surf = evaluation["counters_surf"]
        groupwidth_surf = (num_surf_meas * barwidth)
    #

    #For performance bar plot: real classifs case
    fig = plt.figure(figsize=figsize)
    #Iterate through actual classifs for papers
    for ii in range(0, num_real_act):
        #Iterate through measured classifs for papers
        for jj in range(0, num_real_meas):
            #Prepare legend label, just for first group of bars
            if (ii == (num_real_act-1)):
                label = real_meas_classifs[jj]
                marker = ""
            else:
                label = None
            #
            if (real_act_classifs[ii] == real_meas_classifs[jj]):
                marker = "o"
            else:
                marker = ""
            #

            #Graph current bar
            xval = (ii*groupwidth_real) + (jj*barwidth)
            yval =evaluation_real[real_act_classifs[ii]][real_meas_classifs[jj]]
            if do_norm: #Normalization, if so desired
                if yval != 0:
                    yval /= evaluation_real[real_act_classifs[ii]]["total"]
            plt.bar(xval, yval, width=barwidth, color=colors[jj], alpha=alpha,
                        label=label, hatch=marker)
            #
            #Graph separating line
            if jj == 0:
                plt.axvline((xval-(barwidth/2)),
                            linestyle="--", alpha=0.5, color="gray")
            elif jj == (num_real_meas-1):
                plt.axvline((xval+(barwidth/2)),
                            linestyle="--", alpha=0.5, color="gray")
        #
    #
    #Set axes and legend
    plt.xticks((np.arange(0, (num_real_act*groupwidth_real),
                groupwidth_real)+(groupwidth_real/2)-(barwidth/2)),
                real_act_classifs)
    plt.xlabel("Actual Paragraph Class")
    if not do_norm:
        plt.ylabel("Count of Paragraphs")
    else:
        plt.ylabel("Fraction of Paragraphs")
        plt.ylim(ymax=1.05)
    #
    plt.legend(loc="best", title="Measured Class")
    plt.title("Classification of Keyword Paragraphs: {0}"
                .format(evaluation_real["total"])
                +"\nBars with dots are correct.  Empty bars are incorrect.")
    plt.suptitle(title)
    #
    #Save and close the figure
    plt.tight_layout()
    if do_norm:
        plt.savefig(figpath+"/"+figname+"_bars_real_norm.png")
    else:
        plt.yscale("log")
        plt.savefig(figpath+"/"+figname+"_bars_real_abs.png")
    plt.close()
    #

    #For performance bar plot: surface classifs case
    if (surface_mapper is not None):
        fig = plt.figure(figsize=figsize)
        #Iterate through actual classifs for papers
        for ii in range(0, num_surf_act):
            #Iterate through measured classifs for papers
            for jj in range(0, num_surf_meas):
                #Prepare legend label, just for first group of bars
                if (ii == (num_surf_act-1)):
                    label = surf_act_classifs[jj]
                    marker = ""
                else:
                    label = None
                #
                if (surf_act_classifs[ii] == surf_meas_classifs[jj]):
                    marker = "o"
                else:
                    marker = ""
                #

                #Graph current bar
                xval = (ii*groupwidth_surf) + (jj*barwidth)
                yval =evaluation_surf[surf_act_classifs[ii]
                                                    ][surf_meas_classifs[jj]]
                if do_norm: #Normalization, if so desired
                    if yval != 0:
                        yval /= evaluation_surf[surf_act_classifs[ii]]["total"]
                plt.bar(xval, yval, width=barwidth, color=colors[jj],
                        alpha=alpha, label=label, hatch=marker)
                #
                #Graph separating line
                if jj == 0:
                    plt.axvline((xval-(barwidth/2)),
                                linestyle="--", alpha=0.5, color="gray")
                elif jj == (num_surf_meas-1):
                    plt.axvline((xval+(barwidth/2)),
                                linestyle="--", alpha=0.5, color="gray")
            #
        #
        #Set axes and legend
        plt.xticks((np.arange(0, (num_surf_act*groupwidth_surf),
                    groupwidth_surf)+(groupwidth_surf/2)-(barwidth/2)),
                    surf_act_classifs)
        plt.xlabel("Actual Paragraph Class")
        if not do_norm:
            plt.ylabel("Count of Paragraphs")
        else:
            plt.ylabel("Fraction of Paragraphs")
            plt.ylim(ymax=1.05)
        #
        plt.legend(loc="best", title="Measured Class")
        plt.title("Classification of Keyword Paragraphs: {0}"
                    .format(evaluation_surf["total"])
                    +"\nBars with dots are correct.  Empty bars are incorrect.")
        plt.suptitle(title)
        #
        #Save and close the figure
        plt.tight_layout()
        if do_norm:
            plt.savefig(figpath+"/"+figname+"_bars_surf_norm.png")
        else:
            plt.savefig(figpath+"/"+figname+"_bars_surf_abs.png")
        plt.close()
    #
#

#Function to plot confusion matrix of performance of counters extracted from some approach
def plot_performance_confusion_matrix(evaluation_rules, evaluation_ML, actual_mapper, surface_mapper, figpath, figname, threshold_rules, threshold_ML, figsize=(20, 6), fontsize=16, hspace=15, cmap_abs=plt.cm.BuPu, cmap_norm=plt.cm.PuRd):
    ##Prepare global variables
    tmp_res = fetch_outputs(actual_mapper=actual_mapper,
                            surface_mapper=surface_mapper)
    real_act_classifs = tmp_res["real_act"]
    real_meas_classifs = tmp_res["real_meas"]
    num_real_act = len(real_act_classifs)
    num_real_meas = len(real_meas_classifs)
    #
    if (surface_mapper is not None):
        surf_act_classifs = tmp_res["surf_act"]
        surf_meas_classifs = tmp_res["surf_meas"]
        num_surf_act = len(surf_act_classifs)
        num_surf_meas = len(surf_meas_classifs)
    #
    evaluation_rules_real = evaluation_rules["counters_real"]
    evaluation_rules_surf = evaluation_rules["counters_surf"]
    evaluation_ML_real = evaluation_ML["counters_real"]
    evaluation_ML_surf = evaluation_ML["counters_surf"]
    #
    #Initialize containers for confusion matrices
    confmatr_rules_real_abs = np.zeros(shape=(num_real_act,num_real_meas))
    confmatr_ML_real_abs = np.zeros(shape=(num_real_act,num_real_meas))
    confmatr_rules_real_norm =np.ones(shape=(num_real_act,num_real_meas))*np.nan
    confmatr_ML_real_norm = np.ones(shape=(num_real_act,num_real_meas))*np.nan
    #

    ##Calculate confusion matrices: real output classifs case
    #Iterate through actual classifs for papers
    for ii in range(0, num_real_act):
        #Set current matrix index
        y_ind = ii #act_labels.index(real_act_classifs[ii])
        #Iterate through measured classifs for papers
        for jj in range(0, num_real_meas):
            #Set current matrix index
            x_ind = jj
            #if real_meas_classifs[jj] in calc_labels:
            #    x_ind = calc_labels.index(type_overarchs_calc[jj])
            #elif type_overarchs_calc[jj].startswith("false_"):
            #    x_ind = calc_labels.index("none")
            #else:
            #    raise ValueError("Whoa! Unrecognized label {0}!".format(type_overarchs_calc[jj]))
            #

            #Store current matrix value for each approach
            #For rules:
            if do_rules:
                confmatr_rules_real_abs[y_ind,x_ind] += evaluation_rules_real[
                        real_act_classifs[ii]][real_meas_classifs[jj]]
            #
            #For ML:
            if do_ML:
                confmatr_ML_real_abs[y_ind,x_ind] += evaluation_ML_real[
                        real_act_classifs[ii]][real_meas_classifs[jj]]
        #
    #
    #For normalized confusion matrices
    for ii in range(0, num_real_act):
        #Set current matrix index
        y_ind = ii #act_labels.index(type_overarchs_act[ii])
        for jj in range(0, num_real_meas):
            x_ind = jj
            #For rules:
            if do_rules:
                #if confmatr_rules_real_abs[y_ind,x_ind] != 0: #Norm. separately
                confmatr_rules_real_norm[y_ind,x_ind] = (
                        confmatr_rules_real_abs[y_ind,x_ind]
                        / evaluation_rules_real[real_act_classifs[ii]]["total"])
            #
            #For ML:
            if do_ML:
                #if confmatr_ML_real_abs[y_ind,x_ind] != 0: #Normalize separately
                confmatr_ML_real_norm[y_ind,x_ind] = (
                        confmatr_ML_real_abs[y_ind,x_ind]
                        / evaluation_ML_real[real_act_classifs[ii]]["total"])
        #
    #

    ##Plot absolute confusion matrices: real output classifs case
    fig = plt.figure(figsize=figsize)
    #For rule-based approach
    ax = fig.add_subplot(1, 2, 1)
    ax_confusion_matrix(matr=confmatr_rules_real_abs, ax=ax,
        x_labels=real_meas_classifs, y_labels=real_act_classifs,
        y_title="Actual", x_title="Classification", cbar_title="Absolute Count",
        ax_title="Performance: Tree Approach: Uncertainty Thres.={0:.2f}"
                    .format(threshold_rules), cmap=cmap_abs, fontsize=fontsize,
        is_norm=False)
    #
    #For ML-based approach
    ax = fig.add_subplot(1, 2, 2)
    ax_confusion_matrix(matr=confmatr_ML_real_abs, ax=ax,
        x_labels=real_meas_classifs, y_labels=real_act_classifs,
        y_title="Actual", x_title="Classification", cbar_title="Absolute Count",
        ax_title="Performance: ML Approach: Uncertainty Thres.={0:.2f}"
                    .format(threshold_ML), cmap=cmap_abs, fontsize=fontsize,
        is_norm=False)
    #
    #Save and close the figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    plt.savefig(figpath+"/"+figname+"_confmatr_abs_real.png")
    plt.close()
    #
    ##Plot normalized confusion matrices: real output classifs case
    fig = plt.figure(figsize=figsize)
    #For rule-based approach
    ax = fig.add_subplot(1, 2, 1)
    ax_confusion_matrix(matr=confmatr_rules_real_norm, ax=ax,
        x_labels=real_meas_classifs, y_labels=real_act_classifs,
        y_title="Actual", x_title="Classification",
        cbar_title="Fractional Count (Total={0:.0f})"
                    .format(confmatr_rules_real_abs.sum()),
        ax_title="Performance: Tree Approach: Uncertainty Thres.={0:.2f}"
                    .format(threshold_rules), cmap=cmap_abs,fontsize=fontsize,
        is_norm=True)
    #
    #For ML-based approach
    ax = fig.add_subplot(1, 2, 2)
    ax_confusion_matrix(matr=confmatr_ML_real_norm, ax=ax,
        x_labels=real_meas_classifs, y_labels=real_act_classifs,
        y_title="Actual", x_title="Classification",
        cbar_title="Fractional Count (Total={0:.0f})"
                    .format(confmatr_ML_real_abs.sum()),
        ax_title="Performance: ML Approach: Uncertainty Thres.={0:.2f}"
                    .format(threshold_ML), cmap=cmap_abs, fontsize=fontsize,
        is_norm=True)
    #
    #Save and close the figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    plt.savefig(figpath+"/"+figname+"_confmatr_norm_real.png")
    plt.close()
    #

    ##Compute and plot for surface output case, if requested
    if (surface_mapper is not None):
        #Initialize containers for confusion matrices
        confmatr_rules_surf_abs = np.zeros(shape=(num_surf_act,num_surf_meas))
        confmatr_ML_surf_abs = np.zeros(shape=(num_surf_act,num_surf_meas))
        confmatr_rules_surf_norm =np.ones(shape=(num_surf_act,num_surf_meas)
                                            )*np.nan
        confmatr_ML_surf_norm = np.ones(shape=(num_surf_act,num_surf_meas)
                                            )*np.nan
        #
        ##Calculate confusion matrices: surface output classifs case
        #Iterate through actual classifs for papers
        for ii in range(0, num_surf_act):
            #Set current matrix index
            y_ind = ii #act_labels.index(surf_act_classifs[ii])
            #Iterate through measured classifs for papers
            for jj in range(0, num_surf_meas):
                #Set current matrix index
                x_ind = jj
                #if surf_meas_classifs[jj] in calc_labels:
                #    x_ind = calc_labels.index(type_overarchs_calc[jj])
                #elif type_overarchs_calc[jj].startswith("false_"):
                #    x_ind = calc_labels.index("none")
                #else:
                #    raise ValueError("Whoa! Unrecognized label {0}!".format(type_overarchs_calc[jj]))
                #

                #Store current matrix value for each approach
                #For rules:
                if do_rules:
                    confmatr_rules_surf_abs[y_ind,x_ind] += evaluation_rules_surf[
                            surf_act_classifs[ii]][surf_meas_classifs[jj]]
                #
                #For ML:
                if do_ML:
                    confmatr_ML_surf_abs[y_ind,x_ind] += evaluation_ML_surf[
                            surf_act_classifs[ii]][surf_meas_classifs[jj]]
            #
        #
        #For normalized confusion matrices
        for ii in range(0, num_surf_act):
            #Set current matrix index
            y_ind = ii #act_labels.index(type_overarchs_act[ii])
            for jj in range(0, num_surf_meas):
                x_ind = jj
                #For rules:
                if do_rules:
                    if confmatr_rules_surf_abs[y_ind,x_ind] != 0: #Norm. separately
                        confmatr_rules_surf_norm[y_ind,x_ind] = (
                            confmatr_rules_surf_abs[y_ind,x_ind]
                            / evaluation_rules_surf[surf_act_classifs[ii]]["total"])
                #
                #For ML:
                if do_ML:
                    if confmatr_ML_surf_abs[y_ind,x_ind] != 0: #Normalize separately
                        confmatr_ML_surf_norm[y_ind,x_ind] = (
                            confmatr_ML_surf_abs[y_ind,x_ind]
                            / evaluation_ML_surf[surf_act_classifs[ii]]["total"])
            #
        #

        ##Plot absolute confusion matrices: surface output classifs case
        fig = plt.figure(figsize=figsize)
        #For rule-based approach
        ax = fig.add_subplot(1, 2, 1)
        ax_confusion_matrix(matr=confmatr_rules_surf_abs, ax=ax,
            x_labels=surf_meas_classifs, y_labels=surf_act_classifs,
            y_title="Actual", x_title="Classification",
            cbar_title="Absolute Count",
            ax_title="Performance: Tree Approach: Uncertainty Thres.={0:.2f}"
                        .format(threshold_rules),
            cmap=cmap_abs, fontsize=fontsize, is_norm=False)
        #
        #For ML-based approach
        ax = fig.add_subplot(1, 2, 2)
        ax_confusion_matrix(matr=confmatr_ML_surf_abs, ax=ax,
            x_labels=surf_meas_classifs, y_labels=surf_act_classifs,
            y_title="Actual", x_title="Classification",
            cbar_title="Absolute Count",
            ax_title="Performance: ML Approach: Uncertainty Thres.={0:.2f}"
                        .format(threshold_ML),
            cmap=cmap_abs, fontsize=fontsize, is_norm=False)
        #
        #Save and close the figure
        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace)
        plt.savefig(figpath+"/"+figname+"_confmatr_abs_surf.png")
        plt.close()
        #
        ##Plot normalized confusion matrices: surface output classifs case
        fig = plt.figure(figsize=figsize)
        #For rule-based approach
        ax = fig.add_subplot(1, 2, 1)
        ax_confusion_matrix(matr=confmatr_rules_surf_norm, ax=ax,
            x_labels=surf_meas_classifs, y_labels=surf_act_classifs,
            y_title="Actual", x_title="Classification",
            cbar_title="Fractional Count (Total={0:.0f})"
                        .format(confmatr_rules_surf_abs.sum()),
            ax_title="Performance: Tree Approach: Uncertainty Thres.={0:.2f}"
                        .format(threshold_rules), cmap=cmap_abs,
            fontsize=fontsize, is_norm=True)
        #
        #For ML-based approach
        ax = fig.add_subplot(1, 2, 2)
        ax_confusion_matrix(matr=confmatr_ML_surf_norm, ax=ax,
            x_labels=surf_meas_classifs, y_labels=surf_act_classifs,
            y_title="Actual", x_title="Classification",
            cbar_title="Fractional Count (Total={0:.0f})"
                        .format(confmatr_ML_surf_abs.sum()),
            ax_title="Performance: ML Approach: Uncertainty Thres.={0:.2f}"
                        .format(threshold_ML), cmap=cmap_abs,
            fontsize=fontsize, is_norm=True)
        #
        #Save and close the figure
        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace)
        plt.savefig(figpath+"/"+figname+"_confmatr_norm_surf.png")
        plt.close()
    #
#

#Function to compare performance evaluations between two approaches
def compare_performance(evaluation_ML, evaluation_rules, actual_mapper, surface_mapper, figpath, figname, figsize_confmatr, dict_aes, do_plot_bars, figsize=(10, 6), fontsize=16, barwidth=1, alpha=0.75, markerscale=2, do_verbose=True):
    ##Prepare global variables
    tmp_res = fetch_outputs(actual_mapper=actual_mapper,
                            surface_mapper=surface_mapper)
    real_act_classifs = tmp_res["real_act"]
    real_meas_classifs = tmp_res["real_meas"]
    surf_act_classifs = tmp_res["surf_act"]
    surf_meas_classifs = tmp_res["surf_meas"]
    #
    results_rules = evaluation_rules["_dict_results"]
    threshold_rules = evaluation_rules["threshold"]
    results_ML = evaluation_ML["_dict_results"]
    threshold_ML = evaluation_ML["threshold"]
    #
    num_real_act = len(real_act_classifs)
    num_real_meas = len(real_meas_classifs)
    num_verdicts_real = evaluation_rules["counters_real"]["total"]
    if (surface_mapper is not None):
        num_surf_act = len(surf_act_classifs)
        num_surf_meas = len(surf_meas_classifs)
        num_verdicts_surf = evaluation_rules["counters_surf"]["total"]
    #
    #Print some notes
    if do_verbose:
        print("\n\n\nRunning compare_performance!")
        print("Global error verdicts: {0}".format(set_verdicts_errors_global))
        print("Real actual classifs: {0}".format(real_act_classifs))
        print("Real measured classifs: {0}".format(real_meas_classifs))
        print("Surface actual classifs: {0}".format(surf_act_classifs))
        print("Surface measured classifs: {0}".format(surf_meas_classifs))
        print("\n")
    #

    #Prepare counters for statistics
    dict_indiv_real = {item:{"correct_ML":0, "incorrect_ML":0, "none_ML":0,
                        "correct_rules":0, "incorrect_rules":0, "none_rules":0}
                for item in real_act_classifs}
    #
    dict_both_real = {item:{"correct_rules|correct_ML":0, "none_rules|correct_ML":0,
                "correct_rules|none_ML":0, "incorrect_rules|correct_ML":0,
                "correct_rules|incorrect_ML":0, "incorrect_rules|incorrect_ML":0,
                "none_rules|none_ML":0, "none_rules|incorrect_ML":0,
                "incorrect_rules|none_ML":0}
                for item in real_act_classifs}
    #
    if (surface_mapper is not None):
        dict_indiv_surf = dict_indiv_real.copy()
        dict_both_surf = dict_both_real.copy()
    #

    #Extract and verify indices of evaluations
    if (not np.array_equal(np.sort(list(results_rules.keys())),
                            np.sort(list(results_ML.keys())))):
        raise ValueError("Whoa! Different verdict indices!?")
    #
    list_istr = np.sort(list(results_rules.keys()))

    #Iterate through evaluations: real, surface output classif case
    for key_eval in ["real", "surf"]:
        #Extract info for current output scheme
        if (key_eval == "real"):
            curr_mapper = actual_mapper
            curr_dict_indiv = dict_indiv_real
            curr_dict_both = dict_both_real
        #
        elif (key_eval == "surf"):
            if (surface_mapper is not None):
                curr_mapper = surface_mapper
                curr_dict_indiv = dict_indiv_surf
                curr_dict_both = dict_both_surf
            #Otherwise, skip ahead if no surface scheme requested
            else:
                continue
        #
        else:
            raise ValueError("Whoa! {0} not recognized!".format(key_eval))
        #

        #Iterate through paired verdicts
        for curr_istr in list_istr:
            #Extract current verdict info
            curr_rules = results_rules[curr_istr]
            curr_ML = results_ML[curr_istr]
            curr_act_val =curr_rules["act_classif_{0}".format(key_eval)].upper()
            curr_meas_val_rules =curr_rules["meas_classif_"+key_eval].upper()
            curr_meas_val_ML = curr_ML["meas_classif_"+key_eval].upper()

            #Throw error if unequal actual classifs
            if (curr_rules["act_classif_"+key_eval]
                                != curr_ML["act_classif_"+key_eval]):
                raise ValueError("Whoa! Unequal classifs?!\n{0}\n\n{1}"
                                .format(curr_rules, curr_ML))
            #

            #Print some notes
            if do_verbose:
                print("---\nConsidering Verdict {0}:".format(curr_istr))
                print("Eval. scheme: {0}".format(key_eval))
                print("Mission: {0}".format(curr_rules["mission"]))
                #print("Mission: {0}".format(curr_results["mission"]))
                print("Actual vs. measured rules verdict: {0} vs. {1}."
                        .format(curr_act_val, curr_meas_val_rules))
                print("Actual vs. measured ML verdict: {0} vs. {1}."
                        .format(curr_act_val, curr_meas_val_ML))
            #

            #Initialize correction booleans
            is_correct_rules = False
            is_incorrect_rules = False
            is_error_rules = False
            is_correct_ML = False
            is_incorrect_ML = False
            is_error_ML = False

            #Individually categorize rule-based evaluation
            #For correct cases: same val. or same mapping
            if ((curr_act_val == curr_meas_val_rules)
                    or ((curr_act_val in curr_mapper)
                        and (curr_mapper[curr_act_val]==curr_meas_val_rules))):
                curr_dict_indiv[curr_act_val]["correct_rules"] += 1
                is_correct_rules = True
            #For incorrect case
            elif (curr_meas_val_rules in real_act_classifs):
                curr_dict_indiv[curr_act_val]["incorrect_rules"] += 1
                is_incorrect_rules = True
            #For error case
            elif (curr_meas_val_rules in set_verdicts_errors_global):
                curr_dict_indiv[curr_act_val]["none_rules"] += 1
                is_error_rules = True
            #Otherwise, throw error if not recognized
            else:
                raise ValueError("Whoa! Verdict {0} not recognized!"
                                .format(curr_meas_val_rules))
            #

            #Individually categorize ML-based evaluation
            #For correct cases: same val. or same mapping
            if ((curr_act_val == curr_meas_val_ML)
                    or ((curr_act_val in curr_mapper)
                        and (curr_mapper[curr_act_val] == curr_meas_val_ML))):
                curr_dict_indiv[curr_act_val]["correct_ML"] += 1
                is_correct_ML = True
            #For incorrect case
            elif (curr_meas_val_ML in real_act_classifs):
                curr_dict_indiv[curr_act_val]["incorrect_ML"] += 1
                is_incorrect_ML = True
            #For error case
            elif (curr_meas_val_ML in set_verdicts_errors_global):
                curr_dict_indiv[curr_act_val]["none_ML"] += 1
                is_error_ML = True
            #Otherwise, throw error if not recognized
            else:
                raise ValueError("Whoa! Verdict {0} not recognized!"
                                .format(curr_meas_val_ML))
            #

            #Categorize comparison of rule vs. ML evaluation
            #For correct rules, correct ML
            if (is_correct_rules and is_correct_ML):
                curr_dict_both[curr_act_val]["correct_rules|correct_ML"] += 1
            #For correct rules, incorrect ML
            elif (is_correct_rules and is_incorrect_ML):
                curr_dict_both[curr_act_val]["correct_rules|incorrect_ML"] += 1
            #For correct rules, none ML
            elif (is_correct_rules and is_error_ML):
                curr_dict_both[curr_act_val]["correct_rules|none_ML"] += 1
            #For incorrect rules, correct ML
            elif (is_incorrect_rules and is_correct_ML):
                curr_dict_both[curr_act_val]["incorrect_rules|correct_ML"] += 1
            #For incorrect rules, incorrect ML
            elif (is_incorrect_rules and is_incorrect_ML):
                curr_dict_both[curr_act_val]["incorrect_rules|incorrect_ML"] +=1
            #For incorrect rules, none ML
            elif (is_incorrect_rules and is_error_ML):
                curr_dict_both[curr_act_val]["incorrect_rules|none_ML"] += 1
            #For none rules, correct ML
            elif (is_error_rules and is_correct_ML):
                curr_dict_both[curr_act_val]["none_rules|correct_ML"] += 1
            #For none rules, incorrect ML
            elif (is_error_rules and is_incorrect_ML):
                curr_dict_both[curr_act_val]["none_rules|incorrect_ML"] +=1
            #For none rules, none ML
            elif (is_error_rules and is_error_ML):
                curr_dict_both[curr_act_val]["none_rules|none_ML"] += 1
            #Otherwise, throw error if not recognized
            else:
                raise ValueError("Whoa! Verdict pair {0}, {1} not recognized!"
                                .format(curr_meas_val_rules, curr_meas_val_ML))
        #
    #



    ##----------
    ##BAR PLOTS
    ##For real, surface output schemes
    if do_plot_bars:
        for key_eval in ["real", "surf"]:
            #Extract info for current output scheme
            if (key_eval == "real"):
                curr_dict_indiv = dict_indiv_real
                curr_dict_both = dict_both_real
                curr_list_act = real_act_classifs
                curr_tot = num_verdicts_real
            #
            elif (key_eval == "surf"):
                if (surface_mapper is not None):
                    curr_dict_indiv = dict_indiv_surf
                    curr_dict_both = dict_both_surf
                    curr_list_act = surf_act_classifs
                    curr_tot = num_verdicts_surf
                #Otherwise, skip ahead if no surface scheme requested
                else:
                    continue
            #
            else:
                raise ValueError("Whoa! {0} not recognized!".format(key_eval))
            #

            ##PLOT INDIVIDUAL COUNTER STATS
            #Prepare figure
            fig = plt.figure(figsize=figsize)
            groupwidth =(len(curr_dict_indiv[curr_list_act[0]].keys())*barwidth)
            #
            #Plot the bars
            for ii in range(0, len(curr_list_act)):
                curr_overarch = curr_list_act[ii]
                curr_keys = list(curr_dict_indiv[curr_overarch].keys())
                #Iterate through bars
                for jj in range(0, len(curr_keys)):
                    #Set current bar label (if first group)
                    if (ii == 0):
                        label = curr_keys[jj]
                    else:
                        label = None
                    #Set current bar aesthetics
                    color = dict_aes["color"][curr_keys[jj]]
                    marker = dict_aes["marker"][curr_keys[jj]]

                    #Plot the bar
                    xval = (ii*groupwidth) + (jj*barwidth)
                    yval = curr_dict_indiv[curr_overarch][curr_keys[jj]]
                    plt.bar(xval, yval, width=barwidth, color=color,alpha=alpha,
                                label=label, hatch=marker)
                    #

                    #Graph separating line
                    if jj == 0:
                        plt.axvline((xval-(barwidth/2)),
                                    linestyle="--", alpha=0.5, color="gray")
                    elif jj == (len(curr_keys)-1):
                        plt.axvline((xval+(barwidth/2)),
                                    linestyle="--", alpha=0.5, color="gray")
                #
            #
            #Set axes and legend
            plt.xticks((np.arange(0, (len(curr_list_act)*groupwidth),
                        groupwidth)+(groupwidth/2)-(barwidth/2)), curr_list_act)
            plt.xlabel("Actual Classif")
            plt.ylabel("Count of Verdicts")
            plt.legend(loc="best",title="Measured Class",markerscale=markerscale)
            plt.title("Classification of Keyword Paragraphs: {0}".format(curr_tot)
                +"\nProb. Thres. = {0} for ML, {1} for Rule-Based"
                .format(threshold_ML, threshold_rules))
            #
            #Save and close the figure
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(figpath+"/"+figname+"_bars_indiv_{0}.png".format(key_eval))
            plt.close()
            #

            ##PLOT COMBINED COUNTER STATS
            #Prepare figure
            fig = plt.figure(figsize=figsize)
            groupwidth = (len(curr_dict_both[curr_list_act[0]].keys())*barwidth)
            #
            #Plot the bars
            for ii in range(0, len(curr_list_act)):
                curr_overarch = curr_list_act[ii]
                curr_keys = list(curr_dict_both[curr_overarch].keys())
                #Iterate through bars
                for jj in range(0, len(curr_keys)):
                    #Set current bar label (if first group)
                    if (ii == 0):
                        label = curr_keys[jj]
                    else:
                        label = None
                    #Set current bar aesthetics
                    color = dict_aes["color"][curr_keys[jj]]
                    marker = dict_aes["marker"][curr_keys[jj]]

                    #Plot the bar
                    xval = (ii*groupwidth) + (jj*barwidth)
                    yval = curr_dict_both[curr_overarch][curr_keys[jj]]
                    plt.bar(xval, yval, width=barwidth, color=color, alpha=alpha,
                                label=label, hatch=marker)

                    #Graph separating line
                    if jj == 0:
                        plt.axvline((xval-(barwidth/2)),
                                    linestyle="--", alpha=0.5, color="gray")
                    elif jj == (len(curr_keys)-1):
                        plt.axvline((xval+(barwidth/2)),
                                    linestyle="--", alpha=0.5, color="gray")
                #
            #
            #Set axes and legend
            plt.yscale("log")
            plt.xticks((np.arange(0, (len(curr_list_act)*groupwidth),
                        groupwidth)+(groupwidth/2)-(barwidth/2)), curr_list_act)
            plt.xlabel("Actual Class")
            plt.ylabel("Count of Verdicts")
            plt.legend(loc="best", title="Measured Class", markerscale=markerscale)
            plt.title("Classification of Keyword Paragraphs: {0}".format(curr_tot)
                +"\nProb. Thres. = {0} for ML, {1} for Rule-Based"
                .format(threshold_ML, threshold_rules))
            #
            #Save and close the figure
            plt.tight_layout()
            plt.savefig(figpath+"/"+figname+"_bars_both_{0}.png".format(key_eval))
            plt.close()
    #


    ##----------
    ##CONFUSION MATRICES
    ##For real, surface output schemes
    for key_eval in ["real", "surf"]:
        #Extract info for current output scheme
        if (key_eval == "real"):
            curr_dict_both = dict_both_real
            curr_list_act = real_act_classifs
            curr_tot = num_verdicts_real
        #
        elif (key_eval == "surf"):
            if (surface_mapper is not None):
                curr_dict_both = dict_both_surf
                curr_list_act = surf_act_classifs
                curr_tot = num_verdicts_surf
            #Otherwise, skip ahead if no surface scheme requested
            else:
                continue
        #
        else:
            raise ValueError("Whoa! {0} not recognized!".format(key_eval))
        #

        #Initialize containers and parameters for confusion matrices
        curr_num_act = len(curr_list_act)
        gen_labels = np.unique(([key.split("|")[0].split("_")[0]
                            for key in curr_dict_both[real_act_classifs[0]]]
                            +[key.split("|")[1].split("_")[0]
                            for key in curr_dict_both[real_act_classifs[0]]]))
        ndim = len(gen_labels)
        #
        confmatrs_abs = [None]*curr_num_act
        confmatrs_norm = [None]*curr_num_act
        #

        #Fill in the confusion matrices
        for ii in range(0, curr_num_act):
            confmatrs_abs[ii] = np.ones(shape=(ndim,ndim))*np.nan
            #Iterate through ML-based values
            for mm in range(0, ndim):
                #Iterate through rule-based values
                for rr in range(0, ndim):
                    #Fetch counter from current dictionary
                    tmp_key = "{0}_{1}|{2}_{3}".format(
                                gen_labels[rr], "rules", gen_labels[mm], "ML")
                    curr_val = curr_dict_both[real_act_classifs[ii]][tmp_key]
                    #Store current absolute value
                    confmatrs_abs[ii][rr,mm] = curr_val
            #

            #Compute current normalized matrix
            confmatrs_norm[ii] = confmatrs_abs[ii] / np.sum(confmatrs_abs[ii])
        #

        #Plot absolute confusion matrices
        fig = plt.figure(figsize=figsize_confmatr)
        gen_labels_fin = [item.title() for item in gen_labels]
        for ii in range(0, curr_num_act):
            ax = fig.add_subplot(1, curr_num_act, ii+1)
            ax_confusion_matrix(matr=confmatrs_abs[ii], ax=ax,
                    x_labels=gen_labels_fin, y_labels=gen_labels_fin,
                    x_title="ML Approach", y_title="Tree Approach",
                    cbar_title="Absolute Count",
                    ax_title=curr_list_act[ii].title(), cmap=cmap_both,
                    fontsize=fontsize, is_norm=False)
        #
        #Save and close the figure
        plt.tight_layout()
        plt.savefig(figpath+"/"+figname+"_confmatr_abs_"+key_eval+".png")
        plt.close()
        #

        #Plot normalized confusion matrices
        fig = plt.figure(figsize=figsize_confmatr)
        gen_labels_fin = [item.title() for item in gen_labels]
        for ii in range(0, curr_num_act):
            ax = fig.add_subplot(1, curr_num_act, ii+1)
            ax_confusion_matrix(matr=confmatrs_norm[ii], ax=ax,
                    x_labels=gen_labels_fin, y_labels=gen_labels_fin,
                    x_title="ML Approach", y_title="Tree Approach",
                    cbar_title="Fractional Count (Total={0:.0f})"
                                .format(confmatrs_abs[ii].sum()),
                    ax_title=curr_list_act[ii].title(), cmap=cmap_both,
                    fontsize=fontsize, is_norm=True)
        #
        #Save and close the figure
        plt.tight_layout()
        plt.savefig(figpath+"/"+figname+"_confmatr_norm_"+key_eval+".png")
        plt.close()
    #


    ##Exit the function
    return
#
###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###SHARED PREPARATION
if True in [do_evaluate_pipeline_performance, do_evaluate_pipeline_probability, do_evaluate_pipeline_falsehit]:



    #Temporary block for now
    #target_missions = ["HST", "TESS", "JWST", "PanSTARRS", "GALEX", "HLA"] #, "KEPLER", "K2"]
    #keyword_obj_list = [keyword_obj_HST, keyword_obj_TESS, keyword_obj_JWST, keyword_obj_PanSTARRS, keyword_obj_GALEX, keyword_obj_HLA] #, keyword_obj_Kepler, keyword_obj_K2]

    #
    if True in [do_evaluate_pipeline_performance, do_evaluate_pipeline_probability]:
        do_verify_truematch = False #If True, will exclude mismatching ambiguous phrases
    elif True in [do_evaluate_pipeline_falsehit]:
        do_verify_truematch = True #If True, will exclude mismatching ambiguous phrases
    #

    do_rules = True
    do_ML = True
    do_verbose = True
    do_savemisclass = True #If True, will save misclassif. to .txt files
    #
    cmap_indiv = cmasher.freeze_r
    cmap_both = cmasher.voltage_r
    figsize_confmatr = np.array([16,6])*1.4 #0.8
    #
    marker_ML = ""
    marker_rules = "*"
    color_corr = "navy"
    color_incorr = "salmon"
    color_err = "silver"
    color_ML = "royalblue" #"cornflowerblue"
    color_rules = "gold"
    marker_incorr = "x"
    marker_err = "o"
    dict_aes = {"marker":{"correct_ML":marker_ML, "incorrect_ML":marker_ML, "none_ML":marker_ML, "correct_rules":marker_rules, "incorrect_rules":marker_rules, "none_rules":marker_rules, "correct_rules|correct_ML":"", "none_rules|correct_ML":marker_err, "correct_rules|none_ML":marker_err, "incorrect_rules|correct_ML":marker_incorr, "correct_rules|incorrect_ML":marker_incorr, "incorrect_rules|incorrect_ML":"", "none_rules|none_ML":"", "none_rules|incorrect_ML":"|", "incorrect_rules|none_ML":"||"}, "color":{"correct_ML":color_corr, "incorrect_ML":color_incorr, "none_ML":color_err, "correct_rules":color_corr, "incorrect_rules":color_incorr, "none_rules":color_err, "correct_rules|correct_ML":"black", "none_rules|correct_ML":color_ML, "correct_rules|none_ML":color_rules, "incorrect_rules|correct_ML":color_ML, "correct_rules|incorrect_ML":color_rules, "incorrect_rules|incorrect_ML":color_incorr, "none_rules|none_ML":"gray", "none_rules|incorrect_ML":color_incorr, "incorrect_rules|none_ML":color_incorr}}
    #
    figrootname = ("{0}_{1}_{2}_{3}_seedTVT{4}"
                        .format(textform_ML, which_map, which_scheme_TVT,
                                which_set, seed_TVT))
    #
    fileloc_ML = os.path.join(filepath_models_global,
                        fetch_name_MLmodel(name_map=which_map,
                                name_set=which_set, seed=seed_TVT,
                                name_TVT=which_scheme_TVT,
                                name_textform=textform_ML))
    filepath_model = (fileloc_ML+"_metrics.npy")
    #
#
###-----------------------------------------------------------------------------
#



#
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###-----------------------------------------------------------------------------
###SCRIPTS
#Run code to generate combined dataset with text and Papertrack labels
if do_prepare_dataset:
    #Prepare variables
    filename_papertrack = "./../../datasets/papertrack_export_2021-08-18.csv"
    filename_papertext = "./../../datasets/ST_Request2021_use.json"
    filesave_json = filename_json
    filesave_notinpapertrack = "./../../datasets/bibcodes_notin_papertrack.txt"
    keys_papertrack = ['bibcode', 'name_search',
        'is_falsepos', 'papertype', 'mission', 'year_entry']
    keys_papertext = ['abstract', 'author', 'bibcode', 'body', 'keyword',
                        'keyword_norm', 'pubdate', 'title']
    do_verbose = True
    #Run function
    generate_dataset(filename_papertrack=filename_papertrack,
            filename_papertext=filename_papertext, filesave_json=filesave_json,
            filesave_notinpapertrack=filesave_notinpapertrack,
            keys_papertrack=keys_papertrack, keys_papertext=keys_papertext,
            do_save=do_save, do_verbose=do_verbose)
    #
#
#Run code to prepare directory of classified (and maybe saved) paragraphs
elif do_prepare_paragraphs:
    do_verbose = True
    i_truncate = None #25 #None #100
    do_treeerrors = False #For now
    #
    script_process_text(filename_json, keyword_objs=keyword_obj_list,
                        target_missions=target_missions,
                        map_classifs=map_papertypes,
                        filename_bibnotintrack=filename_bibnotintrack,
                        buffer=buffer, do_save=do_save,
                        dir_forsave=dir_forsave, base_forsave=base_forsave,
                        i_truncate=i_truncate, do_verbose=do_verbose,
                        do_treeerrors=do_treeerrors,
                        filename_bibcodes=filename_bibcodes)
#
#Run code to generate TVT directory (for ML use) by copying over statements
elif do_gen_dir_TVT:
    fraction_TVT = np.array([0.80, 0.10, 0.10])
    do_verbose = True
    #
    gen_dir_TVT(dir_forbase=dir_fortext, dir_forTVT=dir_forTVT_global,
                which_scheme=which_scheme_TVT,
                filename_classifs=filename_classifs,map_classifs=map_papertypes,
                filename_bibcodes=filename_bibcodes, fraction_TVT=fraction_TVT,
                randseed=seed_TVT, exp_statement=exp_statement,do_save=do_save,
                buffer=buffer, do_verbose=do_verbose)
#

#Run code to train ML model on specified TVT dataset
elif do_train_ML_model:
    #Set global variables
    converted_classifs = fetch_classifs(mapper=map_papertypes)["converted"]

    #Train a model for each textform
    for curr_textform in classifys_textforms:
        curr_dir_TVT = dir_forTVT_global
        curr_dir_full = os.path.join(curr_dir_TVT, curr_textform)
        curr_filename_model = fetch_name_MLmodel(name_map=which_map,
                                    name_set=which_set, seed=seed_TVT,
                                    name_TVT=which_scheme_TVT,
                                    name_textform=curr_textform)
        #
        #Train an ML model and save if requested
        train_model_ML(do_save=do_save, class_names=converted_classifs,
                    randseed=seed_ML, dir_TVT=curr_dir_full,
                    filename_model=curr_filename_model)
#

#Run code to extract all unique keyword phrases
elif do_extract_keyword_phrases:
    do_recycle = True
    do_falsehits = True
    do_verbose = True
    #
    keyword_obj_HST = bibcat.Keyword(keywords=["Hubble"], acronyms=[])
    keyword_obj_ACS = bibcat.Keyword(keywords=[], acronyms=[])
    keyword_obj_TESS = bibcat.Keyword(keywords=[], acronyms=[])
    keyword_obj_JWST = bibcat.Keyword(keywords=[], acronyms=[])
    keyword_obj_Kepler = bibcat.Keyword(keywords=["Kepler"], acronyms=[])
    keyword_obj_PanSTARRS = bibcat.Keyword(keywords=[], acronyms=[])
    keyword_obj_GALEX = bibcat.Keyword(keywords=[], acronyms=[])
    keyword_obj_K2 = bibcat.Keyword(keywords=["K2"], acronyms=[])
    keyword_obj_HLA = bibcat.Keyword(keywords=[], acronyms=[])
    #
    keyword_obj_ambigs = [keyword_obj_HST, keyword_obj_ACS, keyword_obj_TESS,
                    keyword_obj_JWST, keyword_obj_Kepler, keyword_obj_PanSTARRS,
                    keyword_obj_GALEX, keyword_obj_K2, keyword_obj_HLA]
    #
    num_keyobj = len(keyword_obj_ambigs)
    list_lookups = ["HST", "ACS", "TESS", "JWST", "Kepler", "PanSTARRS",
                    "GALEX", "K2", "HLA"]
    filepath_ambig = os.path.join(filepath_tests_global, "keywords_ambig.txt")
    filepaths_save = [os.path.join(filepath_tests_global,
                                "phrases_keyword_unique_{0}.txt"
                                .format(list_lookups[jj].replace(".","")))
                        for jj in range(0, num_keyobj)]
    #
    #Print some notes
    print("> Running do_extract_keyword_phrases()!")
    #

    #Load papers from the dataset
    if do_recycle and ("evaluations" not in locals()):
        dict_papers = extract_papers_from_dataset(
            map_classifs_orig=map_papertypes, which_missions=target_missions,
            which_acronym=which_specificacronym, dir_test=None,
            max_papers=max_papers, do_test_only=False,
            do_falsehits=do_falsehits)
        num_papers = len(dict_papers) #Count of papers fitting criteria
        paper_ids = list(dict_papers.keys())
    #

    #Extract keyword phrases from each paper
    set_phrases = [set() for jj in range(0, num_keyobj)]
    list_phrases = [[] for jj in range(0, num_keyobj)]
    #list_verdicts = [[] for jj in range(0, num_keyobj)]
    list_comb = [[] for jj in range(0, num_keyobj)]
    dict_phrases_indiv = [{} for jj in range(0, num_keyobj)]
    #Iterate through papers
    for ii in range(0, num_papers):
        #Print some notes
        if do_verbose:
            print("---"*20)
            print("Fetching keyword phrases from paper {0} (of {1})."
                    .format(ii, num_papers))
        #

        #Extract the current text
        curr_paperid = paper_ids[ii]
        curr_text = dict_papers[curr_paperid]["body"]
        #

        #Iterate through keyword objects
        for jj in range(0, num_keyobj):
            curr_keyobj = keyword_obj_ambigs[jj]
            curr_verdict = list(dict_papers[curr_paperid]["class_missions"
                                                            ].keys())
            #Search the current text for phrases
            curr_paper = bibcat.Paper(text=curr_text,keywords_obj=[curr_keyobj])
            curr_paper.process_phrases(do_exclude_mismatches=False)
            curr_phrases_indiv = curr_paper.get_phrases(keyword_obj=None)
            #

            #Store the current phrases
            dict_phrases_indiv[jj][curr_paperid] = curr_phrases_indiv

            #Store the unique phrases
            curr_unique = [item for item in curr_phrases_indiv
                                if (item.lower() not in set_phrases[jj])]
            set_phrases[jj].update([item.lower() for item in curr_unique])
            list_phrases[jj] += curr_unique
            #list_verdicts[jj] += [curr_verdict for item in curr_unique]
            list_comb[jj] += ["{0}\t{1}".format(curr_unique[zz],
                                                curr_verdict)
                            for zz in range(0, len(curr_unique))]
            #

            #Print some notes
            if do_verbose:
                print("Done with {0}:{1}.\nLatest: {2}\nUnique Subset: {3}"
                        .format(ii, list_lookups[jj], curr_phrases_indiv,
                                curr_unique))
            #

            #Save the latest list of unique keyword phrases
            if do_save:
                if do_verbose:
                    print("Saving the latest version of extracted phrases...")
                #
                #str_save = "\n".join(list_phrases[jj])
                str_save = "\n".join(list_comb[jj])
                write_text(text=str_save, filepath=filepaths_save[jj])
            #
        #
    #

    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Done extracting phrases from papers!")
    #
    #Save a final sorted list of unique keyword phrases
    if do_save:
        if do_verbose:
            print("Saving a final, sorted version of extracted phrases...")
        #
        for jj in range(0, num_keyobj):
            #str_save = "\n".join(sorted(list_phrases[jj]))
            str_save = "\n".join(sorted(list_comb[jj]))
            write_text(text=str_save, filepath=filepaths_save[jj])
        #
        if do_verbose:
            print("Saving a list of the ambiguous keywords...")
        #
        str_save = "\n".join([item2 for item in keyword_obj_ambigs
                            for item2 in item._get_info("keywords")])
        write_text(text=str_save, filepath=filepath_ambig)
    #
    print("Done!")
    #
#

#Run code to determine and extract missing branches from rule-based decision tree
elif do_extract_missing_branches:
    #Initialize rule-based classifier to generate decision tree
    flowerpot = bibcat.Classifier_Rules(do_verbose=False)

    #Generate and extract missing branches
    missing_branches = flowerpot._find_missing_branches(do_verbose=True)
    num_branches = len(missing_branches)

    #Save stats and info for the missing branches
    print("Number of missing branches: {0}".format(num_branches))
    print("Saving to file...")
    str_save = ""
    for ii in range(0, num_branches):
        str_save += "{0}: {1}\n\n".format(ii, missing_branches[ii])
    #
    write_text(text=str_save,
            filepath=os.path.join(filepath_tests_global,
                                "output_missing_branches.txt"))
    #
#

#Run code to evaluate and plot performance of full pipeline (rule+ML)
elif do_evaluate_pipeline_performance:
    do_recycle = True
    threshold_ML = 0.8 #0.9 #9 #5 #0.9 #0.9
    threshold_rules = 0.55 # threshold_ML #None #0.9 #0.9
    do_falsehits = False
    do_raise_innererror = False #True
    #
    do_plot_bars = False
    surface_mapper = None
    #
    if True: #For ease of minimizing content below
        figname_ML = ("fig_perf_ML_{0}_P{1}ov100"
                    .format(figrootname, int(threshold_ML*100)))
        figname_rules = ("fig_perf_rules_{0}_P{1}ov100"
                    .format(figrootname, int(threshold_rules*100)))
        figname_indiv = (
            "fig_perf_indiv_{0}_PofML{1}ov100_Prules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
        figname_compare = (
            "fig_perf_compare_{0}_PML{1}ov100_Prules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
        fileroot_savemisclass = os.path.join(figpath_pipeline,
            "misclass_perf_{0}_PofML{1}ov100_Pofrules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Running do_evaluate_pipeline_performance!")
    #


    ##Extract all papers from the database that qualify for the given set
    if do_recycle and ("evaluations" not in locals()):
        dataset_keep = extract_papers_from_dataset(
            map_classifs_orig=map_papertypes,
            which_missions=target_missions, which_acronym=which_specificacronym,
            dir_test=os.path.join(dir_forTVT_global, textform_ML, "test"),
            max_papers=max_papers, do_test_only=True, do_falsehits=do_falsehits)
        num_datakeep = len(dataset_keep) #Count of papers fitting criteria
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Done iterating through papers.")
        print("Total number of papers kept: {0}\n".format(num_datakeep))
        print("\nPerforming evaluation of the pipeline...")
    #


    ##Evaluate the full pipeline for all classifier approaches
    if do_recycle and ("evaluations" not in locals()):
        evaluations = evaluate_pipeline(dict_papers=dataset_keep,
                all_keyobjs=keyword_obj_list, actual_mapper=map_papertypes,
                threshold_ML=threshold_ML, threshold_rules=threshold_rules,
                surface_mapper=surface_mapper, filepath_model=filepath_model,
                do_verbose=do_verbose, do_rules=do_rules, do_ML=do_ML,
                do_savemisclass=do_savemisclass, which_textform=textform_new,
                fileroot_savemisclass=fileroot_savemisclass,
                fileloc_ML=fileloc_ML,
                do_raise_innererror=do_raise_innererror,
                do_verify_truematch=do_verify_truematch, do_allkeyobjs=do_falsehits)
    #


    ##Plot the performance in bar plot form, if requested
    if do_plot_bars:
        #For rule-based approach
        if do_rules:
            print("Plotting absolute version for rule-based approach...")
            plot_performance_bars(evaluation=evaluations["rule"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_rules,
                    title=("Rule: "+figname_rules+": Absolute"))
            print("Plotting fractional version for rule-based approach...")
            plot_performance_bars(evaluation=evaluations["rule"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline,figname=figname_rules,do_norm=True,
                    title=("Rule: "+figname_rules+": Fraction"))
        #
        #For ML-based approach
        if do_ML:
            print("Plotting absolute version for ML-based approach...")
            plot_performance_bars(evaluation=evaluations["ML"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_ML,
                    title=("ML: "+figname_ML+": Absolute"))
            print("Plotting fractional version for ML-based approach...")
            plot_performance_bars(evaluation=evaluations["ML"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_ML, do_norm=True,
                    title=("ML: "+figname_ML+": Fraction"))
    #

    #For individual confusion matrices
    plot_performance_confusion_matrix(evaluation_rules=evaluations["rule"],
            evaluation_ML=evaluations["ML"], figpath=figpath_pipeline,
            figname=figname_indiv, figsize=figsize_confmatr,
            actual_mapper=map_papertypes, surface_mapper=surface_mapper,
            cmap_abs=cmap_indiv, cmap_norm=cmap_indiv,
            threshold_rules=threshold_rules, threshold_ML=threshold_ML)

    #For comparison between approaches
    if (do_rules and do_ML):
        compare_performance(evaluation_ML=evaluations["ML"],
            evaluation_rules=evaluations["rule"], figpath=figpath_pipeline,
            figsize_confmatr=figsize_confmatr, figname=figname_compare,
            actual_mapper=map_papertypes, surface_mapper=surface_mapper,
            dict_aes=dict_aes, do_plot_bars=do_plot_bars)
    #
    print("Done!")
    #
#

#Run code to evaluate and plot performance of full pipeline (rule+ML) for variety of probability thresholds
elif do_evaluate_pipeline_probability:
    do_recycle = True
    do_falsehits = False
    do_raise_innererror = False
    list_show_rules = ["data_influenced", "mention", "science", "supermention", "z_error", "z_lowprob"] #None
    list_show_ML = ["data_influenced", "mention", "science", "supermention", "z_error", "z_lowprob"] #None
    arr_threshold = np.asarray((np.arange(0.5, 0.96, 0.05).tolist() + [0.99]))
    num_probs = len(arr_threshold)
    #
    surface_mapper = None
    #
    if True: #For ease of minimizing content below
        figname_ML = "fig_gridprob_ML_{0}".format(figrootname)
        figname_rules = "fig_gridprob_rules_{0}".format(figrootname)
        fileroot_savemisclass_raw = [("misclass_gridprob_{0}_{1:.2f}"
                                .format(figrootname,
                                        arr_threshold[ii]).replace(".","p"))
                                for ii in range(0, num_probs)]
        fileroot_savemisclass_arr = [os.path.join(figpath_pipeline,
                                    fileroot_savemisclass_raw[ii])
                                    for ii in range(0, num_probs)]
        #
        figsize = np.array([20, 10])*0.8
        ncol = 3
        tmp_colors = {"MENTION":"tomato", "SCIENCE":"dodgerblue",
                    "SUPERMENTION":"crimson", "DATA_INFLUENCED":"purple",
                    "Z_ERROR":"gray", "Z_FALSEHIT":"green",
                    "Z_LOWPROB":"black", "Z_NOTMATCH":"darkgoldenrod"}
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Running do_evaluate_pipeline_probability!")
    #

    ##Prepare global variables
    tmp_res = fetch_outputs(actual_mapper=map_papertypes,
                            surface_mapper=surface_mapper)
    real_act_classifs = tmp_res["real_act"]
    real_meas_classifs = tmp_res["real_meas"]
    num_real_act = len(real_act_classifs)
    num_real_meas = len(real_meas_classifs)
    #
    if (surface_mapper is not None):
        surf_act_classifs = tmp_res["surf_act"]
        surf_meas_classifs = tmp_res["surf_meas"]
        num_surf_act = len(surf_act_classifs)
        num_surf_meas = len(surf_meas_classifs)
    #

    ##Extract all papers from the database that qualify for the given set
    if do_recycle and ("evaluations" not in locals()):
        dataset_keep = extract_papers_from_dataset(
            map_classifs_orig=map_papertypes,
            which_missions=target_missions, which_acronym=which_specificacronym,
            dir_test=os.path.join(dir_forTVT_global, textform_ML, "test"),
            max_papers=max_papers, do_test_only=True, do_falsehits=False)
        num_datakeep = len(dataset_keep) #Count of papers fitting criteria
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Done iterating through papers.")
        print("Total number of papers kept: {0}\n".format(num_datakeep))
        print("\nPerforming evaluation over the probability grid...")
    #

    ##Evaluate the full pipeline for all classifier approaches and prob. values
    if do_recycle and ("list_res" not in locals()):
        list_res = [None]*num_probs
        for ii in range(0, num_probs):
            curr_prob = arr_threshold[ii]
            #Print some notes
            if do_verbose:
                print("\n\n\n")
                print("---"*100)
                print("Evaluating {0} papers for probability {1} (of {2}-{3}):"
                        .format(max_papers, curr_prob,
                                arr_threshold[0], arr_threshold[-1]))
            #
            #Store current evaluation
            list_res[ii] = evaluate_pipeline(dict_papers=dataset_keep,
                    all_keyobjs=keyword_obj_list, actual_mapper=map_papertypes,
                    threshold_ML=curr_prob, threshold_rules=curr_prob,
                    surface_mapper=surface_mapper,filepath_model=filepath_model,
                    do_verbose=do_verbose, do_rules=do_rules, do_ML=do_ML,
                    do_savemisclass=do_savemisclass, which_textform=textform_new,
                    fileroot_savemisclass=fileroot_savemisclass_arr[ii],
                    fileloc_ML=fileloc_ML,
                    do_raise_innererror=do_raise_innererror,
                    do_verify_truematch=do_verify_truematch,
                    do_allkeyobjs=do_falsehits)
        #
        #Save the results to a dictionary as well
        np.save(os.path.join(filepath_tests_global, "output_perf_prob.npy"),
                            {"probs":arr_threshold,
                            "results":list_res})
        #
    #

    ##Set parameters for mapped iterators as necessary
    ext_mapped = ["real"]
    if (surface_mapper is not None):
        ext_mapped += ["surf"]
    #

    ##Extract and plot the performance per classif as a function of prob.
    #Iterate through mapped schemes
    for mm in range(0, len(ext_mapped)):
        #Extract current mapped scheme
        #For 'real' scheme
        if (ext_mapped[mm] == "real"):
            curr_list_act = real_act_classifs
            curr_list_meas = real_meas_classifs
            curr_num_act = num_real_act
            curr_num_meas = num_real_meas
        #For 'surf' scheme
        elif (ext_mapped[mm] == "surf"):
            curr_list_act = surf_act_classifs
            curr_list_meas = surf_meas_classifs
            curr_num_act = num_surf_act
            curr_num_meas = num_surf_meas
        #Otherwise, throw error if not recognized
        else:
            raise ValueError("Whoa! {0} not recognized!".format(ext_mapped))
        #

        #Initialize containers for counts of measured classifs
        curr_counter_rules = {key1:{key2:(np.ones(num_probs)*np.nan)
                                    for key2 in curr_list_meas}
                                for key1 in curr_list_act}
        curr_counter_ML = {key1:{key2:(np.ones(num_probs)*np.nan)
                                    for key2 in curr_list_meas}
                                for key1 in curr_list_act}
        #

        #Iterate through probabilities
        for ii in range(0, num_probs):
            #Print some notes
            if do_verbose:
                print("Considering probability {0}:{1:.2f}:"
                        .format(ext_mapped[mm], arr_threshold[ii]))
            #

            #Extract current evaluations
            tmp_key = "counters_{0}".format(ext_mapped[mm])
            curr_eval_rules = list_res[ii]["rule"][tmp_key]
            curr_eval_ML = list_res[ii]["ML"][tmp_key]

            #Extract and store counters
            for jj in range(0, curr_num_act):
                for kk in range(0, curr_num_meas):
                    #For rules:
                    curr_counter_rules[curr_list_act[jj]][curr_list_meas[kk]
                                        ][ii] = curr_eval_rules[
                                        curr_list_act[jj]][curr_list_meas[kk]]
                    #For ML:
                    curr_counter_ML[curr_list_act[jj]][curr_list_meas[kk]
                                        ][ii] = curr_eval_ML[
                                        curr_list_act[jj]][curr_list_meas[kk]]
            #
        #

        #Plot the absolute evaluation results
        fig_rules = plt.figure(figsize=figsize)
        fig_ML = plt.figure(figsize=figsize)
        nrow = (curr_num_act // ncol) + int((curr_num_act % ncol != 0))
        #Iterate through actual classifs
        for jj in range(0, curr_num_act):
            ax0_rules = fig_rules.add_subplot(nrow, ncol, (jj+1))
            ax0_ML = fig_ML.add_subplot(nrow, ncol, (jj+1))
            #Iterate through measured classifs
            for kk in range(0, curr_num_meas):
                if (curr_list_meas[kk] == preset.verdict_lowprob.upper()):
                    tmp_linestyle = "--"
                    tmp_linewidth = 4
                elif (curr_list_meas[kk].startswith("Z_")): #For errors, etc.
                    tmp_linestyle = ":"
                    tmp_linewidth = 3
                elif (curr_list_meas[kk] == curr_list_act[jj]):
                    tmp_linestyle = "-"
                    tmp_linewidth = 8
                else:
                    tmp_linestyle = "-"
                    tmp_linewidth = 4
                #
                #For rules:
                tmp_bool = True
                if ((list_show_rules is not None)
                        and (not any([bool(re.search(curr_list_meas[kk], item,
                                                flags=re.IGNORECASE))
                                    for item in list_show_rules]))):
                    tmp_bool = False
                if tmp_bool: #Show if main class or z_error with >0 values
                    ax0_rules.plot(arr_threshold,
                                curr_counter_rules[curr_list_act[jj]
                                                    ][curr_list_meas[kk]],
                                label="{0}".format(curr_list_meas[kk]),
                                marker="o",
                                alpha=0.75,color=tmp_colors[curr_list_meas[kk]],
                                linewidth=tmp_linewidth,linestyle=tmp_linestyle)
                #
                #For ML:
                tmp_bool = True
                if ((list_show_ML is not None)
                        and (not any([bool(re.search(curr_list_meas[kk], item,
                                                flags=re.IGNORECASE))
                                    for item in list_show_ML]))):
                    tmp_bool = False
                if tmp_bool: #Show if main class or z_error with >0 values
                    ax0_ML.plot(arr_threshold,
                                curr_counter_ML[curr_list_act[jj]
                                                    ][curr_list_meas[kk]],
                                label="{0}".format(curr_list_meas[kk]),
                                marker="o",
                                alpha=0.75,color=tmp_colors[curr_list_meas[kk]],
                                linewidth=tmp_linewidth,linestyle=tmp_linestyle)
                #
            #
            #Label the subplots
            ax0_rules.set_xlabel("Uncertainty Threshold")
            ax0_rules.set_ylabel("Count of Classifications")
            ax0_rules.set_title(curr_list_act[jj])
            ax0_ML.set_xlabel("Uncertainty Threshold")
            ax0_ML.set_ylabel("Count of Classifications")
            ax0_ML.set_title(curr_list_act[jj])
            #
            #Generate legends
            if (jj == (curr_num_act - 1)):
                leg = ax0_rules.legend(loc="best", frameon=False)
                leg = ax0_ML.legend(loc="best", frameon=False)
            #
            #Save the figures
            tmp_ext = "_{0}_abs.png".format(ext_mapped[mm])
            fig_rules.tight_layout()
            fig_rules.savefig((os.path.join(figpath_pipeline,
                                            figname_rules)+tmp_ext))
            plt.close(fig_rules)
            fig_ML.tight_layout()
            fig_ML.savefig((os.path.join(figpath_pipeline,
                                            figname_ML)+tmp_ext))
            plt.close(fig_ML)
        #
    #
    print("Done!")
    #
#

#Run code to evaluate and plot performance of full pipeline (rule+ML)
elif do_evaluate_pipeline_falsehit:
    do_recycle = True
    do_raise_innererror = False
    threshold_ML = 0.9 #9 #5 #0.9 #0.9
    threshold_rules = threshold_ML #None #0.9 #0.9
    do_falsehits = True
    do_verbose_mid = True
    #
    do_plot_bars = False
    surface_mapper = {"SCIENCE":"ACCEPTED", "MENTION":"ACCEPTED",
                        "SUPERMENTION":"ACCEPTED", "DATA_INFLUENCED":"ACCEPTED",
                        preset.verdict_error.upper():"ACCEPTED",
                        preset.verdict_lowprob.upper():"ACCEPTED",
                        #preset.verdict_falsehit.upper():"REJECTED",
                        preset.verdict_rejection.upper():"REJECTED"}
    #
    if True: #For ease of minimizing content below
        figname_ML = ("fig_accepts_ML_{0}_P{1}ov100"
                    .format(figrootname, int(threshold_ML*100)))
        figname_rules = ("fig_accepts_rules_{0}_P{1}ov100"
                    .format(figrootname, int(threshold_rules*100)))
        figname_indiv = (
            "fig_accepts_indiv_{0}_PofML{1}ov100_Prules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
        figname_compare = (
            "fig_accepts_compare_{0}_PML{1}ov100_Prules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
        fileroot_savemisclass = os.path.join(figpath_pipeline,
            "misclass_accepts_{0}_PofML{1}ov100_Pofrules{2}ov100"
                    .format(figrootname,
                            int(threshold_ML*100), int(threshold_rules*100)))
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Running do_evaluate_pipeline_performance!")
    #


    ##Extract all papers from the database that qualify for the given set
    if do_recycle and ("evaluations" not in locals()):
        dataset_keep = extract_papers_from_dataset(
            map_classifs_orig=map_papertypes,
            which_missions=target_missions, which_acronym=which_specificacronym,
            dir_test=os.path.join(dir_forTVT_global, textform_ML, "test"),
            max_papers=max_papers, do_test_only=False,
            do_falsehits=do_falsehits)
        num_datakeep = len(dataset_keep) #Count of papers fitting criteria
    #
    #Print some notes
    if do_verbose:
        print("\n")
        print("--"*30)
        print("Done iterating through papers.")
        print("Total number of papers kept: {0}\n".format(num_datakeep))
        print("\nPerforming evaluation of the pipeline...")
    #


    ##Evaluate the full pipeline for all classifier approaches
    if do_recycle and ("evaluations" not in locals()):
        evaluations = evaluate_pipeline(dict_papers=dataset_keep,
                all_keyobjs=keyword_obj_list, actual_mapper=map_papertypes,
                threshold_ML=threshold_ML, threshold_rules=threshold_rules,
                surface_mapper=surface_mapper, filepath_model=filepath_model,
                do_verbose=do_verbose, do_rules=do_rules, do_ML=do_ML,
                do_savemisclass=do_savemisclass, which_textform=textform_new,
                fileroot_savemisclass=fileroot_savemisclass,
                fileloc_ML=fileloc_ML,
                do_raise_innererror=do_raise_innererror,
                do_verify_truematch=do_verify_truematch,
                do_verbose_deep=do_verbose_mid, do_allkeyobjs=do_falsehits)
    #


    ##Plot the performance in bar plot form, if requested
    if do_plot_bars:
        #For rule-based approach
        if do_rules:
            print("Plotting absolute version for rule-based approach...")
            plot_performance_bars(evaluation=evaluations["rule"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_rules,
                    title=("Rule: "+figname_rules+": Absolute"))
            print("Plotting fractional version for rule-based approach...")
            plot_performance_bars(evaluation=evaluations["rule"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline,figname=figname_rules,do_norm=True,
                    title=("Rule: "+figname_rules+": Fraction"))
        #
        #For ML-based approach
        if do_ML:
            print("Plotting absolute version for ML-based approach...")
            plot_performance_bars(evaluation=evaluations["ML"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_ML,
                    title=("ML: "+figname_ML+": Absolute"))
            print("Plotting fractional version for ML-based approach...")
            plot_performance_bars(evaluation=evaluations["ML"],
                    actual_mapper=map_papertypes, surface_mapper=surface_mapper,
                    figpath=figpath_pipeline, figname=figname_ML, do_norm=True,
                    title=("ML: "+figname_ML+": Fraction"))
    #

    #For individual confusion matrices
    plot_performance_confusion_matrix(evaluation_rules=evaluations["rule"],
            evaluation_ML=evaluations["ML"], figpath=figpath_pipeline,
            figname=figname_indiv, figsize=figsize_confmatr,
            actual_mapper=map_papertypes, surface_mapper=surface_mapper,
            cmap_abs=cmap_indiv, cmap_norm=cmap_indiv,
            threshold_ML=threshold_ML, threshold_rules=threshold_rules)
    #
    print("Done!")
    #
#

###-----------------------------------------------------------------------------
#


















#
