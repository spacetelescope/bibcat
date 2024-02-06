###FILE: bibcat_tests.py
###PURPOSE: Container for all tests run for the bibcat package.
###DATA CREATED: 2022-02-28
###DEVELOPERS: (Jamila Pegues, Others)
# python -m unittest test_bibcat.py


###Import necessary modules
import json
import os
import unittest

import numpy as np
import spacy
from nltk.corpus import wordnet

from bibcat import config
from bibcat import parameters as params
from bibcat.core import base, grammar, keyword, operator, paper

nlp = spacy.load(config.spacy_language_model)
#
# -------------------------------------------------------------------------------
###Set global test variables
# Fetch filepaths for model and data
name_model = config.name_model
filepath_input = config.PATH_INPUT
filepath_papertrack = config.path_papertrack
filepath_papertext = config.path_papertext
filepath_dictinfo = config.path_TVTinfo
filepath_modiferrors = config.path_modiferrors
#

##Grammar
test_which_modes = ["none", "skim", "trim", "anon", "skim_trim_anon"]
#

##Mission terms
# Keyword objects
kobj_hla = params.keyword_obj_HLA
kobj_hubble = params.keyword_obj_HST
kobj_kepler = params.keyword_obj_Kepler
kobj_k2 = params.keyword_obj_K2
#
# Keyword-object lookups
list_lookup_kobj = [kobj_hubble, kobj_kepler, kobj_k2]
dict_lookup_kobj = {"Hubble": kobj_hubble, "Kepler": kobj_kepler, "K2": kobj_k2, "HLA": kobj_hla}
#

##Placeholders
placeholder_anon = config.placeholder_anon
placeholder_author = config.placeholder_author
placeholder_number = config.placeholder_number
placeholder_numeric = config.placeholder_numeric
placeholder_website = config.placeholder_website
#

##Classifier setup
mapper = params.map_papertypes
all_kobjs = [
    params.keyword_obj_HST,
    params.keyword_obj_JWST,
    params.keyword_obj_TESS,
    params.keyword_obj_Kepler,
    params.keyword_obj_PanSTARRS,
    params.keyword_obj_GALEX,
    params.keyword_obj_K2,
    params.keyword_obj_HLA,
]
allowed_classifications = ["SCIENCE", "DATA_INFLUENCED", "MENTION", "SUPERMENTION"]
#
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
###Test Classes
#
# """
# class: TestData
# Purpose: Testing datasets
class TestData(unittest.TestCase):
    # For tests of .json dataset file combined from papertrack and papertext:
    if True:
        # Test that combined .json file components are correct
        def test__combined_dataset(self):
            print("Running test__combined_dataset.")
            # Load each of the datasets
            # For the combined dataset
            with open(filepath_input, "r") as openfile:
                data_combined = json.load(openfile)
            #
            # For the original text data
            with open(filepath_papertext, "r") as openfile:
                data_text = json.load(openfile)
            #
            # For the original classification data
            with open(filepath_papertrack, "r") as openfile:
                data_classif = json.load(openfile)
            #

            # Build list of bibcodes for the original data sources
            list_bibcodes_text = [item["bibcode"] for item in data_text]
            list_bibcodes_classif = [item["bibcode"] for item in data_classif]

            # Check each combined data entry against original data sources
            for ii in range(0, len(data_combined)):
                # Skip if no text stored for this entry
                if "body" not in data_combined[ii]:
                    print("test__combined_dataset: No text for index {0}.".format(ii))
                    continue

                # Extract bibcode
                curr_bibcode = data_combined[ii]["bibcode"]  # Curr. bibcode

                # Fetch indices of entries in original data sources
                ind_text = list_bibcodes_text.index(curr_bibcode)
                try:
                    ind_classif = list_bibcodes_classif.index(curr_bibcode)
                except ValueError:
                    print("test__combined_dataset: {0} not classified bibcode.".format(curr_bibcode))
                    continue
                #

                # Verify that combined data entry values match back to originals
                # Check abstract, if exists
                try:
                    if "abstract" in data_combined[ii]:
                        self.assertEqual(data_combined[ii]["abstract"], data_text[ind_text]["abstract"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Diff. abstract in bibcode {0}:\n\n{1}...\nvs\n{2}...".format(
                            curr_bibcode, data_combined[ii]["abstract"][0:500], data_text[ind_text]["abtract"][0:500]
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()
                #
                # Check text
                try:
                    self.assertEqual(data_combined[ii]["body"], data_text[ind_text]["body"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Different text in bibcode {0}:\n\n{1}...\nvs\n{2}...".format(
                            curr_bibcode, data_combined[ii]["body"][0:500], data_text[ind_text]["body"][0:500]
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()
                #
                # Check bibcodes (redundant test but that's ok)
                try:
                    self.assertEqual(curr_bibcode, data_text[ind_text]["bibcode"])
                    self.assertEqual(curr_bibcode, data_classif[ind_classif]["bibcode"])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        ("Different bibcodes:\nCombined: {0}" + "\nText: {1}\nClassif: {2}").format(
                            curr_bibcode, data_text[ind_text]["bibcode"], data_classif[ind_classif]["bibcode"]
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()
                #
                # Check missions and classes
                try:
                    self.assertEqual(
                        len(data_combined[ii]["class_missions"]), len(data_classif[ind_classif]["class_missions"])
                    )  # Ensure equal number of classes
                    for curr_mission in data_combined[ii]["class_missions"]:
                        tmp_list = [item["mission"] for item in data_classif[ind_classif]["class_missions"]]
                        tmp_ind = tmp_list.index(curr_mission)
                        self.assertEqual(
                            data_combined[ii]["class_missions"][curr_mission]["papertype"],
                            data_classif[ind_classif]["class_missions"][tmp_ind]["paper_type"],
                        )
                except (IndexError, AssertionError) as err:
                    print("")
                    print(">")
                    print(
                        ("Different missions and classes:\nCombined: {0}" + "\nClassif: {1}").format(
                            data_combined[ii]["class_missions"], data_classif[ind_classif]["class_missions"]
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()
                #
            #
            print("Run of test__combined_dataset complete.")

        #
        # Test that files of TVT directory (if exist) are correctly stored
        def test__TVT_directory(self):
            print("Running test__TVT_directory.")
            # Load the datasets
            # Check if TVT information exists
            if not os.path.isfile(filepath_dictinfo):
                raise AssertionError("TVT directory does not exist yet at: {0}".format(filepath_dictinfo))
            #
            # If exists, carry on with this test
            dict_info = np.load(filepath_dictinfo, allow_pickle=True).item()
            dict_errors = np.load(filepath_modiferrors, allow_pickle=True).item()
            # Dataset of text and classification information
            with open(filepath_input, "r") as openfile:
                dataset = json.load(openfile)
            #
            list_bibcodes_dataset = [item["bibcode"] for item in dataset]

            # Ensure each entry in TVT storage has correct mission and class
            for curr_key in dict_info:
                ind_dataset = list_bibcodes_dataset.index(curr_key)
                # Fetch only allowed missions and classifs. from dataset
                curr_actuals = {
                    key: dataset[ind_dataset]["class_missions"][key]
                    for key in dataset[ind_dataset]["class_missions"]
                    if (
                        (any([item.is_keyword(key) for item in all_kobjs]))
                        and (dataset[ind_dataset]["class_missions"][key]["papertype"] in allowed_classifications)
                    )
                }
                #
                # Fetch results of test and combine with any errors for bibcode
                curr_test = dict_info[curr_key]["storage"].copy()
                if curr_key in dict_errors:
                    curr_test.update(dict_errors[curr_key])
                #
                # Check each entry
                try:
                    ind_dataset = list_bibcodes_dataset.index(curr_key)
                    # Ensure same number of missions for this bibcode
                    self.assertEqual(len(curr_actuals), len(curr_test))
                    #
                    for curr_textid in curr_test:
                        curr_mission = curr_test[curr_textid]["mission"]
                        curr_class = curr_test[curr_textid]["class"]
                        # Check each mission and class
                        self.assertEqual(curr_class, mapper[curr_actuals[curr_mission]["papertype"].lower()])
                    #
                #
                except (KeyError, AssertionError) as err:
                    print("")
                    print(">")
                    print(
                        ("Diff mission info for {0}:\n{1}\n-\n" + "{2}\nand {3}\n-\nvs\n{4}\n---\n").format(
                            curr_key, curr_test, dict_info[curr_key], dict_errors[curr_key], curr_actuals
                        )
                    )
                    print("---")
                    print("")
                    #
                    raise AssertionError()
                #
            #
            print("Run of test__TVT_directory complete.")

        #
    #


# """


# class: TestBase
# Purpose: Testing the Base class
class TestBase(unittest.TestCase):
    # For tests of _assemble_keyword_wordchunks:
    if True:
        # Test determination of wordchunks around keywords
        def test__assemble_keyword_wordchunks__variety(self):
            # Prepare text and answers for test
            dict_acts_noverbs = {
                "The Hubble calibrated data was used.": ["Hubble calibrated data"],
                "The Hubble database is quite expansive.": ["Hubble database"],
                "Consider reading the paper Hubble (2000).": ["paper Hubble"],
                "The scientists used the HST PSF.": ["HST PSF"],
                "The Hubble Telescope imaged the data.": ["Hubble Telescope"],
                "HST and Hubble both refer to the Hubble Telescope": ["HST", "Hubble", "Hubble Telescope"],
                "hubble and roman are both telescopes": ["hubble"],
                "roman and hubble are both telescopes": ["hubble"],
                "roman, hubble, and kepler are all telescopes": ["hubble", "kepler"],
                "H.S.T stands for Hubble Space Telescope, which is part of an observatory effort.": [
                    "H.S.T",
                    "Hubble Space Telescope",
                ],
                "We tried analysis with Hubble data, before then using a Kepler approach.": [
                    "Hubble data",
                    "Kepler approach",
                ],
                "Kepler's phases are visible in the plot.": ["Kepler's phases"],
                "Perhaps Edwin Hubble was the source of the name for the Hubble Space Telescope.": [
                    "Edwin Hubble",
                    "Hubble Space Telescope",
                ],
            }
            dict_acts_yesverbs = {"The Hubble calibrated data was used for the analysis.": ["Hubble calibrated data"]}
            #

            # Prepare and run test for bibcat class instance
            testbase = base._Base()
            # For tests where verbs are not included
            for phrase in dict_acts_noverbs:
                test_res = testbase._assemble_keyword_wordchunks(
                    text=phrase, keyword_objs=list_lookup_kobj, do_include_verbs=False, do_verbose=False
                )
                test_res = [item.text for item in test_res]
                #

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts_noverbs[phrase])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(
                            test_res, dict_acts_noverbs[phrase], phrase
                        )
                    )
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, dict_acts_noverbs[phrase])
                #
            #
            # For tests where verbs are indeed included
            for phrase in dict_acts_yesverbs:
                test_res = testbase._assemble_keyword_wordchunks(
                    text=phrase, keyword_objs=list_lookup_kobj, do_include_verbs=True, do_verbose=False
                )
                test_res = [item.text for item in test_res]
                #

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts_yesverbs[phrase])
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(
                            test_res, dict_acts_yesverbs[phrase], phrase
                        )
                    )
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, dict_acts_yesverbs[phrase])
                #
            #

        #
    #
    # For tests of _check_importance:
    if True:
        # Test determination of importance of given text with various terms
        def test__check_importance__variety(self):
            # Prepare text and answers for test
            kobj = kobj_hubble
            dict_acts = {
                "I went to the store.": ["is_pron_1st", "is_any"],
                "We all went to their house.": ["is_pron_1st", "is_pron_3rd", "is_any"],
                "We used HST data in our work.": ["is_pron_1st", "is_keyword", "is_any"],
                "Their paper uses both Hubble and JWST observations.": ["is_pron_3rd", "is_keyword", "is_any"],
                "The observations are plotted in Figure 3.": ["is_term_fig", "is_any"],
                "Tab. 7b lists the H.S.T project codes.": ["is_term_fig", "is_keyword", "is_any"],
                "The dog left the doghouse.": [],
                "Fig. 2 plots the trend line we measured.": ["is_pron_1st", "is_term_fig", "is_any"],
                "Blue et al. (2023) published the raw spectra already.": ["is_etal", "is_any"],
                "Green (1999) analyzed the Hubble Telescope data.": ["is_etal", "is_keyword", "is_any"],
                "The references for the Hubble data are given in Section 4 of Yellow et al..": [
                    "is_etal",
                    "is_keyword",
                    "is_term_fig",
                    "is_any",
                ],
            }
            #

            # Prepare and run test for bibcat class instance
            testbase = base._Base()
            for phrase in dict_acts:
                test_res = testbase._check_importance(text=phrase, keyword_objs=[kobj])
                list_res = [key for key in test_res if (test_res[key])]
                #

                # Check answer
                try:
                    self.assertEqual(sorted(list_res), sorted(dict_acts[phrase]))
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[phrase], phrase))
                    print("---")
                    print("")
                    #
                    self.assertEqual(sorted(list_res), sorted(dict_acts[phrase]))
                #
            #

        #
    #
    # For tests of _cleanse_text:
    if True:
        # Test _cleanse_text for variety of text
        def test__cleanse_text__variety(self):
            # Prepare text and answers for test
            kobj = kobj_hubble
            dict_acts = {
                "  J.W .. . S. - T .  .S-c": "J.W. S. - T.S-c",
                " We  walked to the   store/shop - across   the street. \n   We bought:   carrots-celery, and  - tofu-juice /milk.  ": "We walked to the store/shop - across the street. \n We bought: carrots-celery, and - tofu-juice /milk.",
                " The pets ( cats ,  dogs,  sheep  ) all went to the field.  They ' ve all [ allegedly  ] pink (re : : salmon ) fur .   That isn' t typical [apparently (e.g. Cite  2023 ) ].": "The pets (cats, dogs, sheep) all went to the field. They've all [allegedly] pink (re: salmon) fur. That isn't typical [apparently (e.g. Cite 2023)].",
                " The poetry... It was beautiful (according to the audience.).": "The poetry. It was beautiful (according to the audience.).",
            }
            #

            # Prepare and run test for bibcat class instance
            testbase = base._Base()
            for phrase in dict_acts:
                test_res = testbase._cleanse_text(text=phrase, do_streamline_etal=True)
                #

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts[phrase])
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[phrase], phrase))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, dict_acts[phrase])
                #
            #

        #
    #
    # For tests of _extract_core_from_phrase:
    if True:
        # Test core extraction for variety of phrases
        def test_extract_core_from_phrase__variety(self):
            # Prepare text and answers for test
            dict_acts = {
                "Hubble observatory": {"keywords": ["hubble"]},
                "Kepler planet": {"keywords": ["kepler"]},
                "Hubble and K2 observations": {"keywords": ["hubble", "k2"]},
                "Kepler data and K2 error": {"keywords": ["kepler", "k2"]},
                "Simultaneous Kepler phase curves": {"keywords": ["kepler"]},
                "Beautiful Hubble images of stars": {"keywords": ["hubble"]},
                "Both Hubble and Roman papers": {"keywords": ["hubble"]},
                "Hubble UDF": {"keywords": ["hubble"]},
            }
            #
            # Fill in rest of dictionary entries
            for key1 in dict_acts:
                curr_text = key1.split()
                dict_acts[key1]["text"] = key1
                dict_acts[key1]["synsets"] = []
                dict_acts[key1]["roots"] = []
                for curr_word in curr_text:
                    curr_syns = wordnet.synsets(curr_word)
                    curr_kobjs = [item for item in list_lookup_kobj if (item.is_keyword(curr_word))]
                    curr_set = [item.name() for item in curr_syns if (".n." in item.name())]
                    # Store as name if keyword
                    if len(curr_kobjs) > 0:
                        dict_acts[key1]["synsets"] += [[curr_kobjs[0].get_name().lower()]]
                        dict_acts[key1]["roots"] += [[curr_kobjs[0].get_name().lower()]]
                    # Otherwise, store synsets
                    elif len(curr_set) > 0:
                        curr_roots = np.unique([item.split(".")[0] for item in curr_set]).tolist()
                        dict_acts[key1]["synsets"] += [curr_set]
                        dict_acts[key1]["roots"] += [curr_roots]
                    # Otherwise, store word itself
                    else:
                        dict_acts[key1]["synsets"] += [[curr_word.lower()]]
                        dict_acts[key1]["roots"] += [[curr_word.lower()]]
                    #
                #
                # Process synsets into single string representation
                dict_acts[key1]["str_meaning"] = " ".join([" ".join(item) for item in dict_acts[key1]["roots"]])
            #

            # Prepare and run test for bibcat class instance
            testbase = base._Base()
            for key1 in dict_acts:
                test_res = testbase._extract_core_from_phrase(
                    phrase_NLP=nlp(key1), do_skip_useless=False, do_verbose=False, keyword_objs=list_lookup_kobj
                )
                #

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts[key1])
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[key1], key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, dict_acts[key1])
                #
            #

        #
    #
    # For tests of _check_truematch:
    if True:
        # Test verification of ambig. phrases for variety of phrases
        def test_check_truematch__variety1(self):
            # Prepare text and answers for test
            dict_tests = {
                "small Hubble constant": {"lookup": "Kepler", "bool": False},
                "small Hubble's constant": {"lookup": "Kepler", "bool": False},
                "Edwin Hubble's papers": {"lookup": "Hubble", "bool": False},
                "Hubble 1970": {"lookup": "Hubble", "bool": False},
                "Hubble (2000)": {"lookup": "Hubble", "bool": False},
                "high S/N Hubble image": {"lookup": "Hubble", "bool": True},
                "HST observatory": {"lookup": "Hubble", "bool": True},
                "H.S.T. observatory": {"lookup": "Hubble", "bool": True},
                "Hubble calibrated images": {"lookup": "Hubble", "bool": True},
                "Hubble's calibrated data": {"lookup": "Hubble", "bool": True},
                "Hubble's pretty spectra": {"lookup": "Hubble", "bool": True},
                "Edwin Hubble's analysis": {"lookup": "Hubble", "bool": False},
                "A Hubble constant data": {"lookup": "Hubble", "bool": False},
                "Hubble et al. 2000": {"lookup": "Hubble", "bool": False},
                "Hubbleetal 2000": {"lookup": "Hubble", "bool": False},
                "Hubble and more data.": {"lookup": "Hubble", "bool": True},
                "Kepler fields.": {"lookup": "Kepler", "bool": True},
                "Kepler velocities.": {"lookup": "Kepler", "bool": False},
                "Kepler velocity fields.": {"lookup": "Kepler", "bool": False},
                "Kepler rotation velocity fields.": {"lookup": "Kepler", "bool": False},
                "that Kepler data velocity.": {"lookup": "Kepler", "bool": True},
                "true Kepler planets": {"lookup": "Kepler", "bool": False},
                "those Kepler radii": {"lookup": "Kepler", "bool": False},
                "Keplerian orbits": {"lookup": "Kepler", "bool": False},
                "Kepler's law": {"lookup": "Kepler", "bool": False},
                "Kepler observations": {"lookup": "Kepler", "bool": True},
                "K2 database": {"lookup": "K2", "bool": True},
                "K2-123 star": {"lookup": "K2", "bool": False},
                "K2 stars": {"lookup": "K2", "bool": False},
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            dict_ambigs = testbase._process_database_ambig(do_verbose=False, keyword_objs=list_lookup_kobj)
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["bool"]
                    test_res = testbase._check_truematch(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    self.assertEqual(test_res["bool"], answer)
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        "{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(
                            test_res["bool"], answer, key1, test_res
                        )
                    )
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res["bool"], answer)
            #

        #
    #
    # For tests of _is_pos_word:
    if True:
        # Test identification of adjectives in a sentence
        def test_is_pos_word__adjective(self):
            test_pos = "ADJECTIVE"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "Hubble's calibrated data is in the database.": ["calibrated"],
                "She went to the clean store.": ["clean"],
                "Hubble has observed many stars.": ["many"],
                "That is one hideous dog.": ["hideous"],
                "The flying pig forms their sigil.": ["flying"],
                "She, the bird, and the cat all bought the book.": [],
                "The observed spectra are quickly plotted in pretty figures.": ["observed", "pretty"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of aux in a sentence
        def test_is_pos_word__aux(self):
            test_pos = "AUX"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "I think that is a good idea.": [],
                "Also shown are new curves.": ["are"],
                "They were frolicking and will soon be at the party.": ["were", "will"],
                "Hubble has observed many stars.": ["has"],
                "There are only so many entries in the database.": [],
                "That is one hideous dog.": [],
                "We would have gone there for vacation.": ["would", "have"],
                "The doors of the cage were open.": [],
                "The doors of the cage were having their locks fixed.": ["were"],
                "The option was not available at the time.": [],
                "The option was not noted at the time.": ["was"],
                "She could not see the mountain.": ["could"],
                "She will try to be there on time.": ["will", "to"],
                "They shall find the clue.": ["shall"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of marker in a sentence
        def test_is_pos_word__marker(self):
            test_pos = "MARKER"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She bought the book which was written by that team.": ["which"],
                "I think that is a good idea.": ["that"],
                "Which way to the store?": [],
                "They do not know which way to turn.": ["which"],
                "I read the book, which by the way was great.": ["which"],
                "They noted there are many ways to do it.": ["there"],
                "That is a fantastic quote for this poster.": [],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=list_lookup_kobj, pos=test_pos, do_verbose=False
                            )
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of direct object in a sentence
        def test_is_pos_word__directobject(self):
            test_pos = "DIRECT_OBJECT"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She went to the clean store.": [],
                "Hubble has observed many stars.": ["stars"],
                "There are only so many entries in the database.": ["entries"],
                "Many bought cakes that day.": ["cakes"],
                "That is one hideous dog.": ["dog"],
                "She treats them like her own plants.": ["them"],
                "Hubble is short for the Hubble Space Telescope.": [],
                "We went there on vacation.": [],
                "We went quickly on vacation.": [],
                "The door of the cage is open.": [],
                "We wanted to know the name of the movie.": ["name"],
                "They then wanted to know the name and length of the movie.": ["name", "length"],
                "She took a turn on the road along the left lane.": ["turn"],
                "She tried to take a picture.": ["picture"],
                "She tried to send the picture to the agent.": ["picture"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of prepositions in a sentence
        def test_is_pos_word__preposition(self):
            test_pos = "PREPOSITION"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She went to the clean store.": ["to"],
                "Hubble has observed many stars.": [],
                "There are only so many entries in the database.": ["in"],
                "Many bought cakes that day.": [],
                "That is one hideous dog.": [],
                "Hubble is short for the Hubble Space Telescope.": ["for"],
                "We went there on vacation.": ["on"],
                "The door of the cage is open.": ["of"],
                "We wanted to know the name of the movie.": ["of"],
                "She took a turn on the road along the left lane.": ["on", "along"],
                "She gave a treat to the cat.": ["to"],
                "She tried to take a picture.": [],
                "She tried to send the picture to the agent.": ["to"],
                "The book was written by her.": ["by"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of preposition objects in a sentence
        def test_is_pos_word__prepositionobject(self):
            test_pos = "PREPOSITION_OBJECT"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She went to the clean store.": ["store"],
                "Hubble has observed many stars.": [],
                "There are only so many entries in the database.": ["database"],
                "That is one hideous dog.": [],
                "Hubble is short for the Hubble Space Telescope.": ["Telescope"],
                "She treats them like her own plants.": ["plants"],
                "We went there on vacation.": ["vacation"],
                "The door of the cage is open.": [],
                "We wanted to know the name of the movie.": ["movie"],
                "The name of the movie is long.": [],
                "She took a turn on the road along the left lane.": ["road", "lane"],
                "She tried to take a picture.": [],
                "She tried to send the picture to the agent.": ["agent"],
                "She gave a treat to the cat.": ["cat"],
                "She gave a treat to the cat and the dog.": ["cat", "dog"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of preposition subjects in a sentence
        def test_is_pos_word__prepositionsubject(self):
            test_pos = "PREPOSITION_SUBJECT"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She went to the clean store.": [],
                "Hubble has observed many stars.": [],
                "There are only so many entries in the database.": [],
                "The door of the cage is open.": ["cage"],
                "The key in the lock should not be turned.": ["lock"],
                "We wanted to know the name of the movie.": [],
                "The name of the movie is long.": ["movie"],
                "The water in the basin seems gray.": ["basin"],
                "The cats on the couch, chair, and table are sleeping.": ["couch", "chair", "table"],
                "She took a turn on the road along the left lane.": [],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of subjects in a sentence
        def test_is_pos_word__subject(self):
            test_pos = "SUBJECT"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "The Hubble calibrated data was used.": ["data"],
                "Hubble calibrated the data.": ["Hubble"],
                "Calibrated data was used.": ["data"],
                "Calibrated data.": ["data"],
                "Fixed units.": ["units"],
                "She went to the clean store.": ["She"],
                "Hubble has observed many stars.": ["Hubble"],
                "There are only so many entries in the database.": ["There"],
                "That is a good idea.": ["That"],
                "Many bought cakes that day.": ["Many"],
                "That is one hideous dog.": ["That"],
                "Hubble is short for the Hubble Space Telescope.": ["Hubble"],
                "We went there on vacation.": ["We"],
                "The flying pig forms their sigil.": ["pig"],
                "The name of the cat rhymes with mat.": ["name"],
                "She, the bird, and the cat all bought the book.": ["She", "bird", "cat"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=list_lookup_kobj, pos=test_pos, do_verbose=False
                            )
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of useless words in a sentence
        def test_is_pos_word__useless(self):
            test_pos = "USELESS"  # Current p.o.s. to be tested
            # Prepare text and answers for test
            dict_tests = {
                "She went to the clean store.": ["clean"],
                "Hubble has observed many stars.": ["many"],
                "There are only so many entries in the database.": ["only", "so", "many"],
                "There are only so many.": ["only", "so"],
                "Many bought cakes that day.": [],
                "That is one hideous dog.": ["hideous"],
                "Hubble is short for the Hubble Space Telescope.": [],
                "That was not cool.": [],
                "That was not a cool cat.": ["cool"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=list_lookup_kobj, pos=test_pos)
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test identification of verbs in a sentence
        def test_is_pos_word__verb(self):
            test_pos = "VERB"
            # Prepare text and answers for test
            dict_tests = {
                "I think that is a good idea.": ["think", "is"],
                "Also shown are new curves.": ["shown"],
                "They were frolicking and will soon be at the party.": ["frolicking", "be"],
                "The Hubble calibrated data was used.": ["used"],
                "Hubble calibrated the data.": ["calibrated"],
                "Calibrated data was used.": ["used"],
                "Calibrated data.": [],
                "Fixed units.": [],
                "She went to the clean store.": ["went"],
                "Hubble has observed many stars.": ["observed"],
                "There are only so many entries in the database.": ["are"],
                "That is one hideous dog.": ["is"],
                "Many bought cakes that day.": ["bought"],
                "They are watching the movie.": ["watching"],
                "Hubble is short for the Hubble Space Telescope.": ["is"],
                "That was not cool.": ["was"],
                "The flying pig forms their sigil.": ["forms"],
                "Those raining clouds will float by.": ["float"],
                "Those clouds will float by while raining on the emptied town.": ["float", "raining"],
                "Follow me to the rendesvous.": ["Follow"],
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=list_lookup_kobj, pos=test_pos, do_verbose=False
                            )
                            for item in curr_NLP
                        ]
                    )
                    test_res = [item.text for item in np.asarray(curr_NLP)[test_bools]]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_bools))
                    print("All p.o.s.:")
                    for word1 in curr_NLP:
                        print("{0}: {1}, {2}, {3}".format(word1, word1.dep_, word1.pos_, word1.tag_))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
    #
    # For tests of _search_text:
    if True:
        # Test boolean search for keywords and acronyms within text
        def test_search_text__variety(self):
            # Prepare text and answers for test
            dict_tests = {
                "small Hubble constant": True,
                "small Hero constant": False,
                "Keplerian": False,
                " That H S.T data": True,
                " That H.S.T. data": True,
                " That H.S.T data": True,
                " That A.H.S.T data": False,
                " That HST data": True,
                " That MHST data": False,
                " That S T data": False,
                " That H S time": False,
                "no lookup Kepler images": True,
                "K2 images": True,
                "a nothst acronym": False,
                "a hstnot acronym": False,
                "a hsnt acronym": False,
                "an hst-roman project": True,
            }
            #

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._search_text(text=key1, keyword_objs=list_lookup_kobj)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
    #
    # For tests of _streamline_phrase:
    if True:
        # Test streamlining for text with abbreviated phrases
        def test_streamline_phrase__abbreviations(self):
            # Prepare text and answers for test
            dict_tests = {
                "We plot the data in Fig. 3b and list values in Tab. A.": "We plot the data in Figure 3b and list values in Table A.",
                "Fig. 4 shows the red vs. blue stars differ in radii.": "Figure 4 shows the red vs blue stars differ in radii.",
                "There is a fig on the tree. The tree is the Fig Tree.": "There is a fig on the tree. The tree is the Fig Tree.",
                "Put the ginger beer on my tab.": "Put the ginger beer on my tab.",
            }

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._streamline_phrase(text=key1)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test streamlining for text with citations
        def test_streamline_phrase__citations(self):
            # Prepare text and answers for test
            dict_tests = {
                "Somename (2013) published  in SJ.": "{0} published in SJ.".format(placeholder_author),
                "Hubble (1953) was a landmark paper (for that subfield).": "{0} was a landmark paper (for that subfield).".format(
                    placeholder_author
                ),
                "See also: Kepler [2023], Hubble & Author (2020), Author, Somename, and Kepler et al. [1990];": "See also: {0}, {0}, {0};".format(
                    placeholder_author
                ),
                "Also Author papers (Author et al. 1997, 2023),": "Also Author papers,",
                "(Someone, Author, Somename et al. 1511; 1612)": "",
                "(Someone, Author, and Somename et al. 1913,15)": "",
                "(Author et al. 80; Somename & Author 2012)": "",
                "McThatname, Kepler, & Othername [1993] (see our paper)": "{0} (see our paper)".format(
                    placeholder_author
                ),
                "{Othername et al. 1991} (see Hubble observations)": "(see Hubble observations)",
            }

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._streamline_phrase(text=key1)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test streamlining for text with numerics
        def test_streamline_phrase__numerics(self):
            # Prepare text and answers for test
            dict_tests = {
                "There were 200-300 observations done of star AB100+300.": "There were {1} observations done of star AB{0}.".format(
                    placeholder_number, placeholder_numeric
                ),
                "Consider planet J9385-193 and 2MASS293-04-331+101.": "Consider planet J{0} and 2MASS{0}.".format(
                    placeholder_number
                ),
                "Disk HD 193283-10, Kepler-234c, and Planet 312b as well.": "Disk HD{0}, Kepler {0}, and Planet {0} as well.".format(
                    placeholder_number
                ),
                "The latter had ~450 - 650 data points in total.": "The latter had {1} data points in total.".format(
                    placeholder_number, placeholder_numeric
                ),
            }

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._streamline_phrase(text=key1)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
        # Test streamlining for text with websites
        def test_streamline_phrase__websites(self):
            # Prepare text and answers for test
            dict_tests = {
                "Please check out: www.stsci.edu/home for more info.": "Please check out: {0} for more info.".format(
                    placeholder_website
                ),
                "Consider also https://jwst.edu/,": "Consider also {0},".format(placeholder_website),
                "http:hst.edu/lookup=wow?; public.stsci.edu,": "{0}; {0},".format(placeholder_website),
                "   www.roman-telescope.stsci.edu/main/about/. ": "{0}.".format(placeholder_website),
            }

            # Prepare and run tests for bibcat class instance
            testbase = base._Base()
            #
            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._streamline_phrase(text=key1)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
            #

        #
    #


# """
# """
# class: TestKeyword
# Purpose: Testing the Keyword class
class TestKeyword(unittest.TestCase):
    # For tests of get_name:
    if True:
        # Test determination of representative name for Keyword object
        def test_get_name__variety(self):
            # Prepare text and answers for test
            kobj1 = keyword.Keyword(keywords=["Long Phrase", "Phrase", "Longer Phrase", "Mid Phrase"], acronyms=[])
            ans1 = "Phrase"
            #
            kobj2 = keyword.Keyword(
                keywords=["Long Phrase", "Phrase", "Longer Phrase", "Mid Phrase"],
                acronyms=["AB....C", "A....... B...", "A.BC  D", "ABCD E", "AB C"],
            )
            ans2 = "Phrase"
            #
            kobj3 = keyword.Keyword(keywords=[], acronyms=["AB....C", "A....... B...", "A.BC  D", "ABCD E", "AB C"])
            ans3 = "ABCDE"
            #
            list_kobj = [kobj1, kobj2, kobj3]
            list_ans = [ans1, ans2, ans3]
            num_tests = len(list_kobj)
            #

            # Prepare and run test for bibcat class instance
            for ii in range(0, num_tests):
                curr_kobj = list_kobj[ii]
                curr_ans = list_ans[ii]
                test_res = curr_kobj.get_name()
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_ans)
                except AssertionError:
                    print("")
                    print(">")
                    print("Instance: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_ans, curr_kobj))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_ans)
                #
            #

        #
    #
    # For tests of is_keyword:
    if True:
        # Test determination of if given text contains keyword terms
        def test_is_keyword__variety(self):
            # Prepare text and answers for test
            dict_acts = {
                "Keplerian velocity": {"kobj": kobj_kepler, "bool": False},
                "that Hubble data": {"kobj": kobj_hubble, "bool": True},
                "that Hubble Space Telescope data": {"kobj": kobj_hubble, "bool": True},
                "that H S T data": {"kobj": kobj_hubble, "bool": True},
                "that H.S T data": {"kobj": kobj_hubble, "bool": True},
                "that HST data": {"kobj": kobj_hubble, "bool": True},
                "that HST PSF.": {"kobj": kobj_hubble, "bool": True},
                "hst": {"kobj": kobj_hubble, "bool": True},
                "HST": {"kobj": kobj_hubble, "bool": True},
                "HST.": {"kobj": kobj_hubble, "bool": True},
                "HST PSF.": {"kobj": kobj_hubble, "bool": True},
                "that HcST data": {"kobj": kobj_hubble, "bool": False},
                "that A.H.S.T.M. data": {"kobj": kobj_hubble, "bool": False},
                "that A.H.S.T.M data": {"kobj": kobj_hubble, "bool": False},
                "that AHSTM data": {"kobj": kobj_hubble, "bool": False},
                "that LHST data": {"kobj": kobj_hubble, "bool": False},
                "that HS.xT data": {"kobj": kobj_hubble, "bool": False},
                "that K23 data": {"kobj": kobj_k2, "bool": False},
                "that AK2 data": {"kobj": kobj_k2, "bool": False},
                "that K2 data": {"kobj": kobj_k2, "bool": True},
                "that K 2 star": {"kobj": kobj_k2, "bool": False},
                "archive observations": {"kobj": kobj_hla, "bool": False},
                "hubble and archive observations": {"kobj": kobj_hla, "bool": False},
                "Hubble and Archive observations": {"kobj": kobj_hla, "bool": False},
                "they took Hubble images": {"kobj": kobj_hla, "bool": False},
                "they took HST images": {"kobj": kobj_hla, "bool": False},
                "the hubble legacy archive": {"kobj": kobj_hla, "bool": True},
                "the hla uncapitalized": {"kobj": kobj_hla, "bool": True},
                "the Hubble Archive is different": {"kobj": kobj_hla, "bool": False},
            }
            #

            # Prepare and run test for bibcat class instance
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]
                curr_bool = dict_acts[phrase]["bool"]
                test_res = curr_kobj.is_keyword(text=phrase)
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_bool)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_bool, phrase))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_bool)
                #
            #

        #
    #
    # For tests of replace_keyword:
    if True:
        # Test removal of keyword terms from text
        def test_replace_keyword__variety(self):
            # Prepare text and answers for test
            placeholder = placeholder_anon
            dict_acts = {
                "Keplerian velocity": {"kobj": kobj_kepler, "result": "Keplerian velocity"},
                "that Hubble data": {"kobj": kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that Hubble Space Telescope data": {
                    "kobj": kobj_hubble,
                    "result": "that {0} data".format(placeholder),
                },
                "that Hubble Telescope data": {"kobj": kobj_hubble, "result": "that {0} data".format(placeholder)},
                "hst": {"kobj": kobj_hubble, "result": "{0}".format(placeholder)},
                "HST": {"kobj": kobj_hubble, "result": "{0}".format(placeholder)},
                "that H S T data": {"kobj": kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that H.S T data": {"kobj": kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that HST data": {"kobj": kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that HST PSF.": {"kobj": kobj_hubble, "result": "that {0} PSF.".format(placeholder)},
                "HST PSF.": {"kobj": kobj_hubble, "result": "{0} PSF.".format(placeholder)},
                "that HcST data": {"kobj": kobj_hubble, "result": "that HcST data"},
                "that A.H.S.T.M. data": {"kobj": kobj_hubble, "result": "that A.H.S.T.M. data"},
                "that LHST data": {"kobj": kobj_hubble, "result": "that LHST data"},
                "that HS.xT data": {"kobj": kobj_hubble, "result": "that HS.xT data"},
                "that K23 data": {"kobj": kobj_k2, "result": "that K23 data"},
                "that AK2 data": {"kobj": kobj_k2, "result": "that AK2 data"},
                "that K2 data": {"kobj": kobj_k2, "result": "that {0} data".format(placeholder)},
                "that K 2 star": {"kobj": kobj_k2, "result": "that K 2 star"},
            }
            #

            # Prepare and run test for bibcat class instance
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]
                act_res = dict_acts[phrase]["result"]
                test_res = curr_kobj.replace_keyword(text=phrase, placeholder=placeholder)
                #

                # Check answer
                try:
                    self.assertEqual(test_res, act_res)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, act_res, phrase))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, act_res)
                #
            #

        #
    #


# """
# """
# class: TestPaper
# Purpose: Testing the Paper class
class TestPaper(unittest.TestCase):
    # For tests of process_paragraphs and get_paragraphs:
    if True:
        # Test processing and extraction of paragraphs of target terms
        def test_processandget_paragraphs__variety(self):
            placeholder = config.string_anymatch_ambig
            # Prepare text and answers for test
            text1 = "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
            meanings1 = {"Hubble": ["Hubble Space Telescope"], "Kepler": None, "K2": None}
            ambigs1 = [("our Hubble results", True)]
            #
            text2 = "Kepler observations are presented in Section 1. Table 1 then gives the measured Kepler velocity data and Kepler planets. The Kepler images and Kepler plots indicate a correlation. The Kepler results are further shown in part 2. Note the Keplerian rotation."
            meanings2 = {"Hubble": None, "Kepler": None, "K2": None}
            ambigs2 = [
                ("Kepler observations", True),
                ("measured Kepler velocity data", False),
                ("Kepler planets", False),
                ("Kepler images", True),
                ("Kepler plots", True),
                ("Kepler results", True),
            ]

            list_acts = [
                {
                    "text": text1,
                    "kobj": kobj_hubble,
                    "buffer": 0,
                    "answer": [
                        "Figure 1 plots the Hubble Space Telescope (HST) observations.",
                        "The HST stars are especially bright.",
                        "We summarize our Hubble results next.",
                    ],
                    "acronym_meanings": meanings1,
                    "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": kobj_hubble,
                    "buffer": 1,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section.",
                        "Some more filler content. We summarize our Hubble results next.",
                    ],
                    "acronym_meanings": meanings1,
                    "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": kobj_hubble,
                    "buffer": 3,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
                    ],
                    "acronym_meanings": meanings1,
                    "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": kobj_hubble,
                    "buffer": 10,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
                    ],
                    "acronym_meanings": meanings1,
                    "results_ambig": ambigs1,
                },
                {
                    "text": text2,
                    "kobj": kobj_kepler,
                    "buffer": 0,
                    "answer": [
                        "Kepler observations are presented in Section 1.",
                        "The Kepler images and Kepler plots indicate a correlation.",
                        "The Kepler results are further shown in part 2.",
                    ],
                    "acronym_meanings": meanings2,
                    "results_ambig": ambigs2,
                },
                {
                    "text": text2,
                    "kobj": kobj_kepler,
                    "buffer": 1,
                    "answer": [
                        "Kepler observations are presented in Section 1. Table 1 then gives the measured Kepler velocity data and Kepler planets. The Kepler images and Kepler plots indicate a correlation. The Kepler results are further shown in part 2. Note the Keplerian rotation."
                    ],
                    "acronym_meanings": meanings2,
                    "results_ambig": ambigs2,
                },
            ]

            # Determine and check answers
            for info in list_acts:
                curr_text = info["text"]
                curr_kobj = info["kobj"]
                curr_buffer = info["buffer"]
                curr_answer = info["answer"]
                curr_ambig = info["results_ambig"]
                curr_acr_meanings = info["acronym_meanings"]
                curr_name = curr_kobj.get_name()

                # Prepare and run test for bibcat class instance
                testbase = paper.Paper(text=curr_text, keyword_objs=list_lookup_kobj, do_check_truematch=True)
                set_res = testbase.process_paragraphs(buffer=curr_buffer)
                test_res = testbase.get_paragraphs()[curr_name]
                #
                ambig_output = testbase._get_info("_results_ambig")[curr_name]
                test_ambig = [
                    (item2["text_wordchunk"], item2["bool"])
                    for item1 in ambig_output
                    for item2 in item1["info"]
                    if (item2["matcher"] is not None)
                ]
                test_acr_meanings = testbase._get_info("_dict_acronym_meanings")
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                    self.assertEqual(test_ambig, curr_ambig)
                    self.assertEqual(test_acr_meanings, curr_acr_meanings)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_answer, curr_text))
                    print("Test ambig: {0}\nAct. ambig: {1}".format(test_ambig, curr_ambig))
                    print(
                        "Test acronym matches: {0}\nAct. acronym matches: {1}".format(
                            test_acr_meanings, curr_acr_meanings
                        )
                    )
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_answer)
                    self.assertEqual(test_ambig, curr_ambig)
                    self.assertEqual(test_acr_meanings, curr_acr_meanings)
                #
            #

        #
    #
    # For tests of _buffer_indices:
    if True:
        # Test split of text into sentences based on naive sentence boundaries
        def test__buffer_indices__variety(self):
            # Prepare text and answers for test
            list_acts = [
                {
                    "inds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "buffer": 0,
                    "max": 11,
                    "result": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
                },
                {"inds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "buffer": 1, "max": 11, "result": [[0, 11]]},
                {"inds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "buffer": 2, "max": 11, "result": [[0, 11]]},
                {
                    "inds": [1, 2, 3, 6, 7, 8, 10, 11],
                    "buffer": 0,
                    "max": 11,
                    "result": [[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8], [10, 10], [11, 11]],
                },
                {"inds": [1, 2, 3, 7, 8, 11], "buffer": 1, "max": 10, "result": [[0, 4], [6, 9], [10, 10]]},
                {"inds": [1, 2, 3, 7, 8, 11], "buffer": 1, "max": 11, "result": [[0, 4], [6, 9], [10, 11]]},
                {"inds": [1, 2, 3, 7, 8, 11], "buffer": 1, "max": 12, "result": [[0, 4], [6, 9], [10, 12]]},
                {"inds": [1, 2, 3, 7, 8, 11], "buffer": 1, "max": 13, "result": [[0, 4], [6, 9], [10, 12]]},
                {"inds": [1, 2, 7, 12], "buffer": 2, "max": 13, "result": [[0, 4], [5, 9], [10, 13]]},
                {"inds": [1, 2, 7, 12], "buffer": 2, "max": 11, "result": [[0, 4], [5, 9], [10, 11]]},
            ]
            #

            # Determine and check answers
            for info in list_acts:
                curr_inds = info["inds"]
                curr_buffer = info["buffer"]
                curr_max = info["max"]
                curr_answer = info["result"]

                # Prepare and run test for bibcat class instance
                testbase = paper.Paper(text="", keyword_objs=[kobj_hubble], do_check_truematch=False)
                test_res = testbase._buffer_indices(indices=curr_inds, buffer=curr_buffer, max_index=curr_max)
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("Setup: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_answer, info))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_answer)
                #
            #

        #
    #
    # For tests of _split_text:
    if True:
        # Test split of text into sentences based on naive sentence boundaries
        def test__split_text__variety(self):
            # Prepare text and answers for test
            text = (
                "We need to split this text into sentences. H.S.T. is an"
                + " acronym for the Hubble Space Telescope. We show the"
                + " recently observed data in Fig. 3 and Tab. 4. Fig. 7 then"
                + " shows the rest of the results (see also the Appendix.)"
                + " More work is described in the next section.\n"
                + "We see trends in the data. (Those trends are highlighted"
                + " in another paper). This sentence contains an ellipses..."
                + " And should still be split into its own sentence."
                + " Acronyms like J.W.S.T and H.S.T. could mess up"
                + " sentence extraction (as could parentheses.). End."
            )
            #
            answer = [
                "We need to split this text into sentences.",
                "H.S.T. is an acronym for the Hubble Space Telescope.",
                "We show the recently observed data in Fig. 3 and Tab. 4.",
                "Fig. 7 then shows the rest of the results (see also the Appendix.)",
                "More work is described in the next section.",
                "We see trends in the data.",
                "(Those trends are highlighted in another paper).",
                "This sentence contains an ellipses...",
                "And should still be split into its own sentence.",
                "Acronyms like J.W.S.T and H.S.T. could mess up sentence extraction (as could parentheses.).",
                "End.",
            ]

            # Prepare and run test for bibcat class instance
            testbase = paper.Paper(text="", keyword_objs=[kobj_hubble], do_check_truematch=False)
            test_res = testbase._split_text(text=text)
            # Check answer
            try:
                self.assertEqual(test_res, answer)
            except AssertionError:
                print("")
                print(">")
                for ii in range(0, len(answer)):
                    print("Test answer: {0}\nAct. answer: {1}".format(test_res[ii], answer[ii]))
                print("---")
                print("")
                #
                self.assertEqual(test_res, answer)
            #

        #
    #
    # For tests of _verify_acronyms:
    if True:
        # Test search for possible meanings of given acronyms
        def test__verify_acronyms__variety(self):
            # Prepare text and answers for test
            dict_acts = {
                "The Hubble Space Telescope is a telescope often referred to as H.S.T. or HST.": {
                    "kobj": kobj_hubble,
                    "matches": ["Hubble Space Telescope"],
                },
                "The Heralding of the Swan Trumphets will be showing in the Healing Song Theatre Plaza next week.": {
                    "kobj": kobj_hubble,
                    "matches": ["Heralding of the Swan Trumphets", "Healing Song Theatre"],
                },
                "Hello. Space Tyrants is showing in the theatre next door.": {"kobj": kobj_hubble, "matches": []},
                "H. Space Tyrants is showing in the theatre next door.": {"kobj": kobj_hubble, "matches": []},
                "H S T is showing in the theatre next door.": {"kobj": kobj_hubble, "matches": []},
                "Hijinks of Space Tyrants is showing in the theatre next door.": {
                    "kobj": kobj_hubble,
                    "matches": ["Hijinks of Space Tyrants"],
                },
                "Hidden space tyrants is showing in the theatre next door. H. S. TheName will be seeing it.": {
                    "kobj": kobj_hubble,
                    "matches": [],
                },
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]
                curr_answer = dict_acts[phrase]["matches"]

                # Prepare and run test for bibcat class instance
                testbase = paper.Paper(text=phrase, keyword_objs=[kobj_hubble], do_check_truematch=False)
                test_res = testbase._verify_acronyms(keyword_obj=curr_kobj)
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_answer, phrase))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_answer)
                #
            #

        #
    #


# """
# """
# class: TestGrammar
# Purpose: Testing the Grammar class
class TestGrammar(unittest.TestCase):
    # For tests of generating and fetching modifs:
    if True:
        # Test modif generation for basic example text
        def test__modifs__basic(self):
            # Prepare text and answers for test
            dict_acts = {
                "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST. It is a telescope that is used across the world. The HST legacy archive is a rich treasure chest of ultraviolet observations used across the world.": {
                    "kobj": kobj_hubble,
                    "none": "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST.\nThe HST legacy archive is a rich treasure chest of ultraviolet observations used across the world.",
                    "skim": "The Hubble Space Telescope is a telescope that is referred to as Hubble or HST.\nThe HST legacy archive is a treasure chest of observations used across the world.",
                    "trim": "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST.\nThe HST legacy archive is a rich treasure chest of ultraviolet observations.",
                    "anon": "The {0} is a really neat telescope that is often referred to as {0} or {0}.\nThe {0} legacy archive is a rich treasure chest of ultraviolet observations used across the world.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "The {0} is a telescope that is referred to as {0} or {0}.\nThe {0} legacy archive is a treasure chest of observations.".format(
                        placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
        # Test modif generation for example text with many clauses
        def test__modifs__manyclauses(self):
            # Prepare text and answers for test
            dict_acts = {
                "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.": {
                    "kobj": kobj_hubble,
                    "none": "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.",
                    "skim": "Once the transit is observable, we will observe the target in a fashion without delay and at resolution using either JWST or Hubble.",
                    "trim": "we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.",
                    "anon": "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or {0}.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "we will observe the target in a fashion without delay and at resolution using either JWST or {0}.".format(
                        placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(
                    text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
        # Test modif generation for example text with clause with mission information
        def test__modifs__importantclause_mission(self):
            # Prepare text and answers for test
            dict_acts = {
                "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.": {
                    "kobj": kobj_hubble,
                    "none": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.",
                    "skim": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the HST observations, we are unable to detect those targets in the snapshots taken with the Cool Telescope.",
                    "trim": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots.",
                    "anon": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the {0} observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the {0} observations, we are unable to detect those targets in the snapshots.".format(
                        placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
        # Test modif generation for example text with important clause with 1st-person
        def test__modifs__importantclause_1stperson(self):
            # Prepare text and answers for test
            dict_acts = {
                "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.": {
                    "kobj": kobj_hubble,
                    "none": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "skim": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the Cool Telescope observations, we are unable to detect those targets in the snapshots taken with the HST.",
                    "trim": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "anon": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the {0}.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "While we are able to detect (above 5sigma) the three stars within the mosaic, we are unable to detect those targets in the snapshots taken with the {0}.".format(
                        placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(
                    text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
        # Test modif generation for example text with unimportant clause
        def test__modifs__unimportantclause(self):
            # Prepare text and answers for test
            dict_acts = {
                "While that study was able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, this study was unable to even tentatively detect those same targets in the snapshots taken with the HST.": {
                    "kobj": kobj_hubble,
                    "none": "While that study was able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, this study was unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "skim": "While that study was able to detect (above 5sigma) the three stars within the mosaic created by the Cool Telescope observations, this study was unable to detect those targets in the snapshots taken with the HST.",
                    "trim": "this study was unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "anon": "While that study was able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, this study was unable to even tentatively detect those same targets in the snapshots taken with the {0}.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "this study was unable to detect those targets in the snapshots taken with the {0}.".format(
                        placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
        # Test modif generation for example text with unimportant conjunction
        def test__modifs__unimportantconjunction(self):
            # Prepare text and answers for test
            dict_acts = {
                "The red stars are observed with HST and further verified with JWST.": {
                    "kobj": kobj_hubble,
                    "none": "The red stars are observed with HST and further verified with JWST.",
                    "skim": "The stars are observed with HST and verified with JWST.",
                    "trim": "The red stars are observed with HST.",
                    "anon": "The red stars are observed with {0} and further verified with JWST.".format(
                        placeholder_anon
                    ),
                    "skim_trim_anon": "The stars are observed with {0}.".format(placeholder_anon),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = grammar.Grammar(
                    text=phrase, keyword_obj=kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]
                    #

                    # Check answer
                    try:
                        self.assertEqual(test_res, curr_answer)
                    except AssertionError:
                        print("")
                        print(">")
                        print(
                            ("Text: {2}\nMode: {3}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(
                                test_res, curr_answer, phrase, key1
                            )
                        )
                        print("---")
                        print("")
                        #
                        self.assertEqual(test_res, curr_answer)
                    #
                #
            #

        #
    #


# """
# """
# class: TestOperator
# Purpose: Testing the Operator class
class TestOperator(unittest.TestCase):
    # For tests of fetching keyword object using lookup item:
    if True:
        # Test fetching keyword object for objects with overlapping terms
        def test__fetch_keyword_object__overlap(self):
            # Prepare text and answers for test
            tmp_kobj_list = [kobj_hubble, kobj_hla]
            dict_acts = {
                "Hubble Legacy Archive": "HLA",
                "Hubble Legacy Archive results": "HLA",
                "Hubble": "Hubble",
                "HST": "Hubble",
                "HLA": "HLA",
                "Hubble Archive": "Hubble",
                "Hubble Legacy": "Hubble",
            }
            #

            # Prepare and run test for bibcat class instance
            testbase = operator.Operator(
                classifier=None,
                mode=None,
                keyword_objs=tmp_kobj_list,
                do_verbose=False,
                name="operator",
                load_check_truematch=False,
                do_verbose_deep=False,
            )
            # Determine and check answers
            for key1 in dict_acts:
                # Otherwise, check generated modif
                curr_lookup = key1
                test_res = testbase._fetch_keyword_object(
                    lookup=curr_lookup, do_verbose=False, do_raise_emptyerror=True
                )
                curr_answer = dict_lookup_kobj[dict_acts[key1]]
                #

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                except AssertionError:
                    print("")
                    print(">")
                    print(
                        ("Text: {2}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(test_res, curr_answer, key1)
                    )
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, curr_answer)
                #
            #

        #
    #


# """
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
###Run the tests
#
if __name__ == "__main__":
    unittest.main()
#
