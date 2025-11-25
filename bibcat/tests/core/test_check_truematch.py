"""
:title: test_check_truematch.py

Testing the _check_truematch methods of the Paper class.
"""

import unittest

import numpy as np
import spacy
from nltk.corpus import wordnet  # type: ignore

from bibcat import config
from bibcat.core import paper
from bibcat.core import parameters as params
from bibcat.core.parameters import kobj_copernicus, kobj_hubble, kobj_k2, kobj_kepler

nlp = spacy.load(config.grammar.spacy_language_model)

# test Keyword-object lookups
test_dict_lookup_kobj = {
    "Hubble": kobj_hubble,
    "Kepler": kobj_kepler,
    "K2": kobj_k2,
    "Copernicus": kobj_copernicus,
}

# test Keyword-object lookups
test_list_lookup_kobj = [kobj_hubble, kobj_kepler, kobj_k2, kobj_copernicus]


class TestCheckTruematch(unittest.TestCase):
    # For tests of _check_truematch:
    if True:
        # Test verification of ambig. phrases for variety of phrases
        def test_check_truematch__variety1(self):
            # Prepare text and answers for test
            dict_tests = {
                "small Hubble constant": {"lookup": "Kepler", "bool": False},
                "small Hubble's constant": {"lookup": "Kepler", "bool": False},
                "Edwin Hubble's papers": {"lookup": "Hubble", "bool": False},
                # "Hubble 1970": {"lookup": "Hubble", "bool": False}, - not realistic since would be cleaned beforehand normally
                # "Hubble (2000)": {"lookup": "Hubble", "bool": False}, - not realistic since would be cleaned beforehand normally
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
                "Copernicus satellite": {"lookup": "Copernicus", "bool": True},
                "Copernicus model": {"lookup": "Copernicus", "bool": False},
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["bool"]
                    test_res = testpaper._check_truematch(
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

                    self.assertEqual(test_res["bool"], answer)

    # For tests of _early_false_no_keyword_match:
    if True:
        # Test verification of ambig. phrases for variety of phrases
        def test_early_false_no_keyword_match(self):
            # Prepare text and answers for test
            dict_tests = {
                "small Hubble constant": {"lookup": "Kepler", "result": False},
                "small Hubble's constant": {"lookup": "Kepler", "result": False},
                "Edwin Hubble's papers": {"lookup": "Hubble", "result": None},
                "high S/N Hubble image": {"lookup": "Hubble", "result": None},
                "HST observatory": {"lookup": "Hubble", "result": None},
                "H.S.T. observatory": {"lookup": "Hubble", "result": None},
                "Hubble et al. 2000": {"lookup": "Hubble", "result": None},
                "Hubbleetal 2000": {"lookup": "Hubble", "result": False},
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    test_res = testpaper._early_false_no_keyword_match(setup_data)
                    if test_res is not None:
                        test_res = test_res["bool"]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _early_true_acronym_match:
    if True:
        # Test verification of ambig. phrases for variety of phrases
        def test_early_true_acronym_match(self):
            # Prepare text and answers for test
            dict_tests = {
                "small Hubble constant": {"lookup": "Kepler", "result": None},
                "small Hubble's constant": {"lookup": "Kepler", "result": None},
                "Edwin Hubble's papers": {"lookup": "Hubble", "result": None},
                "high S/N Hubble image": {"lookup": "Hubble", "result": None},
                "HST observatory": {"lookup": "Hubble", "result": True},
                "H.S.T. observatory": {"lookup": "Hubble", "result": True},
                "Figure 1 plots the Hubble Space Telescope (HST) observations.": {"lookup": "Hubble", "result": True},
                "We summarize our Hubble results next.": {"lookup": "Hubble", "result": None},
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    test_res = testpaper._early_true_acronym_match(setup_data)
                    if test_res is not None:
                        test_res = test_res["bool"]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _early_true_non_ambig_phrases:
    if True:
        # Return status as a true match if non-ambiguous phrases match to text.
        def test_early_true_non_ambig_phrases(self):
            # Prepare text and answers for test
            long_text_1 = (
                "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST."
            )
            long_text_2 = (
                "Once the transit is observable, we will observe the target in a timely fashion without delay "
                + "and at high resolution using either JWST or Hubble."
            )
            dict_tests = {
                long_text_1: {"lookup": "Hubble", "result": True},
                long_text_2: {"lookup": "Hubble", "result": None},
                "Edwin Hubble's papers": {"lookup": "Hubble", "result": None},
                "high S/N Hubble image": {"lookup": "Hubble", "result": None},
                "HST observatory": {"lookup": "Hubble", "result": None},
                "H.S.T. observatory": {"lookup": "Hubble", "result": None},
                "Figure 1 plots the Hubble Space Telescope (HST) observations.": {"lookup": "Hubble", "result": True},
                "We summarize our Hubble results next.": {"lookup": "Hubble", "result": None},
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    test_res = testpaper._early_true_non_ambig_phrases(setup_data)
                    if test_res is not None:
                        test_res = test_res["bool"]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _assemble_keyword_wordchunks_wrapper:
    if True:
        # Wraps the `_assemble_keyword_wordchunks` method (inherited from `base.py`) with verbose logging and error handling.
        # The base method assembles noun chunks around keyword terms found in the input text.
        def test_assemble_keyword_wordchunks_wrapper(self):
            # Prepare text and answers for test
            dict_tests = {
                "Hubble calibrated images": {"lookup": "Hubble", "result": ["Hubble calibrated images"]},
                "Hubble's calibrated data": {"lookup": "Hubble", "result": ["Hubble's calibrated data"]},
                "Hubble and more data.": {"lookup": "Hubble", "result": ["Hubble"]},
                "Kepler fields.": {"lookup": "Kepler", "result": ["Kepler fields"]},
                "We summarize our Hubble results next.": {"lookup": "Hubble", "result": ["our Hubble results"]},
                "The Kepler images and Kepler plots indicate a correlation.": {
                    "lookup": "Kepler",
                    "result": ["Kepler images", "Kepler plots"],
                },
                "Table 1 then gives the measured Kepler velocity data and Kepler planets.": {
                    "lookup": "Kepler",
                    "result": ["measured Kepler velocity data", "Kepler planets"],
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    test_res = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    test_res = [repr(item) for item in test_res]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _early_true_exact_wordchunk:
    if True:
        # Return status as a true match if any wordchunk is an exact keyword match.
        def test_early_true_exact_wordchunk(self):
            # Prepare text and answers for test
            dict_tests = {
                "Hubble calibrated images": {"lookup": "Hubble", "result": None},
                "Hubble's calibrated data": {"lookup": "Hubble", "result": None},
                "Hubble and more data.": {"lookup": "Hubble", "result": True},
                "Kepler fields.": {"lookup": "Kepler", "result": None},
                "We summarize our Hubble results next.": {"lookup": "Hubble", "result": None},
                "The Kepler images and Kepler plots indicate a correlation.": {"lookup": "Kepler", "result": None},
                "Table 1 then gives the measured Kepler velocity data and Kepler planets.": {
                    "lookup": "Kepler",
                    "result": None,
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    test_res = testpaper._early_true_exact_wordchunk(list_wordchunks, setup_data)
                    if test_res is not None:
                        test_res = test_res["bool"]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _consider_wordchunk:
    if True:
        # Evaluates a given wordchunk (noun phrase) to determine if it matches any known ambiguous phrases or meanings.
        # If a match is found, returns a list of formatted match results.
        # If no match is found, raises a `NotImplementedError` to signal an unrecognized ambiguous phrase.
        def test_consider_wordchunk(self):
            # Prepare text and answers for test
            dict_tests = {
                "Hubble's calibrated data": {
                    "lookup": "Hubble",
                    "result": [("Hubble's calibrated data", True, "AnyMission data")],
                },
                "Hubble's pretty spectra": {
                    "lookup": "Hubble",
                    "result": [("Hubble's pretty spectra", True, "AnyMission spectrum")],
                },
                "Hubble et al. 2000": {
                    "lookup": "Hubble",
                    "result": [("Hubble et al", False, "AnyMission et al")],
                },
                "Table 1 then gives the measured Kepler velocity data and Kepler planets.": {
                    "lookup": "Kepler",
                    "result": [
                        ("measured Kepler velocity data", False, "Kepler velocity"),
                        ("Kepler planets", False, "Kepler planet"),
                    ],
                },
                "The Kepler images and Kepler plots indicate a correlation.": {
                    "lookup": "Kepler",
                    "result": [
                        ("Kepler images", True, "AnyMission image"),
                        ("Kepler plots", True, "AnyMission plot"),
                    ],
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    list_results = [
                        testpaper._consider_wordchunk(curr_chunk, setup_data) for curr_chunk in list_wordchunks
                    ]
                    test_res = []
                    for wordchunk, wcResult in zip(list_wordchunks, list_results):
                        test_res.append(
                            (repr(wordchunk), wcResult["info"][0]["bool"], wcResult["info"][0]["text_database"])
                        )
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _setup_consider_wordchunk:
    if True:
        # Returns setup variables for use by _extract_ambig_phrases_substrings.
        def test_setup_consider_wordchunk(self):
            # Prepare text and answers for test
            dict_tests = {
                "Hubble's calibrated data": {
                    "lookup": "Hubble",
                    "result": [("Hubble's calibrated data", "hubble calibrated data datum", ["hubble"])],
                },
                "Edwin Hubble's papers": {
                    "lookup": "Hubble",
                    "result": [
                        ("Edwin Hubble's papers", "edwin hubble composition document newspaper paper", ["hubble"])
                    ],
                },
                "Hubble et al. 2000": {
                    "lookup": "Hubble",
                    "result": [("Hubble et al", "hubble et alabama aluminum", ["hubble"])],
                },
                "Kepler velocity fields.": {
                    "lookup": "Kepler",
                    "result": [
                        (
                            "Kepler velocity fields",
                            "kepler speed airfield battlefield discipline field fields plain playing_field sphere",
                            ["kepler"],
                        )
                    ],
                },
                "We summarize our Hubble results next.": {
                    "lookup": "Hubble",
                    "result": [
                        ("our Hubble results", "our hubble consequence result resultant_role solution", ["hubble"])
                    ],
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    list_results = [
                        testpaper._setup_consider_wordchunk(curr_chunk, setup_data) for curr_chunk in list_wordchunks
                    ]
                    test_res = []
                    for wordchunk, wcResult in zip(list_wordchunks, list_results):
                        curr_meaning = wcResult[0]
                        curr_inner_kw = wcResult[1]
                        test_res.append((repr(wordchunk), curr_meaning, curr_inner_kw))
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _extract_ambig_phrases_substrings:
    if True:
        # Extract all ambig. phrases+substrings that match to this meaning
        def test__extract_ambig_phrases_substrings(self):
            # Prepare text and answers for test
            dict_tests = {
                "Edwin Hubble's papers": {
                    "lookup": "Hubble",
                    "result": {"Edwin Hubble's papers": {"matches": 1, "meanings": 1}},
                },
                "A Hubble constant data": {
                    "lookup": "Hubble",
                    "result": {"Hubble constant data": {"matches": 1, "meanings": 2}},
                },
                "Hubble et al. 2000": {
                    "lookup": "Hubble",
                    "result": {"Hubble et al": {"matches": 0, "meanings": 1}},
                },
                "Kepler velocity fields.": {
                    "lookup": "Kepler",
                    "result": {"Kepler velocity fields": {"matches": 0, "meanings": 2}},
                },
                "Table 1 then gives the measured Kepler velocity data and Kepler planets.": {
                    "lookup": "Kepler",
                    "result": {
                        "measured Kepler velocity data": {"matches": 0, "meanings": 2},
                        "Kepler planets": {"matches": 1, "meanings": 2},
                    },
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    test_res = {}
                    for wordchunk in list_wordchunks:
                        curr_meaning, curr_inner_kw = testpaper._setup_consider_wordchunk(wordchunk, setup_data)
                        matches = testpaper._extract_ambig_phrases_substrings(
                            setup_data.list_exp_exact_ambigs,
                            "matches",
                            wordchunk.text,
                            curr_meaning,
                            curr_inner_kw,
                            setup_data,
                        )
                        meanings = testpaper._extract_ambig_phrases_substrings(
                            setup_data.list_exp_meaning_ambigs,
                            "meanings",
                            wordchunk.text,
                            curr_meaning,
                            curr_inner_kw,
                            setup_data,
                        )
                        test_res[wordchunk.text] = {"matches": len(matches), "meanings": len(meanings)}
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

    # For tests of _assemble_consider_wordchunk_results:
    if True:
        # Selects the best match from a list of ambiguous phrase matches, based on the shortest matched substring.
        def test__assemble_consider_wordchunk_results(self):
            # Prepare text and answers for test
            dict_tests = {
                "Edwin Hubble's papers": {
                    "lookup": "Hubble",
                    "result": {"Edwin Hubble's papers": ["matcher", "bool", "text_wordchunk", "text_database"]},
                },
                "A Hubble constant data": {
                    "lookup": "Hubble",
                    "result": {"Hubble constant data": ["matcher", "bool", "text_wordchunk", "text_database"]},
                },
                "Hubble et al. 2000": {
                    "lookup": "Hubble",
                    "result": {"Hubble et al": ["matcher", "bool", "text_wordchunk", "text_database"]},
                },
                "Kepler velocity fields.": {
                    "lookup": "Kepler",
                    "result": {"Kepler velocity fields": ["matcher", "bool", "text_wordchunk", "text_database"]},
                },
                "Table 1 then gives the measured Kepler velocity data and Kepler planets.": {
                    "lookup": "Kepler",
                    "result": {
                        "measured Kepler velocity data": ["matcher", "bool", "text_wordchunk", "text_database"],
                        "Kepler planets": ["matcher", "bool", "text_wordchunk", "text_database"],
                    },
                },
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            dict_ambigs = testpaper._process_database_ambig(do_verbose=False, keyword_objs=test_list_lookup_kobj)

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_kobjs = [test_dict_lookup_kobj[dict_tests[key1]["lookup"]]]
                    answer = dict_tests[key1]["result"]
                    # Set up initial variables
                    setup_data = testpaper._setup_check_truematch_vars(
                        text=curr_phrase,
                        keyword_objs=curr_kobjs,
                        do_verbose=False,
                        do_verbose_deep=False,
                        dict_ambigs=dict_ambigs,
                    )
                    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
                    test_res = {}
                    for wordchunk in list_wordchunks:
                        curr_meaning, curr_inner_kw = testpaper._setup_consider_wordchunk(wordchunk, setup_data)
                        set_matches = testpaper._extract_ambig_phrases_substrings(
                            setup_data.list_exp_exact_ambigs,
                            "matches",
                            wordchunk.text,
                            curr_meaning,
                            curr_inner_kw,
                            setup_data,
                        ) or testpaper._extract_ambig_phrases_substrings(
                            setup_data.list_exp_meaning_ambigs,
                            "meanings",
                            wordchunk.text,
                            curr_meaning,
                            curr_inner_kw,
                            setup_data,
                        )
                        list_results = testpaper._assemble_consider_wordchunk_results(
                            set_matches, wordchunk, curr_meaning, setup_data
                        )
                        test_res[wordchunk.text] = list(list_results["info"][0].keys())
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n{3}\n".format(test_res, answer, key1, test_res))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)
