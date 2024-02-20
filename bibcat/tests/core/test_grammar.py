"""
:title: test_grammar.py

Testing the Grammar class and its methods.
"""

import unittest

from bibcat import config
from bibcat import parameters as params
from bibcat.core.grammar import Grammar


class TestGrammar(unittest.TestCase):
    # For tests of generating and fetching modifs:
    if True:
        # Test modif generation for basic example text
        def test__modifs__basic(self):
            # Prepare text and answers for test
            dict_acts = {
                "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST. It is a telescope that is used across the world. The HST legacy archive is a rich treasure chest of ultraviolet observations used across the world.": {
                    "kobj": params.kobj_hubble,
                    "none": "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST.\nThe HST legacy archive is a rich treasure chest of ultraviolet observations used across the world.",
                    "skim": "The Hubble Space Telescope is a telescope that is referred to as Hubble or HST.\nThe HST legacy archive is a treasure chest of observations used across the world.",
                    "trim": "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST.\nThe HST legacy archive is a rich treasure chest of ultraviolet observations.",
                    "anon": "The {0} is a really neat telescope that is often referred to as {0} or {0}.\nThe {0} legacy archive is a rich treasure chest of ultraviolet observations used across the world.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "The {0} is a telescope that is referred to as {0} or {0}.\nThe {0} legacy archive is a treasure chest of observations.".format(
                        config.placeholder_anon
                    ),
                }
            }

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=config.test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]

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
                    "kobj": params.kobj_hubble,
                    "none": "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.",
                    "skim": "Once the transit is observable, we will observe the target in a fashion without delay and at resolution using either JWST or Hubble.",
                    "trim": "we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.",
                    "anon": "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or {0}.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "we will observe the target in a fashion without delay and at resolution using either JWST or {0}.".format(
                        config.placeholder_anon
                    ),
                }
            }

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(
                    text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=config.test_which_modes)
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

        # Test modif generation for example text with clause with mission information
        def test__modifs__importantclause_mission(self):
            # Prepare text and answers for test
            dict_acts = {
                "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.": {
                    "kobj": params.kobj_hubble,
                    "none": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.",
                    "skim": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the HST observations, we are unable to detect those targets in the snapshots taken with the Cool Telescope.",
                    "trim": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the HST observations, we are unable to even tentatively detect those same targets in the snapshots.",
                    "anon": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the {0} observations, we are unable to even tentatively detect those same targets in the snapshots taken with the Cool Telescope.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the {0} observations, we are unable to detect those targets in the snapshots.".format(
                        config.placeholder_anon
                    ),
                }
            }

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=config.test_which_modes)
                # Iterate through modes
                for key1 in dict_acts[phrase]:
                    # Skip over non-mode keys
                    if key1 in ["kobj"]:
                        continue
                    # Otherwise, check generated modif
                    test_res = testbase.get_modifs()[key1]
                    curr_answer = dict_acts[phrase][key1]

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

        # Test modif generation for example text with important clause with 1st-person
        def test__modifs__importantclause_1stperson(self):
            # Prepare text and answers for test
            dict_acts = {
                "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.": {
                    "kobj": params.kobj_hubble,
                    "none": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "skim": "While we are able to detect (above 5sigma) the three stars within the mosaic created by the Cool Telescope observations, we are unable to detect those targets in the snapshots taken with the HST.",
                    "trim": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic, we are unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "anon": "While we are able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, we are unable to even tentatively detect those same targets in the snapshots taken with the {0}.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "While we are able to detect (above 5sigma) the three stars within the mosaic, we are unable to detect those targets in the snapshots taken with the {0}.".format(
                        config.placeholder_anon
                    ),
                }
            }

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(
                    text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=config.test_which_modes)
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
                    "kobj": params.kobj_hubble,
                    "none": "While that study was able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, this study was unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "skim": "While that study was able to detect (above 5sigma) the three stars within the mosaic created by the Cool Telescope observations, this study was unable to detect those targets in the snapshots taken with the HST.",
                    "trim": "this study was unable to even tentatively detect those same targets in the snapshots taken with the HST.",
                    "anon": "While that study was able to significantly detect (above 5sigma) the three massive stars within the mosaic created by the Cool Telescope observations, this study was unable to even tentatively detect those same targets in the snapshots taken with the {0}.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "this study was unable to detect those targets in the snapshots taken with the {0}.".format(
                        config.placeholder_anon
                    ),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True)
                testbase.run_modifications(which_modes=config.test_which_modes)
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
        @unittest.expectedFailure
        def test__modifs__unimportantconjunction(self):
            # Prepare text and answers for test
            dict_acts = {
                "The red stars are observed with HST and further verified with JWST.": {
                    "kobj": params.kobj_hubble,
                    "none": "The red stars are observed with HST and further verified with JWST.",
                    "skim": "The stars are observed with HST and verified with JWST.",
                    "trim": "The red stars are observed with HST.",
                    "anon": "The red stars are observed with {0} and further verified with JWST.".format(
                        config.placeholder_anon
                    ),
                    "skim_trim_anon": "The stars are observed with {0}.".format(config.placeholder_anon),
                }
            }
            #

            # Determine and check answers
            for phrase in dict_acts:
                # curr_kobj = dict_acts[phrase]["kobj"]

                # Prepare and run test for bibcat class instance
                testbase = Grammar(
                    text=phrase, keyword_obj=params.kobj_hubble, do_check_truematch=True, do_verbose=False
                )
                testbase.run_modifications(which_modes=config.test_which_modes)
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
