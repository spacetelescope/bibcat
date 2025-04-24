"""
:title: test_base.py

Testing the Base class and its methods.
"""

import unittest

import numpy as np
import spacy  # type: ignore
from test_config import test_list_lookup_kobj

from bibcat import config
from bibcat.core import parameters as params
from bibcat.core.base import Base

nlp = spacy.load(config.grammar.spacy_language_model)


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

            # Prepare and run test for bibcat class instance
            testbase = Base()
            # For tests where verbs are not included
            for phrase in dict_acts_noverbs:
                test_res = testbase._assemble_keyword_wordchunks(
                    text=phrase, keyword_objs=test_list_lookup_kobj, do_include_verbs=False, do_verbose=False
                )
                test_res = [item.text for item in test_res]

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

                    self.assertEqual(test_res, dict_acts_noverbs[phrase])

            # For tests where verbs are indeed included
            for phrase in dict_acts_yesverbs:
                test_res = testbase._assemble_keyword_wordchunks(
                    text=phrase, keyword_objs=test_list_lookup_kobj, do_include_verbs=True, do_verbose=False
                )
                test_res = [item.text for item in test_res]

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

                    self.assertEqual(test_res, dict_acts_yesverbs[phrase])

    # For tests of _check_importance:
    if True:
        # Test determination of importance of given text with various terms
        def test__check_importance__variety(self):
            # Prepare text and answers for test
            kobj = params.kobj_hubble
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

            # Prepare and run test for bibcat class instance
            testbase = Base()
            for phrase in dict_acts:
                test_res = testbase._check_importance(text=phrase, keyword_objs=[kobj])["bools"]
                list_res = [key for key in test_res if (test_res[key])]

                # Check answer
                try:
                    self.assertEqual(sorted(list_res), sorted(dict_acts[phrase]))
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[phrase], phrase))
                    print("---")
                    print("")

                    self.assertEqual(sorted(list_res), sorted(dict_acts[phrase]))

    # For tests of _cleanse_text:
    if True:
        # Test _cleanse_text for variety of text
        def test__cleanse_text__variety(self):
            # Prepare text and answers for test
            # kobj = params.kobj_hubble
            dict_acts = {
                "  J.W .. . S. - T .  .S-c": "J.W. S. - T.S-c",
                " We  walked to the   store/shop - across   the street. \n   We bought:   carrots-celery, and  - tofu-juice /milk.  ": "We walked to the store/shop - across the street. \n We bought: carrots-celery, and - tofu-juice /milk.",
                " The pets ( cats ,  dogs,  sheep  ) all went to the field.  They ' ve all [ allegedly  ] pink (re : : salmon ) fur .   That isn' t typical [apparently (e.g. Cite  2023 ) ].": "The pets (cats, dogs, sheep) all went to the field. They've all [allegedly] pink (re: salmon) fur. That isn't typical [apparently (e.g. Cite 2023)].",
                " The poetry... It was beautiful (according to the audience.).": "The poetry. It was beautiful (according to the audience.).",
            }
            #

            # Prepare and run test for bibcat class instance
            testbase = Base()
            for phrase in dict_acts:
                test_res = testbase._cleanse_text(text=phrase, do_streamline_etal=True)

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts[phrase])
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[phrase], phrase))
                    print("---")
                    print("")

                    self.assertEqual(test_res, dict_acts[phrase])

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos, do_verbose=False
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

                    self.assertEqual(test_res, answer)

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
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos, do_verbose=False
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos)
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

                    self.assertEqual(test_res, answer)

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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    curr_phrase = key1
                    curr_NLP = nlp(curr_phrase)
                    answer = dict_tests[key1]
                    test_bools = np.array(
                        [
                            testbase._is_pos_word(
                                word=item, keyword_objs=test_list_lookup_kobj, pos=test_pos, do_verbose=False
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

            # Prepare and run tests for bibcat class instance
            testbase = Base()

            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testbase._search_text(text=key1, keyword_objs=test_list_lookup_kobj)["bool"]
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")
                    #
                    self.assertEqual(test_res, answer)
