"""
:title: test_paper.py

Testing the Paper class and its methods.
"""

import unittest

import numpy as np
import spacy
from nltk.corpus import wordnet  # type: ignore
from test_config import test_dict_lookup_kobj, test_list_lookup_kobj

from bibcat import config
from bibcat.core import grammar, paper
from bibcat.core import parameters as params

nlp = spacy.load(config.grammar.spacy_language_model)


class TestPaper(unittest.TestCase):
    # For tests of process_paragraphs and get_paragraphs:
    if True:
        # Test processing and extraction of paragraphs of target terms
        def test_processandget_paragraphs__variety(self):
            # placeholder = config.string_anymatch_ambig
            # Prepare text and answers for test
            text1 = "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
            # meanings1 = None #{"Hubble": ["Hubble Space Telescope"], "Kepler": None, "K2": None}
            # ambigs1 = [("our Hubble results", True)]

            text2 = "Kepler observations are presented in Section 1. Table 1 then gives the measured Kepler velocity data and Kepler planets. The Kepler images and Kepler plots indicate a correlation. The Kepler results are further shown in part 2. Note the Keplerian rotation."
            # meanings2 = None #{"Hubble": None, "Kepler": None, "K2": None}
            # ambigs2 = [
            #    ("Kepler observations", True),
            #    ("measured Kepler velocity data", False),
            #    ("Kepler planets", False),
            #    ("Kepler images", True),
            #    ("Kepler plots", True),
            #    ("Kepler results", True),
            # ]

            list_acts = [
                {
                    "text": text1,
                    "kobj": params.kobj_hubble,
                    "buffer": 0,
                    "answer": [
                        "Figure 1 plots the Hubble Space Telescope (HST) observations.",
                        "The HST stars are especially bright.",
                        "We summarize our Hubble results next.",
                    ],
                    # "acronym_meanings": meanings1,
                    # "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": params.kobj_hubble,
                    "buffer": 1,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section.",
                        "Some more filler content. We summarize our Hubble results next.",
                    ],
                    # "acronym_meanings": meanings1,
                    # "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": params.kobj_hubble,
                    "buffer": 3,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
                    ],
                    # "acronym_meanings": meanings1,
                    # "results_ambig": ambigs1,
                },
                {
                    "text": text1,
                    "kobj": params.kobj_hubble,
                    "buffer": 10,
                    "answer": [
                        "Despite the low S/N of the data, we still detect the stars. Figure 1 plots the Hubble Space Telescope (HST) observations. The HST stars are especially bright. We analyze them in the next section. A filler sentence. Here is another filler sentence (with parentheses). Some more filler content. We summarize our Hubble results next."
                    ],
                    # "acronym_meanings": meanings1,
                    # "results_ambig": ambigs1,
                },
                {
                    "text": text2,
                    "kobj": params.kobj_kepler,
                    "buffer": 0,
                    "answer": [
                        "Kepler observations are presented in Section 1.",
                        "The Kepler images and Kepler plots indicate a correlation.",
                        "The Kepler results are further shown in part 2.",
                    ],
                    # "acronym_meanings": meanings2,
                    # "results_ambig": ambigs2,
                },
                {
                    "text": text2,
                    "kobj": params.kobj_kepler,
                    "buffer": 1,
                    "answer": [
                        "Kepler observations are presented in Section 1. Table 1 then gives the measured Kepler velocity data and Kepler planets. The Kepler images and Kepler plots indicate a correlation. The Kepler results are further shown in part 2. Note the Keplerian rotation."
                    ],
                    # "acronym_meanings": meanings2,
                    # "results_ambig": ambigs2,
                },
            ]

            # Determine and check answers
            for info in list_acts:
                curr_text = info["text"]
                curr_kobj = info["kobj"]
                curr_buffer = info["buffer"]
                curr_answer = info["answer"]
                # curr_ambig = info["results_ambig"]
                # curr_acr_meanings = info["acronym_meanings"]
                curr_name = curr_kobj.get_name()

                # Prepare and run test for bibcat class instance
                testpaper = paper.Paper(text=curr_text, keyword_objs=test_list_lookup_kobj, do_check_truematch=True)
                _ = testpaper.process_paragraphs(buffer=curr_buffer)
                test_res = testpaper.get_paragraphs()[curr_name]

                # ambig_output = testpaper._get_info("_results_ambig")[curr_name]
                # test_ambig = [
                #    (item2["text_wordchunk"], item2["bool"])
                #    for item1 in ambig_output
                #    for item2 in item1["info"]
                #    if (item2["matcher"] is not None)
                # ]
                # test_acr_meanings = testpaper._get_info("_dict_acronym_meanings")

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                    # self.assertEqual(test_ambig, curr_ambig)
                    # self.assertEqual(test_acr_meanings, curr_acr_meanings)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_answer, curr_text))
                    # print("Test ambig: {0}\nAct. ambig: {1}".format(test_ambig, curr_ambig))
                    # print(
                    #    "Test acronym matches: {0}\nAct. acronym matches: {1}".format(
                    #        test_acr_meanings, curr_acr_meanings
                    #    )
                    # )
                    print("---")
                    print("")

                    self.assertEqual(test_res, curr_answer)
                    # self.assertEqual(test_ambig, curr_ambig)
                    # self.assertEqual(test_acr_meanings, curr_acr_meanings)

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

            # Determine and check answers
            for info in list_acts:
                curr_inds = info["inds"]
                curr_buffer = info["buffer"]
                curr_max = info["max"]
                curr_answer = info["result"]

                # Prepare and run test for bibcat class instance
                testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
                test_res = testpaper._buffer_indices(indices=curr_inds, buffer=curr_buffer, max_index=curr_max)

                # Check answer
                try:
                    self.assertEqual(test_res, curr_answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("Setup: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_answer, info))
                    print("---")
                    print("")

                    self.assertEqual(test_res, curr_answer)

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

            # Fill in rest of dictionary entries
            for key1 in dict_acts:
                curr_text = key1.split()
                dict_acts[key1]["text"] = key1
                dict_acts[key1]["synsets"] = []
                dict_acts[key1]["roots"] = []
                for curr_word in curr_text:
                    curr_syns = wordnet.synsets(curr_word)
                    curr_kobjs = [item for item in test_list_lookup_kobj if (item.identify_keyword(curr_word)["bool"])]
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

                # Process synsets into single string representation
                dict_acts[key1]["str_meaning"] = " ".join([" ".join(item) for item in dict_acts[key1]["roots"]])

            # Prepare and run test for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            for key1 in dict_acts:
                test_res = testpaper._extract_core_from_phrase(
                    phrase_NLP=nlp(key1),
                    do_skip_useless=False,
                    do_verbose=False,
                    keyword_objs=test_list_lookup_kobj,
                )

                # Check answer
                try:
                    self.assertEqual(test_res, dict_acts[key1])
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, dict_acts[key1], key1))
                    print("---")
                    print("")

                    self.assertEqual(test_res, dict_acts[key1])

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
            testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
            test_res = testpaper._split_text(text=text)
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

                self.assertEqual(test_res, answer)

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
            testpaper = paper.Paper(text="", keyword_objs=test_list_lookup_kobj, do_check_truematch=True)

            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testpaper._streamline_phrase(text=key1, do_streamline_etal=True)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)

        # Test streamlining for text with citations
        def test_streamline_phrase__citations(self):
            # Prepare text and answers for test
            dict_tests = {
                "Somename (2013) published  in SJ.": "{0} published in SJ.".format(
                    config.textprocessing.placeholder_author
                ),
                "Hubble (1953) was a landmark paper (for that subfield).": "{0} was a landmark paper (for that subfield).".format(
                    config.textprocessing.placeholder_author
                ),
                # "See also: Kepler [2023], Hubble & Author (2020), Author, Somename, and Kepler et al. [1990];": "See also: {0}, {0}, {0};".format(
                #    config.textprocessing.placeholder_author
                # ), - unrealistic citation case
                "See also: Kepler [2023], Hubble & Author (2020), Author and Kepler et al. [1990];": "See also: {0};".format(
                    config.textprocessing.placeholder_author
                ),
                "Also Author papers (Author et al. 1997, 2023),": "Also Author papers,",
                # "(Someone, Author, Somename et al. 1511; 1612)": "", - unrealistic citation case
                # "(Someone, Author, and Somename et al. 1913,15)": "", - unrealistic citation case
                "(Author et al. 80; Somename & Author 2012)": "",
                # "McThatname, Kepler, & Othername [1993] (see our paper)": "{0} (see our paper)".format(
                #    config.textprocessing.placeholder_author
                # ), - unrealistic citation case
                "{Othername et al. 1991} (see Hubble observations)": "(see Hubble observations)",
            }

            # Prepare and run tests for bibcat class instance
            testpaper = paper.Paper(text="", keyword_objs=test_list_lookup_kobj, do_check_truematch=True)

            # Check answers
            for key1 in dict_tests:
                try:
                    answer = dict_tests[key1]
                    test_res = testpaper._streamline_phrase(text=key1, do_streamline_etal=True)
                    self.assertEqual(test_res, answer)
                except AssertionError:
                    print("")
                    print(">")
                    print("{2}\nTest answer: {0}\nAct. answer: {1}\n".format(test_res, answer, key1))
                    print("---")
                    print("")

                    self.assertEqual(test_res, answer)
