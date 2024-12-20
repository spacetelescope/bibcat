"""
:title: test_paper.py

Testing the Paper class and its methods.
"""

import unittest

from bibcat import parameters as params
from bibcat.core import paper


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
                testbase = paper.Paper(
                    text=curr_text, keyword_objs=params.test_list_lookup_kobj, do_check_truematch=True
                )
                _ = testbase.process_paragraphs(buffer=curr_buffer)
                test_res = testbase.get_paragraphs()[curr_name]

                # ambig_output = testbase._get_info("_results_ambig")[curr_name]
                # test_ambig = [
                #    (item2["text_wordchunk"], item2["bool"])
                #    for item1 in ambig_output
                #    for item2 in item1["info"]
                #    if (item2["matcher"] is not None)
                # ]
                # test_acr_meanings = testbase._get_info("_dict_acronym_meanings")

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
                testbase = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
                test_res = testbase._buffer_indices(indices=curr_inds, buffer=curr_buffer, max_index=curr_max)

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
            testbase = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
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

                self.assertEqual(test_res, answer)
