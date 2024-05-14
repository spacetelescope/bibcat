"""
:title: test_paper.py

Testing the Paper class and its methods.
"""

import unittest

from bibcat import config
from bibcat import parameters as params
from bibcat.core import keyword


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

            kobj3 = keyword.Keyword(keywords=[], acronyms=["AB....C", "A....... B...", "A.BC  D", "ABCD E", "AB C"])
            ans3 = "ABCDE"

            list_kobj = [kobj1, kobj2, kobj3]
            list_ans = [ans1, ans2, ans3]
            num_tests = len(list_kobj)

            # Prepare and run test for bibcat class instance
            for ii in range(0, num_tests):
                curr_kobj = list_kobj[ii]
                curr_ans = list_ans[ii]
                test_res = curr_kobj.get_name()

                # Check answer
                try:
                    self.assertEqual(test_res, curr_ans)
                except AssertionError:
                    print("")
                    print(">")
                    print("Instance: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_ans, curr_kobj))
                    print("---")
                    print("")

                    self.assertEqual(test_res, curr_ans)

    # For tests of is_keyword:
    if True:
        # Test determination of if given text contains keyword terms
        def test_is_keyword__variety(self):
            # Prepare text and answers for test
            dict_acts = {
                "Keplerian velocity": {"kobj": params.kobj_kepler, "bool": False},
                "that Hubble data": {"kobj": params.kobj_hubble, "bool": True},
                "that Hubble Space Telescope data": {"kobj": params.kobj_hubble, "bool": True},
                "that H S T data": {"kobj": params.kobj_hubble, "bool": True},
                "that H.S T data": {"kobj": params.kobj_hubble, "bool": True},
                "that HST data": {"kobj": params.kobj_hubble, "bool": True},
                "that HST PSF.": {"kobj": params.kobj_hubble, "bool": True},
                "hst": {"kobj": params.kobj_hubble, "bool": True},
                "HST": {"kobj": params.kobj_hubble, "bool": True},
                "HST.": {"kobj": params.kobj_hubble, "bool": True},
                "HST PSF.": {"kobj": params.kobj_hubble, "bool": True},
                "that HcST data": {"kobj": params.kobj_hubble, "bool": False},
                "that A.H.S.T.M. data": {"kobj": params.kobj_hubble, "bool": False},
                "that A.H.S.T.M data": {"kobj": params.kobj_hubble, "bool": False},
                "that AHSTM data": {"kobj": params.kobj_hubble, "bool": False},
                "that LHST data": {"kobj": params.kobj_hubble, "bool": False},
                "that HS.xT data": {"kobj": params.kobj_hubble, "bool": False},
                "that K23 data": {"kobj": params.kobj_k2, "bool": False},
                "that AK2 data": {"kobj": params.kobj_k2, "bool": False},
                "that K2 data": {"kobj": params.kobj_k2, "bool": True},
                "that K 2 star": {"kobj": params.kobj_k2, "bool": False},
                "archive observations": {"kobj": params.kobj_hla, "bool": False},
                "hubble and archive observations": {"kobj": params.kobj_hla, "bool": False},
                "Hubble and Archive observations": {"kobj": params.kobj_hla, "bool": False},
                "they took Hubble images": {"kobj": params.kobj_hla, "bool": False},
                "they took HST images": {"kobj": params.kobj_hla, "bool": False},
                "the hubble legacy archive": {"kobj": params.kobj_hla, "bool": True},
                "the hla uncapitalized": {"kobj": params.kobj_hla, "bool": True},
                "the Hubble Archive is different": {"kobj": params.kobj_hla, "bool": False},
            }

            # Prepare and run test for bibcat class instance
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]
                curr_bool = dict_acts[phrase]["bool"]
                test_res = curr_kobj.is_keyword(text=phrase)

                # Check answer
                try:
                    self.assertEqual(test_res, curr_bool)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, curr_bool, phrase))
                    print("---")
                    print("")

                    self.assertEqual(test_res, curr_bool)

    # For tests of replace_keyword:
    if True:
        # Test removal of keyword terms from text
        def test_replace_keyword__variety(self):
            # Prepare text and answers for test
            placeholder = config.textprocessing.placeholder_anon
            dict_acts = {
                "Keplerian velocity": {"kobj": params.kobj_kepler, "result": "Keplerian velocity"},
                "that Hubble data": {"kobj": params.kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that Hubble Space Telescope data": {
                    "kobj": params.kobj_hubble,
                    "result": "that {0} data".format(placeholder),
                },
                "that Hubble Telescope data": {
                    "kobj": params.kobj_hubble,
                    "result": "that {0} data".format(placeholder),
                },
                "hst": {"kobj": params.kobj_hubble, "result": "{0}".format(placeholder)},
                "HST": {"kobj": params.kobj_hubble, "result": "{0}".format(placeholder)},
                "that H S T data": {"kobj": params.kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that H.S T data": {"kobj": params.kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that HST data": {"kobj": params.kobj_hubble, "result": "that {0} data".format(placeholder)},
                "that HST PSF.": {"kobj": params.kobj_hubble, "result": "that {0} PSF.".format(placeholder)},
                "HST PSF.": {"kobj": params.kobj_hubble, "result": "{0} PSF.".format(placeholder)},
                "that HcST data": {"kobj": params.kobj_hubble, "result": "that HcST data"},
                "that A.H.S.T.M. data": {"kobj": params.kobj_hubble, "result": "that A.H.S.T.M. data"},
                "that LHST data": {"kobj": params.kobj_hubble, "result": "that LHST data"},
                "that HS.xT data": {"kobj": params.kobj_hubble, "result": "that HS.xT data"},
                "that K23 data": {"kobj": params.kobj_k2, "result": "that K23 data"},
                "that AK2 data": {"kobj": params.kobj_k2, "result": "that AK2 data"},
                "that K2 data": {"kobj": params.kobj_k2, "result": "that {0} data".format(placeholder)},
                "that K 2 star": {"kobj": params.kobj_k2, "result": "that K 2 star"},
            }

            # Prepare and run test for bibcat class instance
            for phrase in dict_acts:
                curr_kobj = dict_acts[phrase]["kobj"]
                act_res = dict_acts[phrase]["result"]
                test_res = curr_kobj.replace_keyword(text=phrase, placeholder=placeholder)

                # Check answer
                try:
                    self.assertEqual(test_res, act_res)
                except AssertionError:
                    print("")
                    print(">")
                    print("Text: {2}\nTest answer: {0}\nAct. answer: {1}".format(test_res, act_res, phrase))
                    print("---")
                    print("")

                    self.assertEqual(test_res, act_res)
