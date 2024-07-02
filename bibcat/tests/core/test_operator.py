"""
:title: test_keyword.py

Testing the Keyword class and its methods.
"""

import unittest

from bibcat import parameters as params
from bibcat.core.operator import Operator


# Purpose: Testing the Operator class
class TestOperator(unittest.TestCase):
    # For tests of fetching keyword object using lookup item:
    if True:
        # Test fetching keyword object for objects with overlapping terms
        def test__fetch_keyword_object__overlap(self):
            # Prepare text and answers for test
            tmp_kobj_list = [params.kobj_hubble, params.kobj_hla]
            dict_acts = {
                "Hubble Legacy Archive": "HLA",
                "Hubble Legacy Archive results": "HLA",
                "Hubble": "Hubble",
                "HST": "Hubble",
                "HLA": "HLA",
                "Hubble Archive": "Hubble",
                "Hubble Legacy": "Hubble",
            }

            # Prepare and run test for bibcat class instance
            testbase = Operator(
                classifier=None,
                mode=None,
                keyword_objs=tmp_kobj_list,
                verbose=False,
                name="operator",
                load_check_truematch=False,
                deep_verbose=False,
            )
            # Determine and check answers
            for key1 in dict_acts:
                # Otherwise, check generated modif
                curr_lookup = key1
                test_res = testbase._fetch_keyword_object(
                    lookup=curr_lookup, do_raise_emptyerror=True
                )
                curr_answer = params.test_dict_lookup_kobj[dict_acts[key1]]

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

                    self.assertEqual(test_res, curr_answer)

                    self.assertEqual(test_res, curr_answer)
