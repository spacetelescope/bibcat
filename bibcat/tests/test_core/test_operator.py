"""
:title: test_keyword.py

Testing the Keyword class and its methods.
"""

import unittest

from bibcat import parameters as params
from bibcat.core.operator import Operator

# Keyword objects
kobj_hla = params.keyword_obj_HLA
kobj_hubble = params.keyword_obj_HST
kobj_kepler = params.keyword_obj_Kepler
kobj_k2 = params.keyword_obj_K2

# Keyword-object lookups
list_lookup_kobj = [kobj_hubble, kobj_kepler, kobj_k2]
dict_lookup_kobj = {"Hubble": kobj_hubble, "Kepler": kobj_kepler, "K2": kobj_k2, "HLA": kobj_hla}


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

            # Prepare and run test for bibcat class instance
            testbase = Operator(
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
