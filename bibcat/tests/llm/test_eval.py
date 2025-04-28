import pytest  # noqa: F401

from bibcat.core import parameters as params
from bibcat.core.operator import Operator
from bibcat.llm.evaluate import _fetch_keyword_object, evaluate_output, group_by_mission
from bibcat.tests.core.test_config import test_dict_lookup_kobj

paper = {
    "bibcode": "2022Sci...377.1211L",
    "title": ["Density, not radius, separates rocky and water-rich small planets orbiting M dwarf stars"],
    "abstract": "This is the abstract",
    "body": "This is the paper text of the source dataset. I am a TESS paper.",
    "class_missions": {"TESS": {"bibcode": "2022Sci...377.1211L", "papertype": "SCIENCE"}},
}

output = {
    "2022Sci...377.1211L": [
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention JWST",
                    "quotes": ["We mention JWST."],
                },
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention TESS",
                    "quotes": ["We mention TESS."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use JWST data",
                    "quotes": ["We use JWST data."],
                },
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention JWST",
                    "quotes": ["We mention JWST."],
                },
            ],
        },
    ],
    "2024Sci...377.1211L": [{"notes": "this paper has no missions.", "missions": []}],
}


def test_evaluate_df(mocker):
    bibcode = "2022Sci...377.1211L"
    mocker.patch("bibcat.llm.evaluate.get_source", return_value=paper)
    mocker.patch("bibcat.llm.evaluate.read_output", return_value=output[bibcode])

    df = evaluate_output(bibcode, write_file=False)

    assert len(df) == 4
    assert set(df["llm_mission"]) == {"TESS", "JWST"}
    # assert the first row (JWST) is hallucinated
    assert df.iloc[0]["hallucination_by_llm"]

    # assert the second row (JWST-SCIENCE) has a low weight
    assert df.iloc[1]["llm_mission"] == "JWST"
    assert df.iloc[1]["llm_papertype"] == "SCIENCE"
    assert df.iloc[1]["mean_llm_confidences"].tolist() == [0.9, 0.1]
    assert df.iloc[1]["count"] == 1
    assert df.iloc[1]["n_runs"] == 6
    assert df.iloc[1]["weighted_confs"].tolist() == [0.15, 0.017]

    # check the last TESS-SCIENCE row
    intext = df["in_human_class"] == True  # noqa: E712
    assert len(df[intext]) == 1
    assert df[intext].iloc[0]["mission_in_text"]
    assert df[intext].iloc[0]["llm_mission"] == "TESS"
    assert df[intext].iloc[0]["llm_papertype"] == "SCIENCE"
    assert not df[intext].iloc[0]["hallucination_by_llm"]

    # assert last row has a high weight
    assert df[intext].iloc[0]["mean_llm_confidences"].tolist() == [0.9, 0.1]
    assert df[intext].iloc[0]["count"] == 5
    assert df[intext].iloc[0]["n_runs"] == 6
    assert df[intext].iloc[0]["weighted_confs"].tolist() == [0.75, 0.083]


def test_group_by_mission(mocker):
    bibcode = "2022Sci...377.1211L"
    mocker.patch("bibcat.llm.evaluate.get_source", return_value=paper)
    mocker.patch("bibcat.llm.evaluate.read_output", return_value=output[bibcode])

    df = evaluate_output(bibcode, write_file=False)
    mm = group_by_mission(df)

    # JWST
    assert mm.iloc[0]["llm_mission"] == "JWST"
    assert mm.iloc[0]["total_mission_conf"] == 0.5
    assert mm.iloc[0]["total_weighted_conf"].tolist() == [0.25, 0.25]
    assert mm.iloc[0]["prob_mission"] == 0.333
    assert mm.iloc[0]["prob_papertype"].tolist() == [0.5, 0.5]

    # TESS
    assert mm.iloc[1]["llm_mission"] == "TESS"
    assert mm.iloc[1]["total_mission_conf"] == 1.0
    assert mm.iloc[1]["total_weighted_conf"].tolist() == [0.8, 0.2]
    assert mm.iloc[1]["prob_mission"] == 0.667
    assert mm.iloc[1]["prob_papertype"].tolist() == [0.8, 0.2]


def test__fetch_keyword_object__overlap(self):
    # Prepare text and answers for test
    # tmp_kobj_list = [params.kobj_hubble, params.kobj_hla]
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
    """testbase = Operator(
        classifier=None,
        mode=None,
        keyword_objs=tmp_kobj_list,
        verbose=False,
        name="operator",
        load_check_truematch=False,
        deep_verbose=False,
    )"""
    # Determine and check answers
    for key1 in dict_acts:
        # Otherwise, check generated modif
        curr_lookup = key1
        test_res = _fetch_keyword_object(lookup=curr_lookup, do_raise_emptyerror=True)
        curr_answer = test_dict_lookup_kobj[dict_acts[key1]]

        # Check answer
        try:
            self.assertEqual(test_res, curr_answer)
        except AssertionError:
            print("")
            print(">")
            print(("Text: {2}\n\nTest answer: {0}\n" + "\nAct. answer: {1}\n").format(test_res, curr_answer, key1))
            print("---")
            print("")

            self.assertEqual(test_res, curr_answer)

            self.assertEqual(test_res, curr_answer)
