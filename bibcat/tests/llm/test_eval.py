import pytest  # noqa: F401

from bibcat.llm.evaluate import evaluate_output, group_by_mission

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
    "2024Sci...123.3451L": [{"notes": "No mission-relevant content found.", "missions": []}],
    "2019arXiv190205569A": [{"notes": "", "missions": []}],
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


@pytest.mark.parametrize(
    "bibcode, return_source_value",
    [("2024Sci...123.3451L", paper), ("2019arXiv190205569A", None)],
)
def test_not_found(mocker, bibcode: str, return_source_value: dict | None):
    bibcode = "2024Sci...123.3451L"
    mocker.patch("bibcat.llm.evaluate.get_source", return_value=return_source_value)
    mocker.patch("bibcat.llm.evaluate.read_output", return_value=output[bibcode])

    df = evaluate_output(bibcode, write_file=False)
    assert df is None, "Expected df to be None"
