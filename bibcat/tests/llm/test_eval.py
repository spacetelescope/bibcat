from bibcat.llm.evaluate import evaluate_output

paper = {
    "bibcode": "2022Sci...377.1211L",
    "title": ["Density, not radius, separates rocky and water-rich small planets orbiting M dwarf stars"],
    "abstract": "This is the abstract",
    "body": "This is the paper text of the source dataset. I am a TESS paper.",
    "class_missions": {"TESS": {"bibcode": "2022Sci...377.1211L", "papertype": "SCIENCE"}},
}

output = {
    "2022Sci...377.1211L": [
        {"TESS": ["SCIENCE", [0.9, 0.1]], "JWST": ["MENTION", [0.3, 0.7]]},
        {"TESS": ["MENTION", [0.3, 0.7]]},
        {"JWST": ["SCIENCE", [0.9, 0.1]], "TESS": ["SCIENCE", [0.9, 0.1]]},
        {"TESS": ["SCIENCE", [0.9, 0.1]]},
        {"TESS": ["SCIENCE", [0.9, 0.1]]},
        {"JWST": ["MENTION", [0.3, 0.7]], "TESS": ["SCIENCE", [0.9, 0.1]]},
    ]
}


def test_evaluate_df(mocker):
    bibcode = "2022Sci...377.1211L"
    mocker.patch("bibcat.llm.evaluate.get_source", return_value=paper)
    mocker.patch("bibcat.llm.evaluate.read_output", return_value=output[bibcode])

    df = evaluate_output(bibcode, write_file=False)

    assert len(df) == 4
    assert set(df["llm_mission"]) == {"TESS", "JWST"}
    # assert the first row is hallucinated
    assert df.iloc[0]["hallucination_by_llm"]

    # check the last TESS-SCIENCE row
    intext = df["in_human_class"] == True  # noqa: E712
    assert len(df[intext]) == 1
    assert df[intext].iloc[0]["mission_in_text"]
    assert df[intext].iloc[0]["llm_mission"] == "TESS"
    assert df[intext].iloc[0]["llm_papertype"] == "SCIENCE"
    assert not df[intext].iloc[0]["hallucination_by_llm"]
