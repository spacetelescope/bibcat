import json
from enum import Enum

import pytest

from bibcat.llm.openai import InfoModel, OpenAIHelper, convert_to_classification, extract_response


def test_convert_to_classes():
    """test we can convert to classifications"""
    data = {"HST": ["MENTION", [0.8, 0.2]], "JWST": ["SCIENCE", [0.9, 0.1]], "Roman": ["MENTION", [0.9, 0.1]]}
    res = convert_to_classification(output=data, bibcode="12345")

    assert "HST" in res
    assert "Kepler" not in res
    assert res["JWST"] == {"bibcode": "12345", "papertype": "SCIENCE"}


def test_convert_error():
    """test convert fails correctly"""
    data = {"error": "No JSON content found in response"}
    res = convert_to_classification(output=data, bibcode="12345")
    assert res is None


def test_convert_failure(caplog):
    """test convert fails correctly"""
    res = convert_to_classification(output={"good": "but", "bad": "json"}, bibcode="12345")
    assert res is None
    assert "Error converting output to classification format" in caplog.record_tuples[0][2]


INPUT_JSON1 = '\nOUTPUT:\n```json\n{\n    "HST": [\n        "MENTION",\n        0.8\n    ],\n    "JWST": [\n        "SCIENCE",\n        0.95\n    ]\n}\n```'
OUTPUT_JSON1 = {"HST": ["MENTION", 0.8], "JWST": ["SCIENCE", 0.95]}

INPUT_JSON2 = '```json\n{\n  "title": "DampingwingsintheLyman-αforest",\n  "primary_mission_or_survey": ["X-Shooter"],\n  "other_missions_or_surveys_mentioned": ["Planck"],\n  "notes": ""\n}\n```'
OUTPUT_JSON2 = {
    "title": "DampingwingsintheLyman-αforest",
    "primary_mission_or_survey": ["X-Shooter"],
    "other_missions_or_surveys_mentioned": ["Planck"],
    "notes": "",
}

INPUT_NOJSON = "There is no json here."
OUTPUT_NOJSON = {"error": "No JSON content found in response"}

INPUT_BADJSON = '```json\n{"field": ["A", "B", "C",]}\n```'
OUTPUT_BADJSON = {"error": 'Error decoding JSON content: "Expecting value: line 1 column 26 (char 25)"'}


@pytest.mark.parametrize(
    "data, exp",
    [
        (
            INPUT_JSON1,
            OUTPUT_JSON1,
        ),
        (
            INPUT_JSON2,
            OUTPUT_JSON2,
        ),
        (
            INPUT_NOJSON,
            OUTPUT_NOJSON,
        ),
        (
            INPUT_BADJSON,
            OUTPUT_BADJSON,
        ),
    ],
    ids=["json1", "json2", "nojson", "badjson"],
)
def test_extract_json(data, exp):
    """test we can extract some json content"""

    output = extract_response(data)
    assert output == exp


# expected model response
exp_response = {
    "notes": "I have read the paper and identified that it includes references to the KEPLER mission, specifically mentioning the code KEPLER in the context of comparing different stellar evolution programs and their output. However, the paper does not present any new observations or data utilizing KEPLER; it simply mentions it as part of the discussion on model comparisons. Therefore, I classify the reference as a mention, not as a use of data in analysis or the presentation of new results.",
    "missions": [
        {
            "mission": "KEPLER",
            "papertype": "MENTION",
            "confidence": [0.1, 0.9],
            "reason": "KEPLER is referenced in the context of comparing stellar evolution models but does not contribute new data or observations.",
            "quotes": [
                "The programs mentioned above are some of the most commonly cited in the literature, but this is by no means an exhaustive list of all current software in use."
            ],
        }
    ],
}


def test_info_model():
    """test the info model parses and dumps correctly"""
    ii = InfoModel(**exp_response)
    assert isinstance(ii.missions[0].mission, Enum)
    assert isinstance(ii.missions[0].papertype, Enum)
    assert ii.missions[0].papertype == "MENTION"

    out = ii.model_dump()
    assert isinstance(out["missions"][0]["papertype"], str)
    assert out["missions"][0]["papertype"] == "MENTION"
    assert out["missions"][0]["mission"] == "KEPLER"


bibcodes = ["2018A&A...610A..11I", "2018A&A...610A..86S"]
paper = {
    "title": ["Density, not radius, separates rocky and water-rich small planets orbiting M dwarf stars"],
    "abstract": "This is the abstract",
    "body": "This is the paper text of the source dataset. I am a TESS paper.",
}


def test_create_batch_file(fixconfig, mocker, monkeypatch, tmp_path):
    """test we can create a batch file"""
    d = tmp_path / "llm"
    d.mkdir()

    mocker.patch("bibcat.llm.openai.get_source", return_value=paper)
    config = fixconfig(str(d), "bibcat.llm.openai")

    monkeypatch.setitem(config.llms, "batch_file", "tmp_batch.jsonl")
    batch_file = d / config.paths.output / f"llms/openai_{config.llms.openai.model}/{config.llms.batch_file}"

    oa = OpenAIHelper()
    oa.create_batch_file(bibcodes=bibcodes)
    # breakpoint()
    assert batch_file.exists()
    assert batch_file.suffix == ".jsonl"

    with open(batch_file, "r") as f:
        lines = f.read().splitlines()
        assert len(lines) == len(bibcodes)
        line = json.loads(lines[0])
        assert line["custom_id"] == bibcodes[0]
        assert "This is the abstract" in line["body"]["input"]