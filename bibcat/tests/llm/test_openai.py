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
    "notes": "I have read the paper and identified that it includes references to the Kepler mission, specifically mentioning the code Kepler in the context of comparing different stellar evolution programs and their output. However, the paper does not present any new observations or data utilizing Kepler; it simply mentions it as part of the discussion on model comparisons. Therefore, I classify the reference as a mention, not as a use of data in analysis or the presentation of new results.",
    "missions": [
        {
            "mission": "Kepler",
            "papertype": "MENTION",
            "confidence": [0.1, 0.9],
            "reason": "Kepler is referenced in the context of comparing stellar evolution models but does not contribute new data or observations.",
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

    assert out["missions"][0]["mission"] == "Kepler"


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

    monkeypatch.setenv("OPENAI_API_KEY", "testkey")
    mocker.patch("bibcat.llm.openai.get_source", return_value=paper)
    config = fixconfig(str(d), "bibcat.llm.openai")

    monkeypatch.setitem(config.llms, "batch_file", "tmp_batch.jsonl")
    batch_file = d / config.paths.output / f"llms/openai_{config.llms.openai.model}/{config.llms.batch_file}"

    oa = OpenAIHelper()
    oa.create_batch_file(bibcodes=bibcodes)
    assert batch_file.exists()
    assert batch_file.suffix == ".jsonl"

    with open(batch_file, "r") as f:
        lines = f.read().splitlines()
        assert len(lines) == len(bibcodes)
        line = json.loads(lines[0])
        assert line["custom_id"] == bibcodes[0]
        assert "This is the abstract" in line["body"]["input"]

# fmt: off
example_request = {'method': 'POST', 'url': '/v1/responses', 'body': {'model': 'gpt-4.1-mini', 'instructions': 'please check this', 'input': 'what is this?', 'text': {'format': {'type': 'json_schema', 'strict': True, 'name': 'InfoModel', 'schema': {'$defs': {'MissionEnum': {'enum': ['HST', 'JWST', 'Roman', 'TESS', 'KEPLER', 'K2', 'GALEX', 'PanSTARRS', 'FUSE', 'IUE', 'HUT', 'UIT', 'WUPPE', 'BEFS', 'TUES', 'IMAPS', 'EUVE'], 'title': 'MissionEnum', 'type': 'string'}, 'MissionInfo': {'description': 'Pydantic model for a mission entry', 'properties': {'mission': {'description': 'The name of the mission.', 'enum': ['HST', 'JWST', 'Roman', 'TESS', 'KEPLER', 'K2', 'GALEX', 'PanSTARRS', 'FUSE', 'IUE', 'HUT', 'UIT', 'WUPPE', 'BEFS', 'TUES', 'IMAPS', 'EUVE'], 'title': 'MissionEnum', 'type': 'string'}, 'papertype': {'description': 'The type of paper you think it is', 'enum': ['SCIENCE', 'MENTION'], 'title': 'PapertypeEnum', 'type': 'string'}, 'quotes': {'description': 'A list of exact quotes from the paper that support your reason', 'items': {'type': 'string'}, 'title': 'Quotes', 'type': 'array'}, 'reason': {'description': 'A short sentence summarizing your reasoning for classifying this mission + papertype', 'title': 'Reason', 'type': 'string'}, 'confidence': {'description': 'Two float values representing confidence for SCIENCE and MENTION. Must sum to 1.0.', 'items': {'type': 'number'}, 'title': 'Confidence', 'type': 'array'}}, 'required': ['mission', 'papertype', 'quotes', 'reason', 'confidence'], 'title': 'MissionInfo', 'type': 'object', 'additionalProperties': False}, 'PapertypeEnum': {'description': 'Enumeration of paper types for classification', 'enum': ['SCIENCE', 'MENTION'], 'title': 'PapertypeEnum', 'type': 'string'}}, 'description': 'Pydantic model for the parsed response from the LLM', 'properties': {'notes': {'description': 'all your notes and thoughts you have written down during your process', 'title': 'Notes', 'type': 'string'}, 'missions': {'description': 'a list of your identified missions', 'items': {'$ref': '#/$defs/MissionInfo'}, 'title': 'Missions', 'type': 'array'}}, 'required': ['notes', 'missions'], 'title': 'InfoModel', 'type': 'object', 'additionalProperties': False}}}}}


@pytest.fixture()
def batchfile(tmp_path):
    """fixture for creating a batch file"""

    def _make_batch(n_samp):
        path = tmp_path / "batch.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(1, n_samp + 1):
                data = {"custom_id": f"bc_{i:03}"} | example_request
                f.write(json.dumps(data) + "\n")
        return path

    return _make_batch


@pytest.mark.parametrize("nsamp, exp", [(50, True), (60000, False)])
def test_validate_batch(batchfile, nsamp, exp):
    """test if batch file is validated correctly"""
    batch = batchfile(nsamp)
    oa = OpenAIHelper()
    valid = oa.is_batch_file_validated(batch)
    assert valid == exp
