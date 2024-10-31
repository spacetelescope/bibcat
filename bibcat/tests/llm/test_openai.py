import pytest

from bibcat.llm.openai import convert_to_classification, extract_response


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


@pytest.mark.parametrize(
    "data, exp",
    [
        (
            '\nOUTPUT:\n```json\n{\n    "HST": [\n        "MENTION",\n        0.8\n    ],\n    "JWST": [\n        "SCIENCE",\n        0.95\n    ]\n}\n```',
            {"HST": ["MENTION", 0.8], "JWST": ["SCIENCE", 0.95]},
        ),
        (
            '```json\n{\n  "title": "DampingwingsintheLyman-αforest",\n  "primary_mission_or_survey": ["X-Shooter"],\n  "other_missions_or_surveys_mentioned": ["Planck"],\n  "notes": ""\n}\n```',
            {
                "title": "DampingwingsintheLyman-αforest",
                "primary_mission_or_survey": ["X-Shooter"],
                "other_missions_or_surveys_mentioned": ["Planck"],
                "notes": "",
            },
        ),
        ("There is no json here.", {"error": "No JSON content found in response"}),
        (
            '```json\n{"field": ["A", "B", "C",]}\n```',
            {"error": 'Error decoding JSON content: "Expecting value: line 1 column 23 (char 22)"'},
        ),
    ],
    ids=["json1", "json2", "nojson", "badjson"],
)
def test_extract_json(data, exp):
    """test we can extract some json content"""

    output = extract_response(data)
    assert output == exp
