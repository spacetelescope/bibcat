
import pytest

from bibcat.llm.openai import check_response, convert_to_classification


def test_convert_to_classes():
    """ test we can convert to classifications """
    data = {'HST': ['MENTION', 0.8], 'JWST': ['SCIENCE', 0.9], 'Roman': ['MENTION', 0.1]}
    res = convert_to_classification(output=data, bibcode='12345')

    assert 'HST' in res
    assert 'Roman' not in res
    assert res['JWST'] == {'bibcode': '12345', 'papertype': 'SCIENCE'}


def test_convert_error():
    """ test convert fails correctly  """
    data = {'error': 'No JSON content found in response'}
    res = convert_to_classification(output=data, bibcode='12345')
    assert res is None


@pytest.mark.parametrize('data, exp',
                    [('\nOUTPUT:\n```json\n{\n    "HST": [\n        "MENTION",\n        0.8\n    ],\n    "JWST": [\n        "SCIENCE",\n        0.95\n    ]\n}\n```',
                      {'HST': ['MENTION', 0.8], 'JWST': ['SCIENCE', 0.95]}),
                     ('```json\n{\n  "title": "Damping wings in the Lyman-α forest",\n  "primary_mission_or_survey": ["X-Shooter"],\n  "other_missions_or_surveys_mentioned": ["Planck"],\n  "notes": ""\n}\n```',
                      {'title': 'Damping wings in the Lyman-α forest', 'primary_mission_or_survey': ['X-Shooter'], 'other_missions_or_surveys_mentioned': ['Planck'], 'notes': ''}),
                     ('There is no json here.', {'error': 'No JSON content found in response'}),
                     ('```json\n{"field": ["A", "B", "C",]}\n```', {'error': 'Error decoding JSON content: "Expecting value: line 1 column 26 (char 25)"'})
                     ], ids=['json1', 'json2', 'nojson', 'badjson'])
def test_extract_json(data, exp):
    """ test we can extract some json content """

    output = check_response(data)
    assert output == exp

