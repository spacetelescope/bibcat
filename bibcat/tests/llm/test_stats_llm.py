import pytest

from bibcat.stats.stats_llm import (
    count_mission_papertype_occurences,
    create_stats_table,
    get_threshold,
    save_evaluation_stats,
    unique_mission_papertypes,
)
from bibcat.utils.utils import load_json_file

data = {
    "Bibcode1": {
        "human": {"JWST": "SCIENCE", "Roman": "MENTION"},
        "threshold_acceptance": 0.7,
        "llm": [{"JWST": "SCIENCE"}, {"Roman": "MENTION"}, {"GALEX": "SCIENCE"}],
    },
    "Bibcode2": {
        "human": {"JWST": "SCIENCE", "HST": "MENTION"},
        "threshold_acceptance": 0.7,
        "llm": [{"JWST": "SCIENCE"}, {"HST": "SCIENCE"}],
    },
}


def test_unique_mission_papertypes():
    """Testing to find a unique set of mission-papertype pairs"""
    unique_set = unique_mission_papertypes(data, target_key="llm")
    assert unique_set == set({("GALEX", "SCIENCE"), ("HST", "SCIENCE"), ("JWST", "SCIENCE"), ("Roman", "MENTION")})


@pytest.mark.parametrize(
    "target_key, mission, papertype, expected_count",
    [
        ("llm", "JWST", "SCIENCE", 2),
        ("human", "Roman", "MENTION", 1),
        ("human", "HST", "MENTION", 1),
        ("llm", "GALEX", "SCIENCE", 1),
    ],
)
def test_count_mission_papertype_occurences(target_key, mission, papertype, expected_count):
    """Test for counting the number of occurences for a mission-papertype pair"""
    assert expected_count == count_mission_papertype_occurences(
        data=data, target_key=target_key, mission=mission, papertype=papertype
    )


def test_get_threshold():
    """Test for getting the evaluation threshold value"""
    threshold = get_threshold(data=data)
    assert threshold == 0.7


def test_create_stats_table():
    """Test for creating stats table for llm and human classifications"""
    llm_table = create_stats_table(data, "llm")
    human_table = create_stats_table(data, "human")

    assert llm_table["threshold"] == 0.7
    assert llm_table["llm_JWST_SCIENCE"] == 2
    assert llm_table["llm_GALEX_SCIENCE"] == 1
    assert human_table["human_Roman_MENTION"] == 1


@pytest.fixture
def temp_file_path(tmp_path):
    return tmp_path / "test.json"


def test_save_evaluation_stats(temp_file_path):
    """Test for saving the evaluation stats"""
    save_evaluation_stats(temp_file_path)

    assert temp_file_path.exists(), f"{temp_file_path} was not created."
    assert temp_file_path.is_file(), f"{temp_file_path} is not a file."
