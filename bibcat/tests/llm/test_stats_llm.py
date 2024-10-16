import pytest

from bibcat.stats.stats_llm import (
    count_mission_papertype_occurences,
    create_stats_table,
    get_threshold,
    save_evaluation_stats,
    save_operation_stats,
    unique_mission_papertypes,
)
from bibcat.utils.utils import load_json_file, save_json_file

eval_data: dict = {
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

ops_data: dict = {"2020A&A...642A.105K": [{"KEPLER": ["SCIENCE", [0.95, 0.05]], "K2": ["MENTION", [0.2, 0.8]]}]}


def test_unique_mission_papertypes():
    """Testing to find a unique set of mission-papertype pairs"""
    unique_set = unique_mission_papertypes(eval_data, target_key="llm")
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
        data=eval_data, target_key=target_key, mission=mission, papertype=papertype
    )


def test_get_threshold():
    """Test for getting the evaluation threshold value"""
    threshold = get_threshold(data=eval_data)
    assert threshold == 0.7


def test_create_stats_table():
    """Test for creating stats table for llm and human classifications"""
    llm_table = create_stats_table(eval_data, "llm")
    human_table = create_stats_table(eval_data, "human")

    assert llm_table["threshold"] == 0.7
    assert llm_table["llm_jwst_science"] == 2
    assert llm_table["llm_galex_science"] == 1
    assert human_table["human_roman_mention"] == 1


@pytest.fixture
def temp_input_filepath(tmp_path):
    return tmp_path / "input.json"


@pytest.fixture
def temp_output_filepath(tmp_path):
    return tmp_path / "output.json"


def test_save_evaluation_stats(temp_input_filepath, temp_output_filepath):
    """Test for saving the evaluation stats"""

    save_json_file(temp_input_filepath, eval_data)
    save_evaluation_stats(temp_input_filepath, temp_output_filepath)

    assert temp_output_filepath.exists(), f"{temp_output_filepath} was not created."
    assert temp_output_filepath.is_file(), f"{temp_output_filepath} is not a file."


def test_save_operation_stats(temp_input_filepath, temp_output_filepath):
    """Test for saving the evaluation stats"""
    save_json_file(temp_input_filepath, ops_data)
    save_operation_stats(temp_input_filepath, temp_output_filepath)

    stats_table = load_json_file(temp_output_filepath)
    assert len(stats_table) == 2, f"the length of stats table is wrong"
    assert stats_table[0]["mission"] == "K2"
    assert stats_table[1]["total_count"] == 1
    assert stats_table[1]["accepted_bibcodes"] == ["2020A&A...642A.105K"]

    assert temp_output_filepath.exists(), f"{temp_output_filepath} was not created."
    assert temp_output_filepath.is_file(), f"{temp_output_filepath} is not a file."
