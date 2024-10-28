import pathlib
from typing import Any, Callable, Dict, List

import pytest

from bibcat import config
from bibcat.llm.stats import save_evaluation_stats, save_operation_stats
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import load_json_file, save_json_file

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


eval_data: Dict[str, Any] = {
    "2020A&A...642A.105K": {
        "human": {"KEPLER": "SCIENCE"},
        "llm": [{"KEPLER": "SCIENCE"}, {"K2": "MENTION"}],
        "df": [
            {
                "llm_mission": "KEPLER",
                "llm_papertype": "SCIENCE",
                "mean_llm_confidences": [0.95, 0.05],
                "consistency": 100.0,
                "in_human_class": True,
                "mission_in_text": True,
            },
            {
                "llm_mission": "K2",
                "llm_papertype": "MENTION",
                "mean_llm_confidences": [0.2, 0.8],
                "consistency": 0.0,
                "in_human_class": False,
                "mission_in_text": False,
            },
        ],
    }
}

# OPS paper_output.json data
ops_data: Dict[str, Any] = {
    "2020A&A...642A.105K": [{"KEPLER": ["SCIENCE", [0.95, 0.05]], "K2": ["MENTION", [0.2, 0.8]]}]
}

expected_stats_data: List[Dict[str, Any]] = [
    {"threshold_acceptance": 0.85, "threshold_inspection": 0.5},
    {
        "mission": "k2",
        "papertype": "mention",
        "total_count": 1,
        "accepted_count": 0,
        "accepted_bibcodes": [],
        "inspection_count": 1,
        "inspection_bibcodes": ["2020A&A...642A.105K"],
    },
    {
        "mission": "kepler",
        "papertype": "science",
        "total_count": 1,
        "accepted_count": 1,
        "accepted_bibcodes": ["2020A&A...642A.105K"],
        "inspection_count": 0,
        "inspection_bibcodes": [],
    },
]

SaveStats = Callable[[pathlib.Path, pathlib.Path, float, float], None]


@pytest.mark.parametrize(
    "input_data, save_stats, test_name",
    [(eval_data, save_evaluation_stats, "evaluation"), (ops_data, save_operation_stats, "operation")],
)
def test_save_stats(tmp_path: str | pathlib.Path, input_data: Dict[str, Any], save_stats: SaveStats, test_name: str):
    """Test for saving the evaluation/operation results stats

    Parameters
    ----------
    tmp_path: str | pathlib.Path
        Temporary path for files
    input_data: Dict[str, Any]
        Input data, either summary_output data or paper_output data
    save_stats: SaveStats
        Save statistics function, either `save_evaluation_stats()` or `save_operation_stats()`
    test_name: str
        Test name

    Returns
    -------
    None

    """

    temp_input_filepath = tmp_path / "input.json"
    temp_output_filepath = tmp_path / "output.json"

    # Debugging: Print input data
    logger.info(f"Testing {test_name} stats with input_data: {input_data}")

    save_json_file(temp_input_filepath, input_data)
    save_stats(temp_input_filepath, temp_output_filepath, threshold_acceptance=0.85, threshold_inspection=0.5)

    assert temp_output_filepath.exists(), f"{temp_output_filepath} was not created for {test_name}."
    assert temp_output_filepath.is_file(), f"{temp_output_filepath} is not a file for {test_name}."

    # Load the output and validate
    stats_table: Dict[str, Any] = load_json_file(temp_output_filepath)
    assert len(stats_table) == len(expected_stats_data), f"the length of stats table is wrong for {test_name}."

    # Key assertions: loop over the values
    for index, (expected, actual) in enumerate(zip(expected_stats_data, stats_table)):
        assert expected == actual, f"Mismatch in {test_name} stats at index {index}: expected {expected} got {actual}"
