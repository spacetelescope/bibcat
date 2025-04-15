from pathlib import Path

import numpy as np

from bibcat import config
from bibcat.llm.metrics import (
    compute_and_save_metrics,
    extract_eval_data,
    get_roc_metrics,
    map_papertype,
    prepare_roc_inputs,
)

data = {
    "Bibcode2024": {
        "human": {"JWST": "SCIENCE", "ROMAN": "SCIENCE", "TESS": "SUPERMENTION"},
        "llm": [{"JWST": "SCIENCE"}, {"ROMAN": "SUPERMENTION"}, {"LAMOST": "SCIENCE"}],
        "threshold_acceptance": 0.7,
        "df": [
            {
                "llm_mission": "JWST",
                "llm_papertype": "MENTION",
                "mission_in_text": True,
            },
            {
                "llm_mission": "ROMAN",
                "llm_papertype": "MENTION",
                "mission_in_text": True,
            },
            {
                "llm_mission": "HST",
                "llm_papertype": "SCIENCE",
                "mission_in_text": False,
            },
            {
                "llm_mission": "LAMOST",
                "llm_papertype": "MENTION",
                "mission_in_text": True,
            },
        ],
        "mission_conf": [
            {"llm_mission": "JWST", "prob_papertype": [0.8, 0.2]},
            {"llm_mission": "ROMAN", "prob_papertype": [0.3, 0.7]},
            {"llm_mission": "HST", "prob_papertype": [0.55, 0.45]},
            {"llm_mission": "LAMOST", "prob_papertype": [0.4, 0.6]},
        ],
    },
    "2024Sci...377.1211L": {"error": "No mission output found for 2024Sci...377.1211L."},
}


missions = ["HST", "JWST", "ROMAN"]


sample_metrics_data = {
    "threshold": 0.7,
    "n_bibcodes": 2,
    "n_human_mission_callouts": 3,
    "n_llm_mission_callouts": 3,
    "n_non_mast_mission_callouts": 1,
    "n_human_llm_mission_callouts": 2,
    "n_human_llm_hallucination": 0,
    "n_missing_output_bibcodes": 1,
    "human_llm_missions": ["JWST", "ROMAN"],
    "non_mast_missions": ["LAMOST"],
    "human_labels": ["NONSCIENCE", "SCIENCE", "SCIENCE", "NONSCIENCE", "NONSCIENCE", "NONSCIENCE"],
    "llm_labels": ["NONSCIENCE", "SCIENCE", "NONSCIENCE", "NONSCIENCE", "NONSCIENCE", "NONSCIENCE"],
}


def test_map_papertype() -> None:
    """Test map_papertype() function"""
    mapped_papertype = map_papertype(data["Bibcode2024"]["human"]["TESS"])
    assert mapped_papertype == "NONSCIENCE", "wrong papertype mapping"


def test_extract_eval_data(mocker) -> None:
    """Test extract_eval_data function"""

    # Mock dependencies
    mock_compute_and_save_metrics = mocker.patch("bibcat.llm.metrics.compute_and_save_metrics")
    mocker.patch("bibcat.llm.metrics.logger")

    # Set mock specific config values only
    mocker.patch.object(config.paths, "output", "/mock/output")
    mocker.patch.object(config.llms.openai, "model", "gpt-4o-mini")
    mocker.patch.object(config.llms, "metrics_file", "metrics_summary")

    # Call function using fixture
    metrics_data = extract_eval_data(data, missions)
    # Expected results

    expected_metrics_data = sample_metrics_data
    assert isinstance(metrics_data, dict), "metrics_data should be a dictionary"
    assert set(metrics_data.keys()) == set(expected_metrics_data.keys()), "keys mismatch in metrics_data"

    for key, value in expected_metrics_data.items():
        assert value == metrics_data[key], f"{key} mismatch"

    # Expected file path
    expected_json_path = str(Path("/mock/output") / "llms/openai_gpt-4o-mini/metrics_summary_t0.7.json")
    expected_ascii_path = str(Path("/mock/output") / "llms/openai_gpt-4o-mini/metrics_summary_t0.7.txt")
    mock_compute_and_save_metrics.assert_any_call(metrics_data, expected_ascii_path, expected_json_path)


def test_compute_and_save_metrics(mocker) -> None:
    """Test test_compute_and_save_metrics function"""

    # Mock dependencies
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("bibcat.llm.metrics.config")
    mock_save_json_file = mocker.patch("bibcat.llm.metrics.save_json_file")
    mocker.patch("bibcat.llm.metrics.logger")

    output_ascii_filepath = "mock_output.ascii"
    output_json_filepath = "mock_output.json"

    compute_and_save_metrics(sample_metrics_data, output_ascii_filepath, output_json_filepath)

    mock_open.assert_called_with(output_ascii_filepath, "w")
    file_handle = mock_open.return_value.__enter__.return_value
    file_handle.write.assert_called()

    mock_save_json_file.assert_called_once()
    json_data = mock_save_json_file.call_args[0][1]

    assert json_data["n_bibcodes"] == 2
    assert json_data["SCIENCE"]["precision"] == 1.0


def test_prepare_roc_inputs() -> None:
    human_labels = ["SCIENCE", "NONSCIENCE", "NONSCIENCE"]
    llm_confidences = [[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]
    binarized_human_labels, llm_confidences_array, n_papertypes, n_verdicts = prepare_roc_inputs(
        human_labels, llm_confidences
    )

    assert np.array_equal(llm_confidences_array, np.array([[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]))
    assert np.array_equal(binarized_human_labels, np.array([[1], [0], [0]]))
    assert n_papertypes == 2
    assert n_verdicts == 3


def test_get_roc_metrics() -> None:
    llm_confidences = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
    binarized_labels = [[1], [0], [0], [1], [1]]
    n_classes = 2
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, binarized_labels, n_classes)

    assert np.array_equal(fpr, np.array([0.0, 0.0, 0.0, 1.0]), "wrong false positive rate")
    assert np.array_equal(np.round(tpr, decimals=3), np.array([0.0, 0.333, 1.0, 1.0]), "wrong true positive rate")
    assert np.array_equal(np.round(thresholds), np.array([np.inf, 1.0, 1.0, 0.0]), "wrong thresholds")

    assert roc_auc == 1, "wrong auc value"
