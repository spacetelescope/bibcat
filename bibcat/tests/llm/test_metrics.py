from pathlib import Path

import numpy as np

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
        "mission_conf": [
            {"llm_mission": "JWST", "prob_papertype": [0.8, 0.2]},
            {"llm_mission": "Roman", "prob_papertype": [0.3, 0.7]},
            {"llm_mission": "GALEX", "prob_papertype": [0.4, 0.6]},
        ],
    }
}
missions = ["HST", "JWST", "ROMAN"]

expected_data = {
    "threshold": 0.7,
    "n_bibcodes": 1,
    "n_human_mission_callouts": 3,
    "n_llm_mission_callouts": 3,
    "n_non_mast_mission_callouts": 1,
    "n_valid_mission_callouts": 2,
    "valid_missions": ["JWST", "ROMAN"],
    "non_mast_missions": ["LAMOST"],
    "human_labels": ["NONSCIENCE", "SCIENCE", "SCIENCE"],
    "llm_labels": ["NONSCIENCE", "SCIENCE", "NONSCIENCE"],
}


def test_map_papertype():
    mapped_papertype = map_papertype(data["Bibcode2024"]["human"]["TESS"])
    assert mapped_papertype == "NONSCIENCE", "wrong papertype mapping"


def test_extract_eval_data():
    metrics_data = extract_eval_data(data, missions)

    assert isinstance(metrics_data, dict), "metrics_data should be a dictionary"
    assert set(expected_data.keys()) == set(metrics_data.keys()), "keys mismatch in metrics_data"
    # assert metrics_data["threshold"] == 0.7, "threshold mismatch in metrics_data"

    for key, value in expected_data.items():
        assert metrics_data[key] == value, f"{key} mismatch"


def test_compute_and_save(tmp_path: str | Path):
    # metrics_data = {
    #     "threshold": 0.7,
    #     "n_bibcodes": 10,
    #     "n_human_mission_callouts": 21,
    #     "n_llm_mission_callouts": 25,
    #     "n_non_mast_mission_callouts": 2,
    #     "n_valid_mission_callouts": 10,
    #     "valid_missions": ["JWST", "HST"],
    #     "non_mast_missions": ["LAMOST", "EDEN"],
    #     "human_labels": ["SCIENCE", "NONSCIENCE"],
    #     "llm_labels": ["SCIENCE", "NONSCIENCE"],
    # }

    temp_output_filepath_1 = tmp_path / "output.txt"
    temp_output_filepath_2 = tmp_path / "output.json"

    compute_and_save_metrics(
        metrics_data=expected_data, output_ascii=temp_output_filepath_1, output_json=temp_output_filepath_2
    )
    assert temp_output_filepath_1.exists(), f"{temp_output_filepath_1} was not created."
    assert temp_output_filepath_1.is_file(), f"{temp_output_filepath_1} is not a file."
    assert temp_output_filepath_2.exists(), f"{temp_output_filepath_2} was not created."
    assert temp_output_filepath_2.is_file(), f"{temp_output_filepath_2} is not a file."


def test_prepare_roc_inputs():
    human_labels = ["SCIENCE", "NONSCIENCE", "NONSCIENCE"]
    llm_confidences = [[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]
    binarized_human_labels, llm_confidences_array, n_papertypes, n_verdicts = prepare_roc_inputs(
        human_labels, llm_confidences
    )

    assert np.array_equal(llm_confidences_array, np.array([[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]))
    assert np.array_equal(binarized_human_labels, np.array([[1], [0], [0]]))
    assert n_papertypes == 2
    assert n_verdicts == 3


def test_get_roc_metrics():
    llm_confidences = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
    binarized_labels = [[1], [0], [0], [1], [1]]
    n_classes = 2
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, binarized_labels, n_classes)

    assert np.array_equal(fpr, np.array([0.0, 0.0, 0.0, 1.0]), "wrong false positive rate")
    assert np.array_equal(np.round(tpr, decimals=3), np.array([0.0, 0.333, 1.0, 1.0]), "wrong true positive rate")
    assert np.array_equal(np.round(thresholds), np.array([np.inf, 1.0, 1.0, 0.0]), "wrong thresholds")

    assert roc_auc == 1, "wrong auc value"
