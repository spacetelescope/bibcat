import numpy as np

from bibcat.llm.metrics import extract_eval_data, get_roc_metrics, prepare_roc_inputs

data = {
    "Bibcode2024": {
        "human": {"JWST": "SCIENCE", "Roman": "MENTION"},
        "threshold_acceptance": 0.7,
        "llm": [{"JWST": "SCIENCE"}, {"Roman": "MENTION"}, {"GALEX": "SCIENCE"}],
        "df": [
            {"llm_mission": "JWST", "mean_llm_confidences": [0.8, 0.2]},
            {"llm_mission": "Roman", "mean_llm_confidences": [0.3, 0.7]},
            {"llm_mission": "GALEX", "mean_llm_confidences": [0.4, 0.6]},
        ],
    }
}
missions = ["HST", "JWST", "Roman"]


def test_extract_eval_data():
    human_labels, llm_labels, threshold, llm_confidences = extract_eval_data(data, missions)

    assert human_labels == ["SCIENCE", "MENTION"]
    assert llm_labels == ["SCIENCE", "MENTION"]
    assert threshold == 0.7
    assert llm_confidences == [[0.8, 0.2], [0.3, 0.7]]


def test_prepare_roc_inpuits():
    llm_confidences, binarized_human_labels, n_classes = prepare_roc_inputs(missions, data)

    assert np.array_equal(llm_confidences, np.array([[0.8, 0.2], [0.3, 0.7]]))
    assert np.array_equal(binarized_human_labels, np.array([[1], [0]]))
    assert n_classes == 2


def test_get_roc_metrics():
    llm_confidences = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
    binarized_labels = [[1], [0], [0], [1], [1]]
    n_classes = 2
    fpr, tpr, roc_auc = get_roc_metrics(llm_confidences, binarized_labels, n_classes)

    assert np.array_equal(fpr[0], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(np.round(tpr[0], decimals=3), np.array([0.0, 0.333, 1.0, 1.0]))
    assert roc_auc[0] == 1
