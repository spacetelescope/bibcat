from bibcat.llm.roc import evaluate_multiple_llm_runs_with_roc, extract_roc_data, get_roc_metrics, prepare_roc_inputs


def test_extract_roc_data(single_run_eval_data, single_run_missions) -> None:
    human_labels, llm_confidences, human_llm_missions = extract_roc_data(single_run_eval_data, single_run_missions)

    assert human_labels == ["NONSCIENCE", "SCIENCE", "SCIENCE", "SCIENCE", "NONSCIENCE", "NONSCIENCE"]
    assert llm_confidences == [[0.55, 0.45], [0.8, 0.2], [0.3, 0.7], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert human_llm_missions == ["JWST", "ROMAN"]


def test_prepare_roc_inputs() -> None:
    human_labels = ["SCIENCE", "NONSCIENCE", "NONSCIENCE"]
    llm_confidences = [[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]
    y_true, confidence_vectors, n_samples = prepare_roc_inputs(human_labels, llm_confidences)

    assert y_true == [1, 0, 0]
    assert confidence_vectors == [[0.8, 0.2], [0.3, 0.7], [0.25, 0.75]]
    assert n_samples == 3


def test_get_roc_metrics() -> None:
    llm_confidences = [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]]
    y_true = [1, 0, 0, 1, 1]
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, y_true)

    assert len(fpr) == len(tpr)
    assert len(thresholds) == len(fpr)
    assert 0.0 <= roc_auc <= 1.0
    assert roc_auc == 1.0


def test_evaluate_multiple_llm_runs_with_roc(multi_run_eval_data, multi_run_llm_runs_data, multi_run_missions) -> None:
    summary = evaluate_multiple_llm_runs_with_roc(
        eval_data=multi_run_eval_data,
        llm_runs_data=multi_run_llm_runs_data,
        missions=multi_run_missions,
    )

    assert summary["n_runs"] == 2
    assert len(summary["per_run_roc"]) == 2
    assert summary["aggregate_auc"]["mean"] == 1.0
    assert summary["aggregate_auc"]["std"] == 0.0

    for run_roc in summary["per_run_roc"]:
        assert set(run_roc.keys()) == {"run_index", "fpr", "tpr", "thresholds", "roc_auc"}
        assert isinstance(run_roc["fpr"], list)
        assert isinstance(run_roc["tpr"], list)
        assert isinstance(run_roc["thresholds"], list)
        assert run_roc["roc_auc"] == 1.0
