from bibcat.llm.roc import (
    evaluate_multiple_llm_runs_with_roc,
    extract_roc_data,
    extract_roc_metrics_for_run,
    get_roc_metrics,
    prepare_roc_inputs,
)


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


def test_extract_roc_metrics_for_run(
    mocker, single_run_eval_data, single_run_missions, multi_run_llm_runs_data
) -> None:
    build_mock = mocker.patch(
        "bibcat.llm.roc.build_eval_data_for_run",
        return_value=single_run_eval_data,
    )

    roc_data = extract_roc_metrics_for_run(
        llm_runs_data=multi_run_llm_runs_data,
        missions=single_run_missions,
        run_index=1,
        bibcodes=["Bibcode2024"],
    )

    assert roc_data["missions"] == single_run_missions
    assert roc_data["n_verdicts"] == 6
    assert 0.0 <= roc_data["roc_auc"] <= 1.0
    assert build_mock.call_args.kwargs["run_index"] == 1
    assert build_mock.call_args.kwargs["bibcodes"] == ["Bibcode2024"]


def test_evaluate_multiple_llm_runs_with_roc(
    mocker, multi_run_eval_data, multi_run_llm_runs_data, multi_run_missions
) -> None:
    per_run_eval_data = [
        {
            "B1": {
                "human": {"HST": "SCIENCE"},
                "llm": [{"HST": "SCIENCE"}],
                "mission_conf": [{"llm_mission": "HST", "prob_papertype": [0.9, 0.1]}],
            },
            "B2": {
                "human": {"HST": "MENTION"},
                "llm": [{"HST": "MENTION"}],
                "mission_conf": [{"llm_mission": "HST", "prob_papertype": [0.1, 0.9]}],
            },
        },
        {
            "B1": {
                "human": {"HST": "SCIENCE"},
                "llm": [{"HST": "MENTION"}],
                "mission_conf": [{"llm_mission": "HST", "prob_papertype": [0.2, 0.8]}],
            },
            "B2": {
                "error": "No mission output found for B2.",
                "human": {"HST": "MENTION"},
            },
        },
    ]
    mocker.patch("bibcat.llm.roc.build_eval_data_for_run", side_effect=per_run_eval_data)
    mocker.patch(
        "bibcat.llm.metrics.load_source_dataset",
        return_value=[{"bibcode": "B1"}, {"bibcode": "B2"}],
    )

    summary = evaluate_multiple_llm_runs_with_roc(
        llm_runs_data=multi_run_llm_runs_data,
        missions=multi_run_missions,
        bibcodes=list(multi_run_eval_data.keys()),
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
