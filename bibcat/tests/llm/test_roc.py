from bibcat.llm.roc import (
    evaluate_multiple_llm_runs_with_roc,
    extract_roc_metrics_for_run,
    get_roc_metrics,
)


def test_get_roc_metrics() -> None:
    llm_confidences = [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]]
    y_true = [1, 0, 0, 1, 1]
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, y_true)

    assert len(fpr) == len(tpr)
    assert len(thresholds) == len(fpr)
    assert 0.0 <= roc_auc <= 1.0
    assert roc_auc == 1.0


def test_extract_roc_metrics_for_run(mocker, multi_run_llm_runs_data) -> None:
    source_lookup = {
        "B1": {"bibcode": "B1", "class_missions": {"HST": {"papertype": "SCIENCE"}}},
        "B2": {"bibcode": "B2", "class_missions": {"HST": {"papertype": "MENTION"}}},
    }

    roc_data = extract_roc_metrics_for_run(
        llm_runs_data=multi_run_llm_runs_data,
        missions=["HST"],
        run_index=1,
        bibcodes=["B1", "B2"],
        source_lookup=source_lookup,
    )

    assert roc_data["missions"] == ["HST"]
    assert roc_data["human_llm_missions"] == ["HST"]
    assert roc_data["n_verdicts"] == 2
    assert 0.0 <= roc_data["roc_auc"] <= 1.0


def test_evaluate_multiple_llm_runs_with_roc(
    mocker, multi_run_eval_data, multi_run_llm_runs_data, multi_run_missions
) -> None:
    source_lookup = {
        "B1": {"bibcode": "B1", "class_missions": {"HST": {"papertype": "SCIENCE"}}},
        "B2": {"bibcode": "B2", "class_missions": {"HST": {"papertype": "MENTION"}}},
    }

    summary = evaluate_multiple_llm_runs_with_roc(
        llm_runs_data=multi_run_llm_runs_data,
        missions=multi_run_missions,
        bibcodes=list(multi_run_eval_data.keys()),
        source_lookup=source_lookup,
    )

    assert summary["missions"] == ["HST"]
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
