from bibcat import config
from bibcat.llm.metrics import evaluate_multiple_llm_runs, extract_eval_data, map_papertype


def test_map_papertype(single_run_eval_data) -> None:
    mapped_papertype = map_papertype(single_run_eval_data["Bibcode2024"]["human"]["TESS"])
    assert mapped_papertype == "NONSCIENCE"


def test_extract_eval_data_single_run(mocker, single_run_eval_data, single_run_missions) -> None:
    mocker.patch("bibcat.llm.metrics.logger")
    mocker.patch.object(config.llms.performance, "threshold", 0.7)

    metrics_data = extract_eval_data(single_run_eval_data, single_run_missions)

    assert metrics_data["threshold"] == 0.7
    assert metrics_data["n_bibcodes"] == 3
    assert metrics_data["n_human_callouts"] == 3
    assert metrics_data["n_llm_callouts"] == 2
    assert metrics_data["n_missing_paper_sources"] == 1
    assert metrics_data["n_missing_output_bibcodes"] == 1
    assert metrics_data["human_llm_missions"] == ["JWST", "ROMAN"]
    assert metrics_data["n_human_llm_mission_callouts"] == 2
    assert metrics_data["n_human_llm_hallucination"] == 0

    assert metrics_data["human_labels"] == ["NONSCIENCE", "SCIENCE", "SCIENCE", "SCIENCE", "NONSCIENCE", "NONSCIENCE"]
    assert metrics_data["llm_labels"] == [
        "NONSCIENCE",
        "SCIENCE",
        "NONSCIENCE",
        "NONSCIENCE",
        "NONSCIENCE",
        "NONSCIENCE",
    ]

    assert len(metrics_data["tn_bibcodes"]) == 3
    assert len(metrics_data["tp_bibcodes"]) == 1
    assert len(metrics_data["fn_bibcodes"]) == 2
    assert len(metrics_data["fp_bibcodes"]) == 0

    assert metrics_data["metrics"]["tn"] == 3
    assert metrics_data["metrics"]["tp"] == 1
    assert metrics_data["metrics"]["fn"] == 2
    assert metrics_data["metrics"]["fp"] == 0
    assert metrics_data["metrics"]["precision"] == 1.0
    assert metrics_data["metrics"]["recall"] == 1 / 3
    assert metrics_data["metrics"]["f1"] == 0.5
    assert metrics_data["metrics"]["accuracy"] == 4 / 6


def test_evaluate_multiple_llm_runs(multi_run_eval_data, multi_run_llm_runs_data, multi_run_missions) -> None:
    summary = evaluate_multiple_llm_runs(
        eval_data=multi_run_eval_data,
        llm_runs_data=multi_run_llm_runs_data,
        missions=multi_run_missions,
    )

    assert summary["n_runs"] == 2
    assert summary["run_coverage"] == 0.75
    assert len(summary["per_run_metrics"]) == 2
    assert summary["per_run_metrics"][0]["accuracy"] == 1.0
    assert summary["per_run_metrics"][1]["accuracy"] == 0.5

    aggregate = summary["aggregate_metrics"]
    assert aggregate["accuracy"]["mean"] == 0.75
    assert aggregate["accuracy"]["std"] == 0.25
    assert aggregate["precision"]["mean"] == 0.5
    assert aggregate["recall"]["mean"] == 0.5
    assert aggregate["recall"]["mean"] == 0.5
    assert aggregate["recall"]["mean"] == 0.5
