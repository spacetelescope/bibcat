from bibcat import config
from bibcat.llm.metrics import evaluate_multiple_llm_runs, extract_eval_data, extract_eval_data_for_run, map_papertype


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


def test_extract_eval_data_for_run(mocker, single_run_eval_data, single_run_missions, multi_run_llm_runs_data) -> None:
    build_mock = mocker.patch(
        "bibcat.llm.metrics.build_eval_data_for_run",
        return_value=single_run_eval_data,
    )

    metrics_data = extract_eval_data_for_run(
        llm_runs_data=multi_run_llm_runs_data,
        missions=single_run_missions,
        run_index=1,
        bibcodes=["Bibcode2024"],
    )

    assert metrics_data == extract_eval_data(single_run_eval_data, single_run_missions)
    assert build_mock.call_args.kwargs["run_index"] == 1
    assert build_mock.call_args.kwargs["bibcodes"] == ["Bibcode2024"]


def test_evaluate_multiple_llm_runs(mocker, multi_run_eval_data, multi_run_llm_runs_data, multi_run_missions) -> None:
    per_run_eval_data = [
        {
            "B1": {"human": {"HST": "SCIENCE"}, "llm": [{"HST": "SCIENCE"}]},
            "B2": {"human": {"HST": "MENTION"}, "llm": [{"HST": "MENTION"}]},
        },
        {
            "B1": {"human": {"HST": "SCIENCE"}, "llm": [{"HST": "MENTION"}]},
            "B2": {"error": "No mission output found for B2.", "human": {"HST": "MENTION"}},
        },
    ]

    mocker.patch("bibcat.llm.metrics.build_eval_data_for_run", side_effect=per_run_eval_data)
    mocker.patch(
        "bibcat.llm.metrics.load_source_dataset",
        return_value=[{"bibcode": "B1"}, {"bibcode": "B2"}],
    )

    summary = evaluate_multiple_llm_runs(
        llm_runs_data=multi_run_llm_runs_data,
        missions=multi_run_missions,
        bibcodes=list(multi_run_eval_data.keys()),
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
