from bibcat.llm.run_eval import build_run_paper_evaluations, compute_run_coverage


def test_build_run_paper_evaluations_handles_missing_source_and_output(multi_run_llm_runs_data) -> None:
    source_lookup = {
        "B1": {"bibcode": "B1", "class_missions": {"HST": {"papertype": "SCIENCE"}}},
        "B2": {"bibcode": "B2", "class_missions": {"HST": {"papertype": "MENTION"}}},
    }

    evaluations = build_run_paper_evaluations(
        llm_runs_data=multi_run_llm_runs_data,
        bibcodes=["B1", "B2", "B3"],
        run_index=1,
        source_lookup=source_lookup,
    )

    assert [item.bibcode for item in evaluations] == ["B1", "B2", "B3"]
    assert evaluations[0].human_labels == {"HST": "SCIENCE"}
    assert evaluations[0].llm_predictions["HST"].papertype == "MENTION"
    assert evaluations[0].llm_predictions["HST"].confidence == [0.2, 0.8]
    assert evaluations[1].has_source
    assert not evaluations[1].has_output
    assert not evaluations[2].has_source
    assert not evaluations[2].has_output


def test_compute_run_coverage_uses_injected_source_bibcodes(multi_run_llm_runs_data) -> None:
    coverage = compute_run_coverage(
        bibcodes=["B1", "B2", "B3"],
        llm_runs_data=multi_run_llm_runs_data,
        n_runs=2,
        source_bibcodes={"B1", "B2"},
    )

    assert coverage == 0.75
