import logging

import pytest  # noqa: F401

from bibcat.llm.cm import build_verdict_summary_for_run
from bibcat.llm.verdict_summary import group_by_mission, summarize_verdict, summarize_verdict_from_runs

SOURCE_PAPER_WITH_MISSIONS = {
    "bibcode": "2022Sci...377.1211L",
    "title": ["Density, not radius, separates rocky and water-rich small planets orbiting M dwarf stars"],
    "abstract": "This is the abstract",
    "body": "This is the paper text of the source dataset. I am a TESS paper.",
    "class_missions": {"TESS": {"bibcode": "2022Sci...377.1211L", "papertype": "SCIENCE"}},
}
SOURCE_PAPER_WITHOUT_MISSIONS = {
    "bibcode": "2024Sci...123.3451L",
    "title": ["This paper does not call out mission"],
    "abstract": "This is the abstract",
    "body": "I am not a MAST paper",
    "class_missions": {},
}

LLM_RUN_OUTPUTS_BY_BIBCODE = {
    "2022Sci...377.1211L": [
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention JWST",
                    "quotes": ["We mention JWST."],
                },
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention TESS",
                    "quotes": ["We mention TESS."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use JWST data",
                    "quotes": ["We use JWST data."],
                },
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                }
            ],
        },
        {
            "notes": "",
            "missions": [
                {
                    "mission": "TESS",
                    "papertype": "SCIENCE",
                    "confidence": [0.9, 0.1],
                    "reason": "They use TESS data",
                    "quotes": ["We use TESS data."],
                },
                {
                    "mission": "JWST",
                    "papertype": "MENTION",
                    "confidence": [0.3, 0.7],
                    "reason": "They mention JWST",
                    "quotes": ["We mention JWST."],
                },
            ],
        },
    ],
    "2024Sci...123.3451L": [{"notes": "No mission-relevant content found.", "missions": []}],
    "2019arXiv190205569A": [{"notes": "", "missions": []}],
}


def test_evaluate_df(mocker):
    bibcode = "2022Sci...377.1211L"
    mocker.patch("bibcat.llm.verdict_summary.get_source", return_value=SOURCE_PAPER_WITH_MISSIONS)
    mocker.patch("bibcat.llm.verdict_summary.read_output", return_value=LLM_RUN_OUTPUTS_BY_BIBCODE[bibcode])

    df = summarize_verdict(bibcode, write_file=False)

    assert len(df) == 4
    assert set(df["llm_mission"]) == {"TESS", "JWST"}
    # assert the first row (JWST) is hallucinated
    assert df.iloc[0]["hallucination_by_llm"]

    # assert the second row (JWST-SCIENCE) has a low weight
    assert df.iloc[1]["llm_mission"] == "JWST"
    assert df.iloc[1]["llm_papertype"] == "SCIENCE"
    assert df.iloc[1]["mean_llm_confidences"].tolist() == [0.9, 0.1]
    assert df.iloc[1]["count"] == 1
    assert df.iloc[1]["n_runs"] == 6
    assert df.iloc[1]["weighted_confs"].tolist() == [0.15, 0.017]

    # check the last TESS-SCIENCE row
    intext = df["in_human_class"] == True  # noqa: E712
    assert len(df[intext]) == 1
    assert df[intext].iloc[0]["mission_in_text"]
    assert df[intext].iloc[0]["llm_mission"] == "TESS"
    assert df[intext].iloc[0]["llm_papertype"] == "SCIENCE"
    assert not df[intext].iloc[0]["hallucination_by_llm"]

    # assert last row has a high weight
    assert df[intext].iloc[0]["mean_llm_confidences"].tolist() == [0.9, 0.1]
    assert df[intext].iloc[0]["count"] == 5
    assert df[intext].iloc[0]["n_runs"] == 6
    assert df[intext].iloc[0]["weighted_confs"].tolist() == [0.75, 0.083]


def test_group_by_mission(mocker):
    bibcode = "2022Sci...377.1211L"
    mocker.patch("bibcat.llm.verdict_summary.get_source", return_value=SOURCE_PAPER_WITH_MISSIONS)
    mocker.patch("bibcat.llm.verdict_summary.read_output", return_value=LLM_RUN_OUTPUTS_BY_BIBCODE[bibcode])

    df = summarize_verdict(bibcode, write_file=False)
    mm = group_by_mission(df)

    # JWST
    assert mm.iloc[0]["llm_mission"] == "JWST"
    assert mm.iloc[0]["total_mission_conf"] == 0.5
    assert mm.iloc[0]["total_weighted_conf"].tolist() == [0.25, 0.25]
    assert mm.iloc[0]["prob_mission"] == 0.333
    assert mm.iloc[0]["prob_papertype"].tolist() == [0.5, 0.5]

    # TESS
    assert mm.iloc[1]["llm_mission"] == "TESS"
    assert mm.iloc[1]["total_mission_conf"] == 1.0
    assert mm.iloc[1]["total_weighted_conf"].tolist() == [0.8, 0.2]
    assert mm.iloc[1]["prob_mission"] == 0.667
    assert mm.iloc[1]["prob_papertype"].tolist() == [0.8, 0.2]


@pytest.mark.parametrize(
    "bibcode, return_source_value",
    [("2024Sci...123.3451L", SOURCE_PAPER_WITHOUT_MISSIONS), ("2019arXiv190205569A", None)],
)
def test_not_found(mocker, bibcode: str, return_source_value: dict | None):
    """test evaluate when either 'error': 'no mission output found' or 'error': 'No paper source found' in llm_output"""

    mocker.patch("bibcat.llm.verdict_summary.get_source", return_value=return_source_value)
    mocker.patch("bibcat.llm.verdict_summary.read_output", return_value=LLM_RUN_OUTPUTS_BY_BIBCODE[bibcode])

    df = summarize_verdict(bibcode, write_file=False)
    assert df is None, "Expected df to be None"


def test_build_verdict_summary_for_run_in_memory(mocker):
    llm_runs_data = {
        SOURCE_PAPER_WITH_MISSIONS["bibcode"]: LLM_RUN_OUTPUTS_BY_BIBCODE[SOURCE_PAPER_WITH_MISSIONS["bibcode"]],
        SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"]: LLM_RUN_OUTPUTS_BY_BIBCODE[SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"]],
        "2019arXiv190205569A": LLM_RUN_OUTPUTS_BY_BIBCODE["2019arXiv190205569A"],
    }
    source_papers_by_bibcode = {
        SOURCE_PAPER_WITH_MISSIONS["bibcode"]: SOURCE_PAPER_WITH_MISSIONS,
        SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"]: SOURCE_PAPER_WITHOUT_MISSIONS,
    }

    def get_source_for_bibcode(*, bibcode, **kwargs):
        return source_papers_by_bibcode.get(bibcode)

    mocker.patch("bibcat.llm.cm.get_source", side_effect=get_source_for_bibcode)
    mocker.patch("bibcat.llm.verdict_summary.identify_missions_in_text", return_value=[True])

    eval_data = build_verdict_summary_for_run(
        llm_runs_data=llm_runs_data,
        run_index=1,
        bibcodes=[
            SOURCE_PAPER_WITH_MISSIONS["bibcode"],
            SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"],
            "2019arXiv190205569A",
        ],
    )

    assert set(eval_data) == {
        SOURCE_PAPER_WITH_MISSIONS["bibcode"],
        SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"],
        "2019arXiv190205569A",
    }
    assert eval_data[SOURCE_PAPER_WITH_MISSIONS["bibcode"]]["human"] == {"TESS": "SCIENCE"}
    assert eval_data[SOURCE_PAPER_WITH_MISSIONS["bibcode"]]["llm"] == [
        {"TESS": "MENTION", "confidence": [0.3, 0.7], "mission_probability": 1.0}
    ]
    assert "No mission output found" in eval_data[SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"]]["error"]
    assert eval_data[SOURCE_PAPER_WITHOUT_MISSIONS["bibcode"]]["human"] == {}
    assert eval_data["2019arXiv190205569A"]["error"] == "No paper source found"


def test_build_verdict_summary_for_run_uses_debug_summary_level(mocker):
    mocker.patch("bibcat.llm.cm.get_source", return_value=SOURCE_PAPER_WITH_MISSIONS)
    evaluate_mock = mocker.patch(
        "bibcat.llm.cm.summarize_verdict_from_runs",
        return_value=(None, {"human": {"TESS": "SCIENCE"}, "llm": []}),
    )

    build_verdict_summary_for_run(
        llm_runs_data={
            SOURCE_PAPER_WITH_MISSIONS["bibcode"]: LLM_RUN_OUTPUTS_BY_BIBCODE[SOURCE_PAPER_WITH_MISSIONS["bibcode"]]
        },
        run_index=0,
        bibcodes=[SOURCE_PAPER_WITH_MISSIONS["bibcode"]],
    )

    assert evaluate_mock.call_args.kwargs["summary_log_level"] == logging.DEBUG


def test_summarize_verdict_from_runs_skips_to_string_and_counts_no_mission_note_in_n_runs(mocker):
    mocker.patch("bibcat.llm.verdict_summary.identify_missions_in_text", return_value=[True])
    mocker.patch("bibcat.llm.verdict_summary.logger.isEnabledFor", return_value=False)
    to_string_mock = mocker.patch("pandas.DataFrame.to_string", autospec=True)

    grouped_df, output_item = summarize_verdict_from_runs(
        SOURCE_PAPER_WITH_MISSIONS,
        [
            {
                "notes": "",
                "missions": [
                    {
                        "mission": "TESS",
                        "papertype": "SCIENCE",
                        "confidence": [0.9, 0.1],
                        "reason": "They use TESS data",
                        "quotes": ["We use TESS data."],
                    }
                ],
            },
            {"notes": "No mission-relevant content found.", "missions": []},
        ],
        summary_log_level=logging.DEBUG,
    )

    assert grouped_df is not None
    assert output_item["human"] == {"TESS": "SCIENCE"}
    assert grouped_df.iloc[0]["n_runs"] == 2
    assert grouped_df.iloc[0]["count"] == 1
    assert grouped_df.iloc[0]["weighted_confs"].tolist() == [0.45, 0.05]
    to_string_mock.assert_not_called()
