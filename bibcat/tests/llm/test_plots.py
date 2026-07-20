import matplotlib.pyplot as plt

from bibcat.llm.plots import cm_plot, roc_plot


def test_cm_plot_uses_prepared_metrics_data(mocker) -> None:
    from_predictions = mocker.patch("bibcat.llm.plots.ConfusionMatrixDisplay.from_predictions")
    savefig = mocker.patch("bibcat.llm.plots.plt.savefig")

    cm_plot(
        metrics_data={
            "human_labels": ["SCIENCE", "NONSCIENCE"],
            "llm_labels": ["SCIENCE", "NONSCIENCE"],
            "threshold": 0.5,
            "missions": ["JWST"],
            "human_llm_missions": ["JWST"],
        },
        metrics_type="single_r3",
    )

    assert from_predictions.call_count == 2
    savefig.assert_called_once()
    assert savefig.call_args.args[0].name.endswith("_single_r3_t0.5.png")
    plt.close("all")


def test_roc_plot_uses_prepared_roc_data(mocker) -> None:
    savefig = mocker.patch("bibcat.llm.plots.plt.savefig")

    roc_plot(
        metrics_data={
            "fpr": [0.0, 0.0, 1.0],
            "tpr": [0.0, 1.0, 1.0],
            "thresholds": [1.9, 0.9, 0.1],
            "roc_auc": 1.0,
            "n_verdicts": 2,
            "missions": ["JWST"],
            "human_llm_missions": ["JWST"],
        },
        metrics_type="single_r4",
    )

    savefig.assert_called_once()
    assert savefig.call_args.args[0].name.endswith("_single_r4.png")
    plt.close("all")


def test_cm_plot_requires_missions() -> None:
    try:
        cm_plot(
            metrics_data={
                "human_labels": ["SCIENCE"],
                "llm_labels": ["SCIENCE"],
                "threshold": 0.5,
                "human_llm_missions": ["JWST"],
            },
            metrics_type="single_r0",
        )
    except ValueError as error:
        assert "missing required field 'missions'" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing missions field")


def test_roc_plot_requires_missions() -> None:
    try:
        roc_plot(
            metrics_data={
                "fpr": [0.0, 1.0],
                "tpr": [0.0, 1.0],
                "thresholds": [1.0, 0.0],
                "roc_auc": 0.5,
                "n_verdicts": 2,
                "human_llm_missions": ["JWST"],
            },
            metrics_type="single_r0",
        )
    except ValueError as error:
        assert "missing required field 'missions'" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing missions field")
