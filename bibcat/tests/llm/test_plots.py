import matplotlib.pyplot as plt

from bibcat.llm.plots import confusion_matrix_plot, roc_plot


def test_confusion_matrix_plot_uses_prepared_metrics_payload(mocker) -> None:
    from_predictions = mocker.patch("bibcat.llm.plots.ConfusionMatrixDisplay.from_predictions")
    savefig = mocker.patch("bibcat.llm.plots.plt.savefig")

    confusion_matrix_plot(
        metrics_data={
            "human_labels": ["SCIENCE", "NONSCIENCE"],
            "llm_labels": ["SCIENCE", "NONSCIENCE"],
            "threshold": 0.5,
            "human_llm_missions": ["JWST"],
        },
        missions=["jwst"],
        metrics_type="single_r3",
    )

    assert from_predictions.call_count == 2
    savefig.assert_called_once()
    assert savefig.call_args.args[0].name.endswith("_single_r3_t0.5.png")
    plt.close("all")


def test_roc_plot_uses_prepared_roc_payload(mocker) -> None:
    savefig = mocker.patch("bibcat.llm.plots.plt.savefig")

    roc_plot(
        roc_data={
            "fpr": [0.0, 0.0, 1.0],
            "tpr": [0.0, 1.0, 1.0],
            "thresholds": [1.9, 0.9, 0.1],
            "roc_auc": 1.0,
            "n_verdicts": 2,
            "human_llm_missions": ["JWST"],
        },
        missions=["jwst"],
        metrics_type="single_r4",
    )

    savefig.assert_called_once()
    assert savefig.call_args.args[0].name.endswith("_single_r4.png")
    plt.close("all")
