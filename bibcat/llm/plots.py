import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.llm.metrics import extract_eval_data, extract_roc_data, get_roc_metrics, prepare_roc_inputs
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def fetch_data() -> list:
    """Read the evaluation summary output file"""
    eval_output = (
        pathlib.Path(config.paths.output)
        / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_output_file}_t{config.llms.performance.threshold}.json"
    )
    logger.info(f"reading {eval_output}")
    return read_output(filename=eval_output)


# create a confusion matrix plot
def confusion_matrix_plot(missions: list[str]) -> None:
    """Create a confusion matrix figure

    Create confusion matrix plots (counts and normalized) given a threshold value.

    Parameters
    ----------

    missions: list[str]
        list of the mission names to extract the classification labels.

    Returns
    -------

    """
    data = fetch_data()

    metrics_data = extract_eval_data(missions=missions, data=data)

    human = metrics_data["human_labels"]
    llm = metrics_data["llm_labels"]
    threshold = metrics_data["threshold"]
    valid_missions = metrics_data["valid_missions"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    papertypes = config.llms.papertypes

    # Absolute label counts
    ax[0].set_title("Count")
    ConfusionMatrixDisplay.from_predictions(human, llm, ax=ax[0], cmap=plt.cm.BuPu, colorbar=False, labels=papertypes)

    # Normalized confusion matrix
    ax[1].set_title("Normalized")
    ConfusionMatrixDisplay.from_predictions(
        human, llm, ax=ax[1], normalize="true", cmap=plt.cm.PuRd, colorbar=False, labels=papertypes
    )

    for axis in ax:
        axis.set_xlabel("LLM label")
        axis.set_ylabel("Human label")

    # Suptitle
    fig.suptitle(f"Confusion Matrix at threshold = {threshold}", fontsize=14, fontweight="bold")

    if len(valid_missions) > 13:
        fig.text(
            0.5,
            0.9,
            "More than 12 MAST Missions",
            ha="center",
            fontsize=12,
            fontstyle="italic",
            color="gray",
        )
    else:
        fig.text(
            0.5,
            0.9,
            f"Mission(s): {', '.join(valid_missions)}",
            ha="center",
            fontsize=12,
            fontstyle="italic",
            color="gray",
        )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Saving the figure
    cm_plot = (
        pathlib.Path(config.paths.output)
        / f"llms/openai_{config.llms.openai.model}/{config.llms.cm_plot}_t{config.llms.performance.threshold}.png"
    )
    plt.savefig(cm_plot)
    logger.info(f"The confusion matrix plot is saved on {cm_plot}!")


# create a ROC curve plot
def roc_plot(missions: list[str]) -> None:
    """Create a Receiver Operating Characteristic (ROC) curve plot

    Parameters
    ----------

    missions: list[str]
        list of the mission names to extract the classification labels.

    Returns
    -------

    """

    # read the evaluation summary output file
    data = fetch_data()
    human_labels, llm_confidences, valid_missions = extract_roc_data(missions=missions, data=data)

    binarized_human_labels, llm_confidences, n_papertypes, n_verdicts = prepare_roc_inputs(
        human_labels, llm_confidences
    )

    # compute ROC curve and ROC AUC (area under curve) for each class
    if n_papertypes > 2:
        fpr, tpr, thresholds, roc_auc, macro_roc_auc_ovr, micro_roc_auc_ovr = get_roc_metrics(
            llm_confidences, binarized_human_labels, n_papertypes
        )
    else:
        fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, binarized_human_labels, n_papertypes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    bbox_args = dict(boxstyle="round", fc="0.8")

    if n_papertypes > 2:
        colors = plt.cm.viridis(np.linspace(0, 1, n_papertypes))

        for i in range(n_papertypes):
            ax.plot(
                fpr[i],
                tpr[i],
                color=colors[i],
                lw=2,
                label=f"{config.llms.papertypes[i]} (AUC = {roc_auc[i]:.2f})",
            )
        ax.annotate(
            f" : macro_roc_auc_ovr = {macro_roc_auc_ovr}\n micro_roc_auc_ovr = {micro_roc_auc_ovr}",
            xy=(1, 0.35),
            xycoords="axes fraction",
            xytext=(-10, -10),
            textcoords="offset points",
            ha="right",
            va="top",
            bbox=bbox_args,
        )

    else:
        ax.plot(fpr, tpr, color="b", lw=2, label=f"SCIENCE (AUC={roc_auc:.2f})")

    # Define the target threshold values to mark
    target_thresholds = np.arange(0.1, 1.0, 0.1)

    # For each target threshold, find the index of the closest threshold in the computed array
    for p in target_thresholds:
        idx = np.abs(thresholds - p).argmin()
        ax.scatter(fpr[idx], tpr[idx], marker="o", color="r")
        ax.annotate(
            f"{thresholds[idx]:.1f}",
            (fpr[idx], tpr[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="r",
        )

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random guessing")
    ax.annotate(
        f"The number of verdicts : {n_verdicts}",
        xy=(1, 0.25),
        xycoords="axes fraction",
        xytext=(-10, -10),
        textcoords="offset points",
        ha="right",
        va="top",
        bbox=bbox_args,
    )

    # dummy scatter for the thresholds label
    ax.scatter([], [], marker="o", color="red", label="Thresholds")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)
    ax.legend(loc="lower right")

    # Suptitle
    fig.suptitle("Reciever Operating Characteristic (ROC)", fontsize=14, fontweight="bold")

    if len(valid_missions) > 13:
        fig.text(
            0.5,
            0.9,
            "More than 12 MAST Missions",
            ha="center",
            fontsize=10,
            fontstyle="italic",
            color="gray",
        )
    else:
        fig.text(
            0.5,
            0.9,
            f"Mission(s): {', '.join(valid_missions)}",
            ha="center",
            fontsize=10,
            fontstyle="italic",
            color="gray",
        )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Saving the figure
    roc = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.roc_plot}"
    plt.savefig(roc)

    logger.info(f"The roc plot is saved on {roc}!")
