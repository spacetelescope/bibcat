import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.llm.metrics import extract_eval_data, get_roc_metrics, prepare_roc_inputs
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def fetch_data():
    # read the evaluation summary output file
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

    human, llm, threshold, _ = extract_eval_data(missions=missions, data=data)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    bbox_args = dict(boxstyle="round", fc="0.8")
    labels = config.llms.classifications

    # Absolute label counts
    ax[0].set_title("Count")
    ConfusionMatrixDisplay.from_predictions(human, llm, ax=ax[0], cmap=plt.cm.BuPu, colorbar=False, labels=labels)

    # Normalized confusion matrix
    ax[1].set_title("Normalized")
    ConfusionMatrixDisplay.from_predictions(
        human, llm, ax=ax[1], normalize="true", cmap=plt.cm.PuRd, colorbar=False, labels=labels
    )

    for axis in ax:
        axis.set_xlabel("LLM label")
        axis.set_ylabel("Human label")
        axis.annotate(
            f"Threshold={threshold}",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(-10, -10),
            textcoords="offset points",
            ha="right",
            va="top",
            bbox=bbox_args,
        )
    # Suptitle
    if len(missions) < 5:
        fig.suptitle(f"Confusion Matrix\nMissions: {missions}", fontsize=14)
    elif missions == config.missions:
        fig.suptitle("Confusion Matrix\nAll missions", fontsize=14)
    else:
        fig.suptitle("Confusion Matrix\nMulti-missions", fontsize=14)

    plt.tight_layout()

    # Saving the figure
    cm_plot = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.cm_plot}"
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

    llm_confidences, binarized_human_labels, n_classes = prepare_roc_inputs(missions, data)

    # compute ROC curve and ROC AUC (area under curve) for each class
    fpr, tpr, roc_auc = get_roc_metrics(llm_confidences, binarized_human_labels, n_classes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if n_classes > 2:
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            ax.plot(
                fpr[i],
                tpr[i],
                color=colors[i],
                lw=2,
                label=f"{config.llms.classifications[i]} (AUC = {roc_auc[i]:.2f})",
            )
    else:
        ax.plot(fpr[0], tpr[0], color="b", lw=2, label=f"SCIENCE (AUC={roc_auc[0]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random guessing")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)
    ax.legend(loc="lower right")
    plt.title("Reciever Operating Characteristic (ROC)")

    # Saving the figure
    roc = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.roc_plot}"
    plt.savefig(roc)
    logger.info(f"The roc plot is saved on {roc}!")
