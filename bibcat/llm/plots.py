import pathlib

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.llm.metrics import missions_classes
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


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

    # read the evaluation summary output file
    eval_output = (
        pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_output_file}"
    )
    logger.info(f"reading {eval_output}")
    data = read_output(filename=eval_output)
    if missions:
        human, llm, threshold = missions_classes(missions=missions, data=data)

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
        fig.suptitle(f"Confusion Matrix\nAll missions", fontsize=14)
    else:
        fig.suptitle(f"Confusion Matrix\nMulti-missions", fontsize=14)

    plt.tight_layout()

    # Saving the figure
    cm_plot = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.cm_plot}"
    plt.savefig(cm_plot)
    logger.info(f"The confusion matrix plot is saved on {cm_plot}!")
