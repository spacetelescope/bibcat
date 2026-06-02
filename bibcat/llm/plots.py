import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def _plot_output_path(plot_name: str, metrics_type: str, threshold: float | None = None) -> pathlib.Path:
    """Build a saved plot path with run-aware naming.

    Parameters
    ----------
    plot_name : str
        Configured base plot filename.
    metrics_type : str
        Run or aggregation label such as ``single_r0`` or ``aggregate``.
    threshold : float or None, optional
        Threshold value to include in the saved name when applicable.

    Returns
    -------
    pathlib.Path
        Output path under the model-specific llm directory.
    """
    plot_path = pathlib.Path(plot_name)
    suffix = plot_path.suffix or ".png"
    stem = plot_path.stem if plot_path.suffix else plot_path.name
    threshold_suffix = f"_t{threshold}" if threshold is not None else ""
    filename = f"{stem}_{metrics_type}{threshold_suffix}{suffix}"
    return pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}" / filename


# create a confusion matrix plot
def confusion_matrix_plot(metrics_data: dict[str, Any], missions: list[str], metrics_type: str) -> None:
    """Create a confusion matrix figure

    Create confusion matrix plots (counts and normalized) from a prepared
    single-run confusion-matrix metrics data.

    Parameters
    ----------
    metrics_data: dict[str, Any]
        Single-run confusion-matrix metrics data.
    missions: list[str]
        Mission names requested by the CLI.
    metrics_type: str
        Run label used for the saved figure name.

    Returns
    -------

    """

    # capitalize all mission names just in case when is not
    missions = [mission.upper() for mission in missions]

    human = metrics_data["human_labels"]
    llm = metrics_data["llm_labels"]
    threshold = metrics_data["threshold"]
    human_llm_missions = metrics_data["human_llm_missions"]

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
    # Leave more room above the subplots
    fig.subplots_adjust(top=0.75)

    # Suptitle
    fig.suptitle(
        f"Confusion Matrix at threshold = {threshold} ({config.llms.openai.model}) ", fontsize=14, fontweight="bold"
    )

    if len(missions) == len(config.missions):
        fig.text(
            0.5,
            0.9,
            "All Missions considered",
            ha="center",
            fontsize=12,
            fontstyle="italic",
            color="gray",
        )
    else:
        fig.text(
            0.5,
            0.9,
            f"Mission(s) considered: {', '.join(missions)}",
            ha="center",
            fontsize=12,
            fontstyle="italic",
            color="gray",
        )
    fig.text(
        0.5,
        0.85,
        f"Mission(s) found: {', '.join(human_llm_missions)}",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="gray",
    )

    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Saving the figure
    cm_plot = _plot_output_path(
        plot_name=config.llms.cm_plot,
        metrics_type=metrics_type,
        threshold=config.llms.performance.threshold,
    )
    plt.savefig(cm_plot, dpi=300, bbox_inches="tight")
    logger.info(f"The confusion matrix plot is saved on {cm_plot}!")


# create a ROC curve plot
def roc_plot(roc_data: dict[str, Any], missions: list[str], metrics_type: str) -> None:
    """Create a Receiver Operating Characteristic (ROC) curve plot

    Parameters
    ----------
    roc_data: dict[str, Any]
        Single-run ROC metrics data.
    missions: list[str]
        Mission names requested by the CLI.
    metrics_type: str
        Run label used for the saved figure name.

    Returns
    -------

    """

    # capitalize all mission names just in case when is not
    missions = [mission.upper() for mission in missions]

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    thresholds = roc_data["thresholds"]
    roc_auc = roc_data["roc_auc"]
    n_verdicts = roc_data["n_verdicts"]
    human_llm_missions = roc_data["human_llm_missions"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    bbox_args = dict(boxstyle="round", fc="0.8")

    ax.plot(fpr, tpr, color="b", lw=2, label=f"SCIENCE (AUC={roc_auc:.2f})")

    # Define the target threshold values to mark
    target_thresholds = np.arange(0.1, 1.0, 0.1)
    thresholds_arr = np.array(thresholds)

    # For each target threshold, find the index of the closest threshold in the computed array
    for p in target_thresholds:
        idx = np.abs(thresholds_arr - p).argmin()
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

    if len(human_llm_missions) > 13:
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
            f"Mission(s): {', '.join(human_llm_missions)}",
            ha="center",
            fontsize=10,
            fontstyle="italic",
            color="gray",
        )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Saving the figure
    roc = _plot_output_path(
        plot_name=config.llms.roc_plot,
        metrics_type=metrics_type,
    )
    plt.savefig(roc, dpi=300, bbox_inches="tight")

    logger.info(f"The roc plot is saved on {roc}!")
