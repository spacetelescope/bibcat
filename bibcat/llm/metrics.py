from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def map_papertype(papertype: str):
    """Map human classified papertype to allowed papertypes, for instance, if `papertype` is "SUPERMENTION", it will returns "MENTION".

    Parameters
    ==========
    papertype: str, uppercase
        human classified papertype, e.g., "SCIENCE", "DATA_INFLUECED"

    Returns
    =======
    mapped_papertype: str, uppercase
        the mapped papertype follwing `config.map_papertypes`, e.g., "MENTION" if `papertype` is "SUPERMETION"
    """
    logger.debug(f"map_papertype(): human classified papertype to map = '{papertype}'")
    try:
        if papertype.lower() in config.llms.map_papertypes:
            mapped_value = config.llms.map_papertypes.get(papertype.lower())
            if mapped_value.upper() in config.llms.papertypes:
                mapped_papertype = mapped_value.upper()
                logger.debug(f"map_papertype(): mapped papertype is '{mapped_papertype}'")
                return mapped_papertype
            else:
                raise ValueError(
                    f"The mapped papertype '{mapped_value}' for human papertype '{papertype}' is not a valid classification."
                )
        else:
            raise KeyError(f"The human papertype '{papertype}' is an invalid papertype.")
    except KeyError as ke:
        logger.error(f"KeyError encountered: {ke}", exc_info=True)
    except ValueError as ve:
        logger.error(f"ValeError encountered: {ve}", exc_info=True)


def extract_eval_data(data: dict, missions: list[str], is_cm: bool = False):
    """Extract the human and llm classification labels and confidences

    Extract the human and llm classes and confidence values from the evaluation json file,
    `config.llms.eval_output_file (summary_output.json)`. You can extract data from only a single
    mission or a list of missions. The labels will be used to create confusion matrix plots and the
    confidence values are used for ROC curves.

    Parameters
    ----------
    data : dict
        the dict of the evaluation data of `config.llms.eval_output_file (summary_output.json)`
    missions: list[str]
        list of the mission names to extract the classification labels.
    is_cm: bool, default = False
        Whether to compute and save the metrics summary when plotting confusion matrix

    Returns
    -------
    tuple
        A tuple of the list of human labels, llm labels, and the hreshold value for verdict acceptance.
    human_labels: list[str]
        True labels by human, a list of papertypes, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.papertypes`
    llm_labels: list[str]
        Predicted labels by llm, a list of papertypes, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.papertypes`
    threshold: float
        The threshold value used to determine if the LLM papertype is accepted.
        If the papertype's confidence is greater than or equal to this threshold,
        the mission along with the papertype is recorded as accepted.
    llm_confidences: list[list[float]]
        A list of confidence score sets for all verdicts.
        For example: [[0.9, 0.1], [0.4, 0.6]], where the first set corresponds to 'SCIENCE' and the second to 'MENTION'.
    valid_missions: set[str]
        A set of missions, each containing both human- and LLM-classified paper types, used for evaluation plots.


    """

    n_bibcodes = len(data)
    logger.info(f"The number of evaluation summary data, e.g., the number of bibcodes = {n_bibcodes}")
    logger.info(f"{len(missions)} mission(s): {', '.join(missions)} is/are evaluated! ")

    human_labels = []  # for Confusion matrix and ROC
    llm_labels = []  # for Confusion matrix
    llm_confidences = []  # for ROC
    valid_mission_callouts = []  # missions that have both human and llm classified papertypes

    # counting mission callouts by human
    n_human_mission_callouts = 0
    # counting mission callouts by llm, and matched between human and llm
    n_llm_mission_callouts = 0

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        n_human_mission_callouts += len(human_data)

        llm_data = item["llm"]
        llm_mission = {k for i in llm_data for k in i.keys()}
        n_llm_mission_callouts += len(llm_mission)

        # llm missions for ROC
        llm_mission_conf = item["mission_conf"]

        # extracting human labels and llm labels
        for mission in missions:
            if mission in human_data and mission in llm_mission:
                logger.info(f"Checking {mission} summary output")
                valid_mission_callouts.append(mission)

                # human labels needed for confusion matrix and ROC
                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                # map papertype to allowed papertype
                mapped_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype = '{mapped_papertype}'")
                human_labels.append(mapped_papertype)

                # llm labels = final llm papertypes of missions in "llm: []" for ROC
                labels = [v for i in llm_data for k, v in i.items() if k == mission]
                llm_labels.extend(labels)

                # To generate an ROC curve, we need the full range of confidence values. Use "prob_papertype" for each mission, as "mean_llm_confidences" only reflect the scores of the finally accepted papertypes in "llm:[]", which are always above the threshold. We require the varying values provided by "prob_papertype where human labels exist."
                confs = [i["prob_papertype"] for i in llm_mission_conf if i["llm_mission"] == mission]
                llm_confidences.extend(confs)

        threshold = data[bibcode]["threshold_acceptance"]
        valid_missions = sorted(list(set(valid_mission_callouts)))

    logger.debug(f"threshold = {threshold}")
    logger.debug(f"human_labels = {human_labels}")
    logger.debug(f"llm_labels = {llm_labels}")
    logger.debug(f"llm_confidences = {llm_confidences}")
    logger.info(
        f"The total numbers of mission callouts by human and llm are {n_human_mission_callouts} and {n_llm_mission_callouts} respectively. \n Among these callouts, only {len(valid_mission_callouts)} cases are called out by both llm and human and valid for further evaluations!"
    )

    if is_cm:
        # compute accuracy, f1 score, precision score, and recall score
        compute_and_save_metrics(
            n_bibcodes,
            n_human_mission_callouts,
            n_llm_mission_callouts,
            len(valid_missions),
            valid_missions,
            human_labels,
            llm_labels,
            output_file=Path(config.paths.output)
            / f"llms/openai_{config.llms.openai.model}/metrics_summary_t{config.llms.performance.threshold}.txt",
        )

    return (
        human_labels,
        llm_labels,
        threshold,
        llm_confidences,
        valid_missions,
    )


def compute_and_save_metrics(
    n_bibcodes: int,
    n_human_mission_callouts: int,
    n_llm_mission_callouts: int,
    n_valid_missions_callouts: int,
    valid_missions: list[str],
    human_labels: list[str],
    llm_labels: list[str],
    output_file: str = "metrics_summary.txt",
):
    """Compute evaluation metrics (accuracy, f1, precision, and recall scores) and save results to an ascii file

    Parameters
    ==========
    n_bibcodes: int
        The number of bibcodes (papers)
    n_human_mission_callouts: int
        The number of mission callouts by human classification
    n_llm_mission_callouts: int
        The number of mission callouts by llm classification
    n_valid_missions_callouts: int
        The number of mission callouts by both human and llm
    valid_missions: list[str]
        The missions called out by both human and llm
    human_labels: list[str]
        True labels, human classified labels like ["SCIENCE", "MENTION"]
    llm_labels: list[str]
        Predicted labels by llm

    Return
    ======
    None

    """
    # Encode string labels into numeric values using LabelEncoder
    label_encoder = LabelEncoder()
    human_labels_encoded = label_encoder.fit_transform(human_labels)
    llm_labels_encoded = label_encoder.transform(llm_labels)

    # Determine number of classes
    n_classes = len(label_encoder.classes_)

    # Compute metrics (single computation for all cases)
    accuracy = accuracy_score(human_labels_encoded, llm_labels_encoded)

    precision_macro = precision_score(human_labels_encoded, llm_labels_encoded, average="macro")
    recall_macro = recall_score(human_labels_encoded, llm_labels_encoded, average="macro")
    f1_macro = f1_score(human_labels_encoded, llm_labels_encoded, average="macro")

    precision_micro = precision_score(human_labels_encoded, llm_labels_encoded, average="micro")
    recall_micro = recall_score(human_labels_encoded, llm_labels_encoded, average="micro")
    f1_micro = f1_score(human_labels_encoded, llm_labels_encoded, average="micro")

    # Compute per-class precision, recall, and F1-score
    precision_per_class = precision_score(human_labels_encoded, llm_labels_encoded, average=None)
    recall_per_class = recall_score(human_labels_encoded, llm_labels_encoded, average=None)
    f1_per_class = f1_score(human_labels_encoded, llm_labels_encoded, average=None)

    # Write results to an ASCII file
    with open(output_file, "w") as f:
        f.write(f"The number of bibcodes (papers) for evaluation metrics: {n_bibcodes}\n")
        f.write(f"The number of mission callouts by human: {n_human_mission_callouts}\n")
        f.write(f"The number of mission callouts by llm: {n_llm_mission_callouts}\n")
        f.write(f"The number of mission callouts by both human and llm: {n_valid_missions_callouts}\n\n")

        f.write(f"Missions called out by both human and llm: {', '.join(valid_missions)}\n")
        f.write(f"{n_classes} papertypes: {', '.join(label_encoder.classes_)}\n\n")

        f.write(f"Accuracy Score: {accuracy:.4f}\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}\n")
        f.write(f"Recall (Macro): {recall_macro:.4f}\n")
        f.write(f"F1-score (Macro): {f1_macro:.4f}\n\n")
        f.write(f"Precision (Micro): {precision_micro:.4f}\n")
        f.write(f"Recall (Micro): {recall_micro:.4f}\n")
        f.write(f"F1-score (Micro): {f1_micro:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        for i, class_label in enumerate(label_encoder.classes_):
            f.write(f"  - {class_label}:\n")
            f.write(f"    Precision: {precision_per_class[i]:.4f}\n")
            f.write(f"    Recall: {recall_per_class[i]:.4f}\n")
            f.write(f"    F1-score: {f1_per_class[i]:.4f}\n")

    logger.info(f"Metrics saved to {output_file}")


# def prepare_roc_inputs(data: dict, missions: list[str]):
def prepare_roc_inputs(human_labels: list[str], llm_confidences: list[list[float]]):
    """Prepare input data for ROC and AUC (area under curve)

    Parameters
    ----------
    human_labels: list[str]
        True labels by human, a list papertypes, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.papertypes`
    llm_confidences: list[list[float]]
        Predicted labels by llm, a list of confidence score pairs for all verdicts.


    Returns
    -------
    tuple
        A tuple of confidences, binarized_human_labels, and n_classes.
    binarized_human_labels: NDArray[np.int64]
        Array-like of shape (n_samples,) if the binary case or (n_samples, n_classes) if the multi-class case. Binarized human labels as ROC input, e.g.,[[0][1]..],[[0 1 0 0][0 1 0 0]...]
    llm_confidences: NDArray[np.float64]
        Array-like of shape (n_samples,) if the binary case or (n_samples, n_classes) if the multi-class case. A list of confidence score pairs for all verdicts. Each inner list contains two floats:
        the first for "SCIENCE" and the second for "MENTION". For example: [[0.9 0.1] [0.4 0.6]]
    n_papertype: int
        the number of available papertypes
    n_verdicts: int
        the number of MAST mission papertype verdicts by LLM

    """

    logger.debug(f"human_labels before binarization:{human_labels}")
    logger.debug(f"llm_confidences before binarization:{llm_confidences}")

    # prep data for the roc plot
    lb = LabelBinarizer()
    binarized_human_labels = lb.fit_transform(human_labels)
    logger.debug(f"binarized_human_labels={binarized_human_labels}")

    llm_confidences = np.array(llm_confidences)
    logger.debug(f"llm_confidences ={llm_confidences}")

    n_papertype = len(set(human_labels))
    n_verdicts = len(human_labels)
    logger.info(f"The number of verdicts for ROC = {n_verdicts}")
    return binarized_human_labels, llm_confidences, n_papertype, n_verdicts


def get_roc_metrics(llm_confidences: NDArray[np.float64], binarized_human_labels: NDArray[np.int64], n_papertype: int):
    """Compute ROC curve and ROC AUC (area under curve)

    Parameters
    ----------
    llm_confidences : array-like of shape (n_samples,) if the binary case or (n_samples, n_classes) if the multi-class case
        the numpy array of llm_confidences
    binarized_true_labels: array-like of shape (n_samples,) if the binary case or (n_samples, n_classes) if the multi-class case
        binarized_human_labels, e.g., [[0] [1] [1] [0] [0]] if the binary

    Returns
    -------
    tuple
        a tuple of false positive rate(fpr), true positive rate(tpr), and roc_auc
    fpr: float
        false positive rate
    tpr: float
        true positive rate
    roc_auc: float
        ROC area under curve
    macro_roc_auc_ovr: float
        Macro-averaged One-vs-Rest ROC AUC score for the multiclass case (only when n_papertypes > 2)
    micro_roc_auc_ovr: float
        Micro-averaged One-vs-Rest ROC AUC score for the multiclass case (only when n_papertypes > 2)


    """

    # compute ROC curve and ROC AUC (area under curve) for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    logger.info("Creating ROC curve ouptput")

    try:
        if n_papertype > 2:
            for i in range(n_papertype):
                logger.debug(f"human_labels = {binarized_human_labels}, confidences = {llm_confidences[:, i]}")
                fpr[i], tpr[i], _ = roc_curve(binarized_human_labels[:, i], llm_confidences[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            macro_roc_auc_ovr = roc_auc_score(
                binarized_human_labels, llm_confidences, multi_class="ovr", average="macro"
            )
            micro_roc_auc_ovr = roc_auc_score(
                binarized_human_labels, llm_confidences, multi_class="ovr", average="micro"
            )
            logger.info(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")
            logger.info(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")
            return fpr, tpr, roc_auc, macro_roc_auc_ovr, micro_roc_auc_ovr

        elif n_papertype == 2:
            fpr, tpr, _ = roc_curve(binarized_human_labels, llm_confidences[:, 0])
            roc_auc = auc(fpr, tpr)
            return fpr, tpr, roc_auc

        else:
            raise ValueError(
                f"'n_papertype' ={n_papertype} is invalid. The number of papertypes should be larger than or equal to 2."
            )
    except ValueError as ve:
        logger.error(f"ValeError encountered: {ve}", exc_info=True)

    logger.info(f"fpr={fpr}")
    logger.info(f"tpr={tpr}")
    logger.info(f"auc ={roc_auc}")


# compute metrics and save results
def compute_and_save_metrics(y_true, y_pred, output_file="metrics_results.txt"):
    # Encode string labels into numeric values using LabelEncoder
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)  # Converts labels to integers
    y_pred_encoded = label_encoder.transform(y_pred)

    # Determine number of classes
    n_classes = len(label_encoder.classes_)

    # Compute metrics (single computation for all cases)
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    precision_macro = precision_score(y_true_encoded, y_pred_encoded, average="macro")
    recall_macro = recall_score(y_true_encoded, y_pred_encoded, average="macro")
    f1_macro = f1_score(y_true_encoded, y_pred_encoded, average="macro")


# Call function to compute metrics and save results
compute_and_save_metrics(y_true_example, y_pred_example)
