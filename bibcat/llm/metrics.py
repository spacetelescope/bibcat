from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from bibcat import config
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import save_json_file

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def map_papertype(papertype: str):
    """Map human classified papertype to allowed papertypes, for instance, if `papertype` is "SUPERMENTION", it will returns "NONSCIENCE" or a custom papertype.

    Parameters
    ==========
    papertype: str, uppercase
        human classified papertype, e.g., "SCIENCE", "DATA_INFLUENCED"

    Returns
    =======
    mapped_papertype: str, uppercase
        the mapped papertype follwing `config.llms.map_papertypes`, e.g., "MENTION" if `papertype` is "SUPERMENTION"
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


def extract_eval_data(data: dict, missions: list[str]) -> dict[str, Any]:
    """Extract the evaluation data for confusion matrix and stats related to mission call-outs, and save to files.

    Extract the human/llm labels and other stats related to valid MAST mission and non MAST mission call-outs from the evaluation json file,
    `config.llms.eval_output_file (summary_output.json)`. This function is called when plotting a confusion matrix plot in `bibcat.llm.plots.py`

    Parameters
    ----------
    data : dict
        the dict of the evaluation data of `config.llms.eval_output_file (*summary_output.json)`
    missions: list[str]
        list of the mission names to extract the classification labels.

    Returns
    -------
    metrics_data: dict[str]
        contains various metrics

        threshold: float
            threshold
        n_bibcodes: int
            The number of bibcodes (papers)
        n_human_mission_callouts: int
            The number of mission callouts by human classification
        n_llm_mission_callouts: int
            The number of mission callouts by llm classification
        n_human_llm_mission_callouts: int
            The number of mission callouts by both human and llm
        n_non_mast_mission_callouts: int
            The number of non-MAST missions by llm
        n_human_llm_hallucination: int
            The number of apparent hallucination by both human and llm
            when "mission_in_text" = false
        human_llm_missions: list[str]
            The missions called out by both human and llm
        non_mast_missions; list[str]
            Non MAST missions called out by llm
        human_labels: list[str]
            True labels, human classified labels like ["SCIENCE", "MENTION"]
        llm_labels: list[str]
            Predicted labels by llm

    """

    n_bibcodes = len(data)
    logger.info(f"The number of evaluation summary data, e.g., the number of bibcodes = {n_bibcodes}")
    logger.info(f"{len(missions)} mission(s): {', '.join(missions)} is/are evaluated! ")

    # prepare lists
    human_labels = []  # Store ground truth papertypes
    llm_labels = []  # Store LLM papertypes
    human_llm_mission_callouts = []  # missions that have both human and llm classified papertypes
    non_mast_mission_callouts = []  # non-MAST missions outside the config.missions list

    # counting numbers
    n_human_mission_callouts = 0  # counting mission callouts by human
    n_llm_mission_callouts = 0  # counting mission callouts by llm, and matched between human and llm
    n_non_mast_mission_callouts = 0  # counting non-MAST missions

    # counting mission_in_text = false in both llm and human callouts
    n_human_llm_hallucination = 0

    # set the papertype for llm or human ignored the paper
    ignored_papertype = [config.llms.map_papertypes.ignore.upper()]

    for bibcode, item in data.items():
        logger.info(f"\nbibcode: {bibcode}")
        human_data = item["human"]
        n_human_mission_callouts += len(human_data)

        llm_data = item["llm"]  # only llm classification accepted by the threshold value
        llm_missions = [next(iter(i)) for i in llm_data]
        logger.info(f"llm classification accepted ={llm_missions}")
        n_llm_mission_callouts += len(llm_missions)

        llm_df_missions = [i["llm_mission"] for i in item["df"]]  # pure llm call-out/classification
        logger.info(f"llm_df_missions = {llm_df_missions}")

        non_mast_mission = [
            next(iter(i)) for i in llm_data if next(iter(i)) not in [s.upper() for s in config.missions]
        ]
        non_mast_mission_callouts.extend(non_mast_mission)

        # extracting human labels and llm labels
        for mission in missions:
            logger.info(f"Checking {mission} summary output")
            llm_mission_in_text = next((i["mission_in_text"] for i in item["df"] if i["llm_mission"] == mission), False)

            if mission in human_data and mission in llm_missions:
                logger.info(f"{mission}:both human_label and llm_label are available!")
                human_llm_mission_callouts.append(mission)

                # human labels needed for confusion matrix
                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                # map papertype to allowed papertype
                mapped_human_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype for human = '{mapped_human_papertype}'")
                human_labels.append(mapped_human_papertype)

                # llm labels = final llm papertypes of missions in "llm: []"
                labels = [v for i in llm_data for k, v in i.items() if k == mission]
                mapped_llm_papertype = map_papertype(labels[0])
                logger.debug(f"mapped papertype for llm = '{mapped_llm_papertype}'")
                llm_labels.append(mapped_llm_papertype)
                if not llm_mission_in_text:
                    logger.warning(
                        f"It appears that both human and LLM are hallucinating {mission}! Check out if the keyword search is failing"
                    )
                    n_human_llm_hallucination += 1

            elif mission in human_data and mission not in llm_missions:  # llm missing call-out
                if mission in llm_df_missions:
                    logger.info(
                        f"{mission}: Human_label is available and LLM called out {mission} but the confidence value is below the threshold."
                    )
                else:
                    logger.warning(
                        f"{mission}: Human_label is available but no llm_label is available! LLM is missing call-out! Check why LLM fails to call out mission!"
                    )

                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                # map papertype to allowed papertype for human label
                mapped_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype = '{mapped_papertype}'")
                human_labels.append(mapped_papertype)

                llm_labels.extend(ignored_papertype)

            elif mission not in human_data and mission in llm_missions:
                if llm_mission_in_text:
                    logger.warning(f"{mission}: check if human misses call-out! or the keyword search is failing")
                else:
                    logger.warning(f"{mission}: check if LLM is hallucinating or the keyword search is failing!")

                human_labels.extend(ignored_papertype)
                # llm labels = final llm papertypes of missions in "llm: []"
                labels = [v for i in llm_data for k, v in i.items() if k == mission]
                mapped_llm_papertype = map_papertype(labels[0])
                logger.debug(f"mapped papertype for llm = '{mapped_llm_papertype}'")
                llm_labels.append(mapped_llm_papertype)

            else:  # both llm and human labels not found
                if mission in llm_df_missions:
                    logger.warning(
                        f"Human misses calling out and LLM called out {mission} but the confidence value is below the threshold. Also, check out if the keyword search is failing!"
                    )
                else:
                    logger.info(f"Both human and LLM ignored {mission}!")

                human_labels.extend(ignored_papertype)
                llm_labels.extend(ignored_papertype)

    # valid mission callouts
    n_human_llm_mission_callouts = len(human_llm_mission_callouts)
    human_llm_missions = sorted(list(set(human_llm_mission_callouts)))

    # non-MAST mission callouts
    n_non_mast_mission_callouts = len(non_mast_mission_callouts)
    non_mast_missions = sorted(list(set(non_mast_mission_callouts)))
    logger.info(f"Non MAST missions: {non_mast_missions} called out; \n")
    logger.debug(f"Non MAST mission call outs: \n {non_mast_mission_callouts}")

    threshold = data[next(iter(data))]["threshold_acceptance"]
    logger.debug(f"threshold = {threshold}")

    logger.debug(f"human_labels = {human_labels}")
    logger.debug(f"llm_labels = {llm_labels}")
    logger.info(f" Set of human_labels = {set(human_labels)} and set of llm_labels = {set(llm_labels)}")

    logger.info(
        f"""The total numbers of mission callouts by human and llm are {n_human_mission_callouts} and {n_llm_mission_callouts} respectively. \n
        Among these callouts, only {n_human_llm_mission_callouts} cases are called out by both llm and human and valid for further evaluations!\n
        {n_non_mast_mission_callouts} non-MAST missions are called out!\n"""
    )

    metrics_data = {
        "threshold": threshold,
        "n_bibcodes": n_bibcodes,
        "n_human_mission_callouts": n_human_mission_callouts,
        "n_llm_mission_callouts": n_llm_mission_callouts,
        "n_non_mast_mission_callouts": n_non_mast_mission_callouts,
        "n_human_llm_mission_callouts": n_human_llm_mission_callouts,
        "n_human_llm_hallucination": n_human_llm_hallucination,
        "human_llm_missions": human_llm_missions,
        "non_mast_missions": non_mast_missions,
        "human_labels": human_labels,
        "llm_labels": llm_labels,
    }

    for k, v in metrics_data.items():
        logger.info(f"{k} : {v}")

    # evaluation metrics summary including call-outs, confusion_matrix_report, llm performance scores, etc
    output_filename = (
        Path(config.paths.output)
        / f"llms/openai_{config.llms.openai.model}/{config.llms.metrics_file}_t{metrics_data['threshold']}"
    )
    compute_and_save_metrics(metrics_data, str(output_filename) + ".txt", str(output_filename) + ".json")

    return metrics_data


def compute_and_save_metrics(
    metrics_data: dict[str],
    output_ascii_path: str | Path = "metrics_summary.txt",
    output_json_path: str | Path = "metrics_summary.json",
):
    """Compute llm performance metrics (accuracy, f1, precision, and recall scores) and other stats and save results to an ascii file

    Parameters
    ==========
    metrics_data: dict[str]
        contains various metrics

        threshold: float
            threshold
        n_bibcodes: int
            The number of bibcodes (papers)
        n_human_mission_callouts: int
            The number of mission callouts by human classification
        n_llm_mission_callouts: int
            The number of mission callouts by llm classification
        n_human_llm_mission_callouts: int
            The number of mission callouts by both human and llm
        n_non_mast_mission_callouts: int
            The number of non-MAST missions by llm
        n_human_llm_hallucination: int
            The number of apparent hallucination by both human and llm
            when "mission_in_text" = false
        human_llm_missions: list[str]
            The missions called out by both human and llm
        non_mast_missions; list[str]
            Non MAST missions called out by llm
        human_labels: list[str]
            True labels, human classified labels like ["SCIENCE", "MENTION"]
        llm_labels: list[str]
            Predicted labels by llm
    output_ascii_path: str | Path
        output file path to save the metrics summary in .txt
    output_json_path: str | Path
        output file path to save the metrics summary in .json

    Return
    ======
    None

    """

    # t: true, f: false, p: positive, n: negative
    tn, fp, fn, tp = confusion_matrix(metrics_data["human_labels"], metrics_data["llm_labels"]).ravel()

    # Encode string labels into numeric values using LabelEncoder
    label_encoder = LabelEncoder()
    human_labels_encoded = label_encoder.fit_transform(metrics_data["human_labels"])
    llm_labels_encoded = label_encoder.transform(metrics_data["llm_labels"])
    papertypes = label_encoder.classes_
    # Determine number of classes
    n_classes = len(papertypes)

    # Create classification report
    classification_performance_report = classification_report(
        human_labels_encoded, llm_labels_encoded, target_names=papertypes, digits=4, output_dict=True
    )

    logger.info(f"classification report\n {classification_performance_report}")

    # Write results to an ASCII file
    with open(output_ascii_path, "w") as f:
        f.write(f"The number of bibcodes (papers) for evaluation metrics: {metrics_data['n_bibcodes']}\n")
        f.write(f"The number of mission callouts by human: {metrics_data['n_human_mission_callouts']}\n")
        f.write(
            f"The number of mission callouts by llm with the threshold value, {metrics_data['threshold']}: {metrics_data['n_llm_mission_callouts']}\n\n"
        )
        f.write(
            f"The number of mission callouts by both human and llm: {metrics_data['n_human_llm_mission_callouts']}\n"
        )
        f.write(f"Missions called out by both human and llm: {', '.join(metrics_data['human_llm_missions'])}\n\n")

        f.write(f"The number of non-MAST mission callouts by llm: {metrics_data['n_non_mast_mission_callouts']}\n")

        f.write(
            f"The number of hallunications by both human and llm: {metrics_data['n_human_llm_hallucination']}\n Check out if the keyword search is failing\n\n"
        )

        f.write(f"Non-MAST missions called out by llm: {', '.join(metrics_data['non_mast_missions'])}\n\n")

        f.write(f"{n_classes} papertypes: {', '.join(papertypes)} are labeled\n")
        f.write(f"True Negative = {tn}, False Positive = {fp}, False Negative = {fn}, True Positive = {tp}\n\n")

        f.write(
            f"classification report\n {classification_report(human_labels_encoded, llm_labels_encoded, target_names=papertypes, digits=4)}\n"
        )
    logger.info(f"Metrics saved to {output_ascii_path}")

    # Save metrics_data and classlifcation report to a json file
    filtered_metrics_data = {k: v for k, v in metrics_data.items() if k not in {"human_labels", "llm_labels"}}
    save_json_file(output_json_path, {**filtered_metrics_data, **classification_performance_report})


def extract_roc_data(data: dict, missions: list[str]):
    """Extract the human and llm classification labels and confidences

    Extract the human classes and confidence values from the evaluation json file,
    `config.llms.eval_output_file (summary_output.json)`. You can extract data from only a single
    mission or a list of missions. The labels and confidence values will be used to create a ROC curve.

    Parameters
    ----------
    data : dict
        the dict of the evaluation data of `config.llms.eval_output_file (summary_output.json)`
    missions: list[str]
        list of the mission names to extract the classification labels.

    Returns
    -------
    tuple
        A tuple of the list of human labels, llm labels, and the hreshold value for verdict acceptance.
    human_labels: list[str]
        True labels by human, a list of papertypes, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.papertypes`
    llm_confidences: list[list[float]]
        A list of confidence score sets for all verdicts.
        For example: [[0.9, 0.1], [0.4, 0.6]], where the first set corresponds to 'SCIENCE' and the second to 'MENTION'.
    human_llm_missions: list[str], sorted
        A set of missions, each containing both human- and LLM-classified paper types, used for evaluation plots.

    """

    n_bibcodes = len(data)
    logger.info(f"The number of evaluation summary data, e.g., the number of bibcodes = {n_bibcodes}")
    logger.info(f"{len(missions)} mission(s): {', '.join(missions)} is/are evaluated! ")

    human_labels = []
    llm_confidences = []  # for ROC
    human_llm_mission_callouts = []  # missions that have both human and llm classified papertypes

    # counting mission callouts by human
    n_human_mission_callouts = 0
    # counting mission callouts by llm, and matched between human and llm
    n_llm_mission_callouts = 0

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        n_human_mission_callouts += len(human_data)

        # llm missions for ROC; need to extract from `mission_conf` data frame
        llm_mission_conf = item["mission_conf"]
        llm_mission = [item["llm_mission"] for item in llm_mission_conf]
        n_llm_mission_callouts += len(llm_mission)

        # extracting human labels and llm confidences
        for mission in missions:
            if mission in human_data and mission in llm_mission:
                logger.info(f"Checking {mission} summary output")
                human_llm_mission_callouts.append(mission)

                # human labels needed for ROC
                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                # map papertype to allowed papertype
                mapped_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype = '{mapped_papertype}'")
                human_labels.append(mapped_papertype)

                # To generate an ROC curve, we need the full range of confidence values. Use "prob_papertype" for each mission, as "mean_llm_confidences" only reflect the scores of the finally accepted papertypes in "llm:[]", which are always above the threshold. We require the varying values provided by "prob_papertype where human labels exist."
                confs = [i["prob_papertype"] for i in llm_mission_conf if i["llm_mission"] == mission]
                llm_confidences.extend(confs)
    logger.info(f"The number of valid mission callouts is {len(human_llm_mission_callouts)}")
    human_llm_missions = sorted(list(set(human_llm_mission_callouts)))

    return human_labels, llm_confidences, human_llm_missions


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
    # Invert the encoding to force "science" to be 1 and "mention" to be 0
    binarized_human_labels = lb.fit_transform(human_labels)
    logger.info(f"Classes: {lb.classes_}")  # Classes are sorted alphabetically
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
                fpr[i], tpr[i], thresholds = roc_curve(binarized_human_labels[:, i], llm_confidences[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            macro_roc_auc_ovr = roc_auc_score(
                binarized_human_labels, llm_confidences, multi_class="ovr", average="macro"
            )
            micro_roc_auc_ovr = roc_auc_score(
                binarized_human_labels, llm_confidences, multi_class="ovr", average="micro"
            )
            logger.info(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")
            logger.info(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")
            return fpr, tpr, thresholds, roc_auc, macro_roc_auc_ovr, micro_roc_auc_ovr

        elif n_papertype == 2:
            fpr, tpr, thresholds = roc_curve(binarized_human_labels, llm_confidences[:, 0])
            roc_auc = auc(fpr, tpr)
            return fpr, tpr, thresholds, roc_auc

        else:
            raise ValueError(
                f"'n_papertype' ={n_papertype} is invalid. The number of papertypes should be larger than or equal to 2."
            )
    except ValueError as ve:
        logger.error(f"ValeError encountered: {ve}", exc_info=True)

    logger.info(f"fpr={fpr}")
    logger.info(f"tpr={tpr}")
    logger.info(f"thresholds ={thresholds}")
    logger.info(f"auc ={roc_auc}")
