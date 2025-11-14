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
    metrics_data contains following variables:
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
    n_missing_ouptput_bibcodes: int
        The number of bibcodes missing output
    human_llm_missions: list[str]
        The missions called out by both human and llm
    non_mast_missions; list[str], sorted
        Non MAST missions called out by llm
    human_labels: list[str]
        True labels, human classified labels like ["SCIENCE", "MENTION"]
    llm_labels: list[str]
        Predicted labels by llm

    """

    n_bibcodes = len(data)
    threshold = config.llms.performance.threshold
    logger.info(f"The {n_bibcodes} bibcodes are evaluated in the summary_ouput_t{threshold}.json")
    logger.info(f"{len(missions)} mission(s): {', '.join(missions)} is/are evaluated!\nLooping through papers! ")

    human_labels, llm_labels = [], []  # Store ground truth papertypes and LLM papertypes
    # parallel list of bibcodes corresponding to each label pair (one entry per mission)
    label_bibcodes: list[str] = []
    # parallel list of raw labels (before mapping) for each sample: dict with keys 'human_raw' and 'llm_raw'
    label_raws: list[dict] = []
    human_llm_mission_callouts = []  # missions that have both human and llm classified papertypes
    non_mast_mission_callouts = []  # non-MAST missions outside the config.missions list

    n_human_mission_callouts = n_llm_callouts = (
        0  # counting human mission callouts and llm callouts including non-MAST missions
    )
    n_human_llm_hallucination = 0  # counting mission_in_text = false in both llm and human callouts
    n_missing_output_bibcodes = 0  # counting papers that ignored by both human and llm

    # set the papertype for llm or human ignored the paper
    ignored_papertype = config.llms.map_papertypes.ignore.upper()

    for bibcode, item in data.items():
        logger.info(f"\nbibcode: {bibcode}")
        err = item.get("error", "")
        if not err:
            human_data = item.get("human")
            n_human_mission_callouts += len(human_data)

            llm_data = item.get("llm")  # only llm classification accepted by the threshold value
            n_llm_callouts += len(llm_data)
            llm_missions = [next(iter(i)) for i in llm_data]  # get llm missions
            logger.info(f"llm classification accepted ={llm_missions}")

            # all llm mission call-out
            llm_df_missions = [i["llm_mission"] for i in item.get("df")]
            logger.info(f"llm_df_missions = {llm_df_missions}")

            # store the list of non MAST missions
            non_mast_mission = [
                next(iter(i)) for i in llm_data if next(iter(i)) not in [s.upper() for s in config.missions]
            ]
            non_mast_mission_callouts.extend(non_mast_mission)

            # extracting human labels and llm labels (also record bibcodes and raw labels per mission)
            human_labels, llm_labels, n_human_llm_hallucination = extract_labels(
                missions,
                human_labels,
                llm_labels,
                human_llm_mission_callouts,
                ignored_papertype,
                item,
                human_data,
                llm_data,
                llm_missions,
                llm_df_missions,
                n_human_llm_hallucination,
                bibcode,
                label_bibcodes,
                label_raws,
            )
            # processed non-error summary output for this bibcode

        elif "No paper source found" in err:
            # should not count as missing llm output when paper source is not found
            pass

        elif "No mission output found" in err:
            n_missing_output_bibcodes += 1
            # set llm labels to ignored papertype
            llm_labels.extend([ignored_papertype] * len(missions))

            human_data = item.get("human") or {}

            # record bibcodes and raw label placeholders for these missions (one per mission)
            for mission in missions:
                label_bibcodes.append(bibcode)
                # human raw label if present, else explicit marker; llm raw set to explicit marker since no output
                human_raw = human_data.get(mission) if human_data and mission in human_data else "IGNORED"
                label_raws.append({"human_raw": human_raw, "llm_raw": "IGNORED"})

            n_human_mission_callouts += len(human_data)
            # assign human labels when human classifications exist
            human_labels = human_labels_when_no_llm_output(missions, human_data, human_labels, ignored_papertype)

            # handled missing mission output for this bibcode

    # non-MAST mission callouts
    logger.info(f"Non MAST missions: {sorted(list(set(non_mast_mission_callouts)))} called out; \n")
    logger.debug(f"Non MAST mission call outs: \n {non_mast_mission_callouts}")

    logger.debug(f"human_labels = {human_labels}")
    logger.debug(f"llm_labels = {llm_labels}")
    logger.info(f" Set of human_labels = {set(human_labels)} and set of llm_labels = {set(llm_labels)}")

    n_llm_mission_callouts = n_llm_callouts - len(non_mast_mission_callouts)
    logger.info(
        f"""The total numbers of mission callouts by human and llm are {n_human_mission_callouts} and {n_llm_mission_callouts} respectively. \n
        Among these callouts, only {len(human_llm_mission_callouts)} cases are called out by both llm and human and valid for further evaluations!\n
        {len(non_mast_mission_callouts)} non-MAST missions are called out!\n"""
    )

    metrics_data = {
        "threshold": threshold,
        "n_bibcodes": n_bibcodes,
        "n_human_mission_callouts": n_human_mission_callouts,
        "n_llm_mission_callouts": n_llm_mission_callouts,
        "n_non_mast_mission_callouts": len(non_mast_mission_callouts),
        "n_human_llm_mission_callouts": len(human_llm_mission_callouts),
        "n_human_llm_hallucination": n_human_llm_hallucination,
        "n_missing_output_bibcodes": n_missing_output_bibcodes,
        "human_llm_missions": sorted(list(set(human_llm_mission_callouts))),
        "non_mast_missions": sorted(list(set(non_mast_mission_callouts))),
        "human_labels": human_labels,
        "llm_labels": llm_labels,
        "label_bibcodes": label_bibcodes,
        "label_raws": label_raws,
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


def extract_labels(
    missions: list[str],
    human_labels: list[str],
    llm_labels: list[str],
    human_llm_mission_callouts: list[str],
    ignored_papertype: str,
    item: dict[str, dict[str, Any]],
    human_data: dict[str, str],
    llm_data: list[dict[str, Any]],
    llm_missions: list[str],
    llm_df_missions: list[str],
    n_human_llm_hallucination: int,
    bibcode: str,
    label_bibcodes: list[str],
    label_raws: list[dict],
) -> tuple[list[str], list[str], int]:
    """
    Extract human and llm papertype labels when the summary output of a bibcode
    has classification items other than "error"

    This function extracts human and llm papertype labels from the summary_output for constructing confusion matrix,
    then map the papertypes to the allowed papertypes (for instance, `MENTION` maps to `NONSCIENCE`).
    Because the summary_output provides only human and llm callouts of only relevant missions, not all MAST missions,
    we need to extract the relevant labels depending on the following various conditions:

    1. When both human and LLM call out a given mission and their papertypes, assign them to the relevant papertypes.
    2. When human calls out the mission but LLM ignores the paper, assign the human label to its relevant papertype but the LLM label to `ignored_papertype`.
    3. When human ignores the paper but LLM calls out with a papertype, assign the LLM label to its relevant papertype but the human label to `ignored_papertype`.
    4. When both human and LLM ignore the paper for the mission, assign both to `ignored_papertype`.

    Parameters
    ----------
    missions: list[str]
        MAST missions of interest
    human_labels: list[str]
        human papertypes before this bibcode
    llm_labels: list[str]
        llm papertypes before this bibcode
    human_llm_mission_callouts: list[str]
        Missions called out by both human and llm
    ignored_papertype: str, uppercase
        `config.llms.map_papertypes.ignore.upper()`, for instance, `NONSCIENCE`
    item: dict[str, dict[str, Any]]
        bibcode dictionary item
    human_data: dict[str, str]
        dictionary values of `item["human"]`, e.g., `{"JWST": "SCIENCE"}`
    llm_data: list[dict[str, Any]]
        dictionary value of `item["llm"]`
    llm_missions: list[str]
        list of LLM mission callouts in `item["llm"]`
    llm_df_missions: list[str]
        list of LLM mision callouts in `item["df"]`
    n_human_llm_hallucination: int
        the number of hallucinations before the current bibcode

    Returns
    -------
    human_labels: list[str]
        human papertype labels updated after the current bibcode
    llm_labels: list[str]
        llm papertype labels updated after the current bibcode
    n_human_llm_hallucination: int
        the number of hallucinations updated after the current bibcode

    """
    for mission in missions:
        logger.info(f"Checking {mission} summary output")
        llm_mission_in_text = next((i["mission_in_text"] for i in item.get("df") if i["llm_mission"] == mission), False)

        # capture raw labels before mapping for this mission
        # use explicit "IGNORED" marker when absent to make outputs clearer
        human_raw = human_data.get(mission) if human_data and mission in human_data else "IGNORED"
        llm_raw = next((v for i in llm_data for k, v in i.items() if k == mission), "IGNORED")

        # When both human and llm callout the mission with its papertype,
        # this below blcok will extract and map the human/llm papertypes to their designated papertype in the config file
        if mission in human_data and mission in llm_missions:
            logger.info(f"{mission}:both human_label and llm_label are available!")
            human_llm_mission_callouts.append(mission)

            # set human labels after mapping papertype
            human_labels = append_human_labels_with_mapped_papertype(human_data, mission, human_labels)

            # set llm labels = final llm papertypes of missions in "llm: []" after mapping
            llm_labels = append_llm_labels_with_mapped_papertype(llm_data, mission, llm_labels)

            if not llm_mission_in_text:
                logger.warning(
                    f"It appears that both human and LLM are hallucinating {mission}! Check out if the keyword search is failing"
                )
                n_human_llm_hallucination += 1

        # When human classification is available but llm doesn't call out,
        # this block will extract and map human papertype to designated papertype in the config file
        # and assign llm_papertype to ignored_papertype (i.e., NONSCIENCE)
        elif mission in human_data and mission not in llm_missions:  # llm missing call-out
            if mission in llm_df_missions:
                logger.info(
                    f"{mission}: Human_label is available and LLM called out {mission} but the confidence value is below the threshold."
                )
            else:
                logger.warning(
                    f"{mission}: Human_label is available but no llm_label is available! LLM is missing call-out! Check why LLM fails to call out mission!"
                )
                # set human labels after mapping papertype
            human_labels = append_human_labels_with_mapped_papertype(human_data, mission, human_labels)

            # set llm label to ignored papertype
            llm_labels.append(ignored_papertype)

        # When there is not human callout but llm callouts mission with papertype, we assgin "NONSCIENCE"
        # to human papertype and extract and map human papertype to designated papertype in the config file
        elif mission not in human_data and mission in llm_missions:
            if llm_mission_in_text:
                logger.warning(f"{mission}: check if human misses {mission} call-out! or the keyword search is failing")
            else:
                logger.warning(
                    f"{mission}: check if LLM is hallucinating {mission} call-out or the keyword search is failing!"
                )

            human_labels.append(ignored_papertype)

            # llm labels = final llm papertypes of missions in "llm: []" after mapping
            llm_labels = append_llm_labels_with_mapped_papertype(llm_data, mission, llm_labels)

        # both llm and human labels not found in the main level ("llm:[]"), so assign ignored type
        else:
            if mission in llm_df_missions:
                logger.warning(
                    f"Human misses calling out and LLM called out {mission} but the confidence value is below the threshold. Also, check out if the keyword search is failing!"
                )
            else:
                logger.info(f"Both human and LLM ignored {mission}!")

            human_labels.append(ignored_papertype)
            llm_labels.append(ignored_papertype)

        # record bibcode and raw labels for this mission sample (one entry per mission)
        label_bibcodes.append(bibcode)
        label_raws.append({"human_raw": human_raw, "llm_raw": llm_raw})
    return human_labels, llm_labels, n_human_llm_hallucination


def map_papertype(papertype: str) -> str | None:
    """Map a classified papertype to an allowed papertypes, for instance, if `papertype` is "SUPERMENTION" or "IGNORE", it will returns "NONSCIENCE" or a custom papertype.

    Parameters
    ==========
    papertype: str, uppercase
        human or llm classified papertype, e.g., "SCIENCE", "DATA_INFLUENCED"

    Returns
    =======
    mapped_papertype: str, uppercase
        mapped papertype follwing `config.llms.map_papertypes`, e.g., "MENTION" if `papertype` is "SUPERMENTION"
    """
    logger.debug(f"map_papertype(): input classified papertype to map = '{papertype}'")
    try:
        if papertype.lower() in config.llms.map_papertypes:
            mapped_value = config.llms.map_papertypes.get(papertype.lower())
            if mapped_value.upper() in config.llms.papertypes:
                mapped_papertype = mapped_value.upper()
                logger.debug(f"map_papertype(): mapped papertype is '{mapped_papertype}'")
                return mapped_papertype
            else:
                raise ValueError(
                    f"The mapped papertype '{mapped_value}' for the input papertype '{papertype}' is not a valid classification."
                )
        else:
            raise KeyError(f"The input papertype '{papertype}' is an invalid papertype.")
    except KeyError as ke:
        logger.error(f"KeyError encountered: {ke}", exc_info=True)
    except ValueError as ve:
        logger.error(f"ValeError encountered: {ve}", exc_info=True)


def append_human_labels_with_mapped_papertype(
    human_data: dict[str, str], mission: str, human_labels: list[str]
) -> None:
    """Append human papertype to the `human_labels` list after mapping it to the allowed papertype

    Parameters
    ==========
    human_data: dict[str]
        human classification data per bibcode in summary_output.
        e.g., "human": {"GALEX": "SCIENCE", "HST": "DATA-INFLUENCED"}
    mission: str
        mission name, e.g., ROMAN
    human_labels: list[str]
        list of human papertype labels for confusion matrix, e.g., ["SCIENCE","NONSCIENCE","SCIENCE"]

    Returns
    ======
    None
    """

    logger.debug(f"initial human papertype = '{human_data.get(mission)}'")
    mapped_human_papertype = map_papertype(human_data.get(mission))
    logger.debug(f"mapped papertype = '{mapped_human_papertype}'")
    human_labels.append(mapped_human_papertype)
    return human_labels


def human_labels_when_no_llm_output(missions, human_data, human_labels, ignored_papertype):
    """Assign human labels when human classifications exist even with no llm output

    Parameters
    ----------
    missions: list[str]
        list of missions
    human_data: dict[str, str]
        dictionary values of item["human"], e.g., {"JWST": "SCIENCE"}
    human_labels: list[str]
        True labels by human, a list of papertypes before papertype mapping,
        For example, ["SCIENCE", "MENTION"]
    ignored_papertype: str, uppercase
        config.llms.map_papertypes.ignore.upper(), for instance, "NONSCIENCE"

    Returns
    -------
    human_labels: list[str]
        updated human labels based on the presence of human classifications
    """

    # when no human label found for any mission at all, human:[]
    if not human_data:
        human_labels.extend([ignored_papertype] * len(missions))
        return human_labels

    # when at least one mission found in human_data
    for mission in missions:
        # e.g., if "HST": "SCIENCE", this condition is met
        if mission in human_data:
            human_labels = append_human_labels_with_mapped_papertype(human_data, mission, human_labels)
        # e.g., the below condition meets if if "HST": "SCIENCE" and mission!="HST"
        else:
            human_labels.append(ignored_papertype)
    return human_labels


def append_llm_labels_with_mapped_papertype(llm_data: list[dict], mission: str, llm_labels: list[str]) -> None:
    """Append llm papertype to the `llm_labels` list after mapping it to the allowed papertype

    Parameters
    ==========
    llm_data: list[dict]
        llm classification data per bibcode in summary_output.
        e.g., "llm": [{"JWST": "SCIENCE"}, {"ROMAN": "SUPERMENTION"}, {"HST": "SCIENCE"}]
    mission: str
        mission name, e.g., ROMAN
    llm_labels: list[str]
        list of llm papertype labels for confusion matrix, e.g., ["SCIENCE","NONSCIENCE","SCIENCE"]

    Returns
    ======
    None
    """

    label = next((v for i in llm_data for k, v in i.items() if k == mission), None)
    logger.debug(f"initial llm papertype = {label}")
    mapped_llm_papertype = map_papertype(label)
    logger.debug(f"mapped papertype for llm = '{mapped_llm_papertype}'")
    llm_labels.append(mapped_llm_papertype)
    return llm_labels


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
        n_missing_output_bibcodes: int
            The number of bibcodes missing output
        human_llm_missions: list[str], sorted
            The missions called out by both human and llm
        non_mast_missions; list[str], sorted
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
    # normalize confusion matrix over the true (rows)
    tnr, fpr, fnr, tpr = confusion_matrix(
        metrics_data["human_labels"], metrics_data["llm_labels"], normalize="true"
    ).ravel()

    confusion_matrix_metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
    }
    # Encode string labels into numeric values using LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(config.llms.papertypes)

    human_labels_encoded = label_encoder.transform(metrics_data["human_labels"])
    llm_labels_encoded = label_encoder.transform(metrics_data["llm_labels"])
    papertypes = label_encoder.classes_
    # Determine number of classes
    n_classes = len(papertypes)

    # Create classification report
    classification_performance_report = classification_report(
        human_labels_encoded, llm_labels_encoded, target_names=papertypes, digits=4, output_dict=True
    )

    logger.info(f"classification report\n {classification_performance_report}")
    # For binary classification, also collect bibcodes for TN/FP/FN/TP along with raw labels.
    label_bibcodes = metrics_data.get("label_bibcodes", [])
    label_raws = metrics_data.get("label_raws", [])
    entries = collect_confusion_cell_entries(
        human_labels_encoded, llm_labels_encoded, label_bibcodes, label_raws, n_classes
    )

    # Write results to an ASCII file
    with open(output_ascii_path, "w") as f:
        f.write(f"The number of bibcodes (papers) for evaluation metrics: {metrics_data['n_bibcodes']}\n")
        f.write(
            f"The number of bibcodes missing output, i.e., ignored papers by flagship and mast: {metrics_data['n_missing_output_bibcodes']}\n"
        )
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
            f"True Negative Rate / Specificity = {tnr.round(4)}, False Positive Rate = {fpr.round(4)}, False Negative Rate = {fnr.round(4)}, True Positive Rate / Recall = {tpr.round(4)}\n\n"
        )

        f.write(
            f"classification report\n {classification_report(human_labels_encoded, llm_labels_encoded, target_names=papertypes, digits=4)}\n"
        )
    logger.info(f"Metrics saved to {output_ascii_path}")

    # Save metrics_data and classlifcation report to a json file
    filtered_metrics_data = {
        k: v for k, v in metrics_data.items() if k not in {"human_labels", "llm_labels", "label_bibcodes", "label_raws"}
    }

    # For binary classification, also collect bibcodes for TN/FP/FN/TP along with raw labels.
    # Delegate to helper functions for modularity and testing.
    label_bibcodes = metrics_data.get("label_bibcodes", [])
    label_raws = metrics_data.get("label_raws", [])
    entries = collect_confusion_cell_entries(
        human_labels_encoded, llm_labels_encoded, label_bibcodes, label_raws, n_classes
    )

    # append bibcode lists into the saved json
    save_json_file(
        output_json_path,
        {
            **filtered_metrics_data,
            **confusion_matrix_metrics,
            **classification_performance_report,
            "fp_bibcodes": entries.get("fp", []),
            "fn_bibcodes": entries.get("fn", []),
            "tp_bibcodes": entries.get("tp", []),
            "tn_bibcodes": entries.get("tn", []),
        },
    )

    # all done; JSON saved below


def collect_confusion_cell_entries(
    human_labels_encoded: NDArray[np.int64],
    llm_labels_encoded: NDArray[np.int64],
    label_bibcodes: list[str],
    label_raws: list[dict],
    n_classes: int,
) -> dict:
    """Collect bibcode + raw-label dicts for confusion matrix cells.

    Returns a dict with keys 'tn','fp','fn','tp' each mapping to a list of entry dicts.
    Each entry dict has keys: 'bibcode','human_raw','llm_raw'.
    """
    # Default empty structure
    entries = {"tn": [], "fp": [], "fn": [], "tp": []}

    if not (
        n_classes == 2
        and label_bibcodes
        and label_raws
        and len(label_bibcodes) == len(human_labels_encoded)
        and len(label_raws) == len(human_labels_encoded)
    ):
        return entries

    for t, p, bc, raw in zip(human_labels_encoded, llm_labels_encoded, label_bibcodes, label_raws):
        entry = {"bibcode": bc, "human_raw": raw.get("human_raw"), "llm_raw": raw.get("llm_raw")}
        if t == 0 and p == 0:
            entries["tn"].append(entry)
        elif t == 0 and p == 1:
            entries["fp"].append(entry)
        elif t == 1 and p == 0:
            entries["fn"].append(entry)
        elif t == 1 and p == 1:
            entries["tp"].append(entry)
    return entries


def extract_roc_data(data: dict[str, dict[str, Any]], missions: list[str]):
    """Extract the human and llm classification labels and confidences

    Extract the human classes and confidence values from the evaluation json file,
    `config.llms.eval_output_file (summary_output.json)`.
    You can extract data from only a single mission or a list of missions.
    The human labels (ground truth) and llm confidence values will be used to create a ROC curve.

    Parameters
    ----------
    data : dict[str, dict[str, Any]]
        the dict of the evaluation data of `config.llms.eval_output_file (summary_output.json)`
    missions: list[str]
        list of the mission names to extract the classification labels.

    Returns
    -------
    tuple
        A tuple of the list of human labels, llm labels, and the hreshold value for verdict acceptance.
    human_labels: list[str]
        True labels by human, a list of papertypes, .e.g, "SCIENCE" or "MENTION"(or "NONSCIENCE"),
        see the allowed classifications in `config.llms.papertypes`
        For example, ["SCIENCE", "MENTION"]
    llm_confidences: list[list[float]]
        A list of confidence score sets for all verdicts ([[p_science, p_mention],])
        where p_science and p_mention represent confidence values of "SCIENCE" and "MENTION"(or "NONSCIENCE") respectively.
        For example: [[0.9, 0.1], [0.4, 0.6]]
    human_llm_missions: list[str], sorted
        A set of missions, each containing both human- and LLM-classified paper types, used for evaluation plots.

    """

    n_bibcodes = len(data)
    logger.info(f"The number of evaluation summary data, e.g., the number of bibcodes = {n_bibcodes}")
    logger.info(f"{len(missions)} mission(s): {', '.join(missions)} is/are evaluated! ")

    human_labels = []
    llm_confidences = []  # for ROC
    human_llm_mission_callouts = []  # missions that have both human and llm classified papertypes
    n_missing_output_bibcodes = 0

    # set the papertype for llm or human ignored the paper
    ignored_papertype = config.llms.map_papertypes.ignore.upper()

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")

        # when llm output summary exists
        err = item.get("error")
        if not err:
            human_data = item["human"]

            # llm missions for ROC; need to extract confidence values from `mission_conf` data frame
            # where missions are accepted base on missions from item["llm"].
            llm_data = item.get("llm")
            llm_mission_conf = item["mission_conf"]
            llm_missions = [next(iter(i)) for i in llm_data]  # get llm missions
            logger.info(f"llm classification accepted ={llm_missions}")

            # extracting/assigning human labels and llm confidences
            for mission in missions:
                # When both human and llm callout the mission with its papertype,
                # this clause will extract and map the human papertype to its designated papertype in the config file
                # and extend the values of item["mission_conf"]["llm_mission"]["prob_papertype"] to `llm_confidences`
                if mission in human_data and mission in llm_missions:
                    logger.info(f"Checking {mission} summary output")
                    human_llm_mission_callouts.append(mission)

                    # set human labels after mapping papertype
                    append_human_labels_with_mapped_papertype(human_data, mission, human_labels)

                    # To generate an ROC curve, we need the full range of confidence values.
                    # Use "prob_papertype" for each mission, as "mean_llm_confidences"
                    # only reflect the scores of the finally accepted papertypes in "llm:[]",
                    # which are always above the threshold. We require the varying values
                    # provided by "prob_papertype where human labels exist."
                    confs = [i["prob_papertype"] for i in llm_mission_conf if i["llm_mission"] == mission]
                    llm_confidences.extend(confs)

                # When human classification is available but llm doesn't call out,
                # this block will extract and map human papertype to designated papertype in the config file
                # but extend [0.0,1.0] ("NONSCIENCE") to `llm_confidences'
                elif mission in human_data and mission not in llm_missions:  # llm missing call-out
                    append_human_labels_with_mapped_papertype(human_data, mission, human_labels)
                    llm_confidences.append([0.0, 1.0])

                # When there is not human callout but llm callouts mission with papertype
                # we assgin "NONSCIENCE" to human papertype and extract llm confidences from "prob_papertype"
                elif mission not in human_data and mission in llm_missions:
                    human_labels.append(ignored_papertype)
                    confs = [i["prob_papertype"] for i in llm_mission_conf if i["llm_mission"] == mission]
                    llm_confidences.extend(confs)

                # both llm and human labels not found in the main level ("llm:[]"), so assign ignored type
                # but item["mission_conf"] could have mission callouts
                else:
                    human_labels.append(ignored_papertype)
                    llm_confidences.append([0.0, 1.0])

        elif "No paper source found" in err:
            # should not count as missing llm output when paper source is not found
            pass

        # assign the roc input values to NONSCIENCE and [0.0, 1.0] when there is no llm output
        elif "No mission output found" in err:
            n_missing_output_bibcodes += 1
            llm_confidences.extend([[0.0, 1.0]] * len(missions))

            human_data = item.get("human") or {}
            # assign human labels when human classifications exist.
            human_labels = human_labels_when_no_llm_output(missions, human_data, human_labels, ignored_papertype)

    logger.info(f"The number of the mission callouts by both human and llm is {len(human_llm_mission_callouts)}")

    return human_labels, llm_confidences, sorted(list(set(human_llm_mission_callouts)))


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
