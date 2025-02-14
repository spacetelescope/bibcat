import numpy as np
import numpy.typing as npt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from bibcat import config

# from bibcat import parameters as params
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
            if mapped_value.upper() in config.llms.classifications:
                mapped_papertype = mapped_value.upper()
                logger.debug(f"map_papertype(): mapped papertype is '{mapped_papertype}'")
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

    return mapped_papertype


def extract_eval_data(data: dict, missions: list[str]):
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

    Returns
    -------
    tuple
        a tuple of the list of human labels, llm labels, and the hreshold value for verdict acceptance.
    human_labels: list[str]
        papertype, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.classifications`
    llm_labels: list[str]
        papertype, .e.g, "SCIENCE" or "MENTION", see the allowed classifications in `config.llms.classifications`
    threshold: float
        The threshold value used to determine if the LLM papertype is accepted.
        If the papertype's confidence is greater than or equal to this threshold,
        the mission along with the papertype is recorded as accepted.

    """

    human_labels = []
    llm_labels = []
    logger.debug(f"The number of evaluation summary data, e.g., the number of bibcodes = {len(data)}")

    logger.info(f"Missions for Classes = {missions}")

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        llm_data = item["llm"]

        # extracting human labels and llm labels
        llm_mission = {k for i in llm_data for k in i.keys()}
        for mission in missions:
            if mission in human_data and mission in llm_mission:
                # human and llm labels needed for confusion matrix
                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                mapped_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype = '{mapped_papertype}'")
                human_labels.append(mapped_papertype)
                # labels = final llm papertypes of missions in "llm: []"
                labels = [v for i in llm_data for k, v in i.items() if k == mission]
                llm_labels.extend(labels)

        threshold = data[bibcode]["threshold_acceptance"]

    logger.debug(f"threshold = {threshold}")
    logger.info(
        f"the numbers of the valid human labels and llm labels = {len(human_labels)} and {len(llm_labels)} respectively."
    )
    logger.debug(f"human_labels = {human_labels}")
    logger.debug(f"llm_labels = {llm_labels}")

    return human_labels, llm_labels, threshold


def prepare_roc_inputs(data: dict, missions: list[str]):
    """Prepare input data for ROC and AUC (area under curve)

    Parameters
    ----------
    data: dict
        the dict of the evaluation data of `config.llms.eval_output_file (summary_output.json)`
    missions : list[str]
        the list of mission names

    Returns
    -------
    tuple
        a tuple of confidences, binarized_human_labels, and n_classes.
    llm_confidences: npt.NDArray[np.float64]
        A list of confidence score pairs for all verdicts. Each inner list contains two floats:
        the first for "SCIENCE" and the second for "MENTION". For example: [[0.9 0.1] [0.4 0.6]]
    binarized_human_labels: list[list[int]]
        Binarized human labels as ROC input, e.g., [[0 1 0 0][0 1 0 0]]
    n_papertype: int
        the number of available papertypes
    n_verdicts: int
        the number of MAST mission papertype verdicts by LLM

    """

    human_labels = []
    llm_confidences = []

    logger.info(f"The number of evaluation summary data, e.g., the number of bibcodes = {len(data)}")
    logger.info(f"Preparing data for ROC curve for {missions}")

    # Extract human labels and llm confidence values
    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        llm_mission_conf = item["mission_conf"]

        llm_missions = {i["llm_mission"] for i in llm_mission_conf}
        for mission in missions:
            if mission in human_data and mission in llm_missions:
                logger.info(f"Checking {mission} summary output")
                logger.debug(f"human classified papertype = '{human_data.get(mission)}'")
                mapped_papertype = map_papertype(human_data.get(mission))
                logger.debug(f"mapped papertype = '{mapped_papertype}'")
                human_labels.append(mapped_papertype)

                # To generate an ROC curve, we need the full range of confidence values. Use "prob_papertype" for each mission, as "mean_llm_confidences" only reflect the scores of the finally accepted papertypes in "llm:[]", which are always above the threshold. We require the varying values provided by "prob_papertype where human labels exist."
                confs = [i["prob_papertype"] for i in llm_mission_conf if i["llm_mission"] == mission]
                llm_confidences.extend(confs)
            # elif mission not in human_missions and mission not in llm_missions:

    logger.debug(f"human_labels:{human_labels}")

    # prep data for the roc plot
    lb = LabelBinarizer()
    binarized_human_labels = lb.fit_transform(human_labels)
    logger.debug(f"binarized_human_labels={binarized_human_labels}")

    llm_confidences = np.array(llm_confidences)
    logger.debug(f"llm_confidences ={llm_confidences}")

    n_papertype = len(set(human_labels))
    n_verdicts = len(human_labels)
    logger.info(f"The number of verdicts for ROC = {n_verdicts}")
    return llm_confidences, binarized_human_labels, n_papertype, n_verdicts


def get_roc_metrics(llm_confidences: npt.NDArray[np.float64], binarized_human_labels: list[int], n_papertype: int):
    """Compute ROC curve and ROC AUC (area under curve)

    Parameters
    ----------
    llm_confidences : npt.NDArray[np.float64]
        the numpy array of llm_confidences
    binarized_true_labels: list[int]
        binarized_human_labels, e.g., [[0] [1] [1] [0] [0]]

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

        elif n_papertype == 2:
            fpr, tpr, _ = roc_curve(binarized_human_labels, llm_confidences[:, 0])
            roc_auc = auc(fpr, tpr)

        else:
            raise ValueError(
                f"'n_papertype' ={n_papertype} is invalid. The number of papertypes should be larger than or equal to 2."
            )
    except ValueError as ve:
        logger.error(f"ValeError encountered: {ve}", exc_info=True)

    logger.info(f"fpr={fpr}")
    logger.info(f"tpr={tpr}")
    logger.info(f"auc ={roc_auc}")

    return fpr, tpr, roc_auc
