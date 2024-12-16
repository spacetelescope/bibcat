import numpy as np
import numpy.typing as npt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


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
        a tuple of the list of human labels, llm labels, and a threshold value.

    """

    human_labels = []
    llm_labels = []
    llm_confidences = []
    logger.debug(f"data dict = {data}")

    logger.info(f"Missions for Classes = {missions}")

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        llm_data = item["llm"]
        df = item["df"]
        # set of missions captured by llm
        llm_mission = {k for i in llm_data for k in i.keys()}
        for mission in missions:
            if mission in human_data and mission in llm_mission:
                human_labels.append(human_data.get(mission))
                labels = [v for i in llm_data for k, v in i.items() if k == mission]
                confs = [i["mean_llm_confidences"] for i in df if i["llm_mission"] == mission and i["llm_papertype"] == labels[0]]
                llm_labels.extend(labels)
                llm_confidences.extend(confs)
        threshold = data[bibcode]["threshold_acceptance"]

    logger.debug(
        f"human_labels = {human_labels}, llm_labels = {llm_labels}, threshold = {threshold}, confidences = {llm_confidences}"
    )
    logger.debug(f"confidences = {llm_confidences}")

    return human_labels, llm_labels, threshold, llm_confidences


def prepare_roc_inputs(missions: list[str], data: dict):
    """Prepare input data for ROC and AUC (area under curve)

    Parameters
    ----------
    missions : list[str]
        the list of mission names
    data: dict
        the dict of the evaluation data of `config.llms.eval_output_file (summary_output.json)`

    Returns
    -------
    tuple
        a tuple of confidences, binarized_human_labels, and n_classes.

    """

    human_labels, _, _, llm_confidences = extract_eval_data(missions=missions, data=data)

    # prep data for the roc plot
    lb = LabelBinarizer()
    binarized_human_labels = lb.fit_transform(human_labels)
    llm_confidences = np.array(llm_confidences)
    n_classes = len(set(human_labels))

    logger.debug(f"human_labels:{human_labels}")
    logger.debug(f"binarized_human_class={binarized_human_labels}")
    logger.debug(f"llm_confidences ={llm_confidences}")
    return llm_confidences, binarized_human_labels, n_classes


def get_roc_metrics(llm_confidences: npt.NDArray[np.float64], binarized_human_labels: list, n_classes: int):
    """Compute ROC curve and ROC AUC (area under curve)

    Parameters
    ----------
    llm_confidences : npt.NDArray[np.float64]
        the numpy array of llm_confidences
    binarized_true_labels: list[int]
        binarized_human_labels, e.g., [[0] [1] [1] [0][0]]

    Returns
    -------
    tuple
        a tuple of false positive rate(fpr), true positive rate(tpr), and roc_auc

    """

    # compute ROC curve and ROC AUC (area under curve) for each class
    fpr = dict()  # false positive rate
    tpr = dict()  # true positive rate
    roc_auc = dict()

    for i in range(n_classes):
        logger.debug(f"human_labels = {binarized_human_labels}, confidences = {llm_confidences[:, i]}")
        fpr[i], tpr[i], _ = roc_curve(binarized_human_labels, llm_confidences[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    logger.debug(f"fpr={fpr}, tpr[{i}]={tpr}")
    logger.debug(f"auc ={roc_auc}")

    return fpr, tpr, roc_auc
