from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def missions_classes(data: dict, missions: list[str]):
    """Extract the human and llm classification labels

    Extract the human and llm classes from the evaluation json file, `config.llms.eval_output_file (summary_output.json)`. You can extract data from only a single mission or a list of missions These labels will be used to create a confusion matrix plot.

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
    logger.debug(f"data dict = {data}")

    logger.info(f"Missions for Classes = {missions}")

    for bibcode, item in data.items():
        logger.debug(f"bibcode: {bibcode}")
        human_data = item["human"]
        llm_data = item["llm"]
        llm_mission = {k for i in llm_data for k in i.keys()}
        for mission in missions:
            if mission in human_data and mission in llm_mission:
                human_labels.append(human_data.get(mission))
                llm_labels.extend(v for i in llm_data for k, v in i.items() if k == mission)
        threshold = data[bibcode]["threshold"]

    logger.debug(f"human_labels = {human_labels}, llm_labels = {llm_labels}, threshold = {threshold}")

    return human_labels, llm_labels, threshold
