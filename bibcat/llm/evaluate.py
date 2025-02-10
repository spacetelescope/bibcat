import pathlib

import numpy as np
import pandas as pd

from bibcat import config
from bibcat import parameters as params
from bibcat.core.operator import Operator
from bibcat.core.paper import Paper
from bibcat.llm.io import get_source, read_output, write_summary
from bibcat.utils.logger_config import setup_logger

# set up global operator
op = Operator(classifier="ML", mode=None, keyword_objs=params.all_kobjs)

# set up logger
logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def evaluate_output(bibcode: str = None, index: int = None, write_file: bool = False) -> pd.DataFrame:
    """Evaluate the output from the LLM model

    For a given paper bibcode, reads in the output from the LLM model and
    evaluates its performance against the human paper classifications. It matches
    the LLM's predicted mission and papertype against the human classification, and
    computes a cursory accuracy score based on the number of runs. It also provides
    a flag indicating whether the LLM mission + papertype was in the set of human
    classification.  It logs to the console and prints which prediced missions are missing by
    the humans, and which human missions are missing by the LLM.

    Returns a pandas DataFrame grouped by LLM mission and paper type, with columns for
    its mean confidence score, the count of this entry in the LLM output, the total number
    of trial runs, the accuracy of the LLM classification, and a flag indicating whether
    the LLM classification was in the human classification.

    Parameters
    ----------
    bibcode : str, optional
        the paper bibcode, by default None
    index : int, optional
        the dataset array index, by default None
    write_file : bool, optional
        Flag to write the summary output to a file, by default False

    Returns
    -------
    pd.DataFrame
        an output pandas dataframe
    """
    paper_output = (
        pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.prompt_output_file}"
    )

    paper = get_source(bibcode=bibcode, index=index)
    if not paper:
        logger.warning(f"No paper source found for {bibcode}")
        return None

    bibcode = paper["bibcode"]
    response = read_output(bibcode=bibcode, filename=paper_output)

    # filter out any cases where the llm returns an error, or there is no missions in output
    response = [i for i in response if "error" not in i.keys() and i["missions"]]

    # response is structured as:
    # - notes: str
    # - missions: [{mission: str, papertype: str, confidence: list[float], reason: str, quotes: list[str]}]

    # exit if no bibcode found in output
    if not response:
        logger.warning(f"No output found for {bibcode}")
        return None

    n_runs = len(response)

    logger.info(f"Evaluating output for {bibcode}")
    logger.info(f"Number of runs: {n_runs}")

    # convert output to a dataframe
    df = pd.DataFrame([j | {"notes": i["notes"]} for i in response for j in i["missions"]])
    df = df.rename(columns={"confidence": "llm_confidences"})
    df = df.sort_values("mission").reset_index(drop=True)

    # group by mission and paper type,
    grouped_df = group_by_mission_papertype(df)
    grouped_df["n_runs"] = n_runs

    # weight the mean confidences by the frequency of occurrence
    # these represent a combined measure of frequency and confidence across multiple independent trials and categories
    grouped_df["weighted_confs"] = grouped_df.apply(
        lambda df: (df["mean_llm_confidences"] * (df["count"] / df["n_runs"])).round(3), axis=1
    )

    # group by mission and compute final probabilities and confidences
    mission_group = group_by_mission(grouped_df)

    # get the human paper classifications
    human_classes = get_human_classification(paper)

    # compute consistency of matches to human classification and identify missing missions
    missing_by_human, missing_by_llm = compute_consistency(paper, grouped_df, human_classes)

    # check if llm hallucinates mission classification
    hallucinated_missions = check_hallucination(grouped_df)

    # log the output
    logger.info("Output Stats by LLM Mission and Paper Type:\n" + grouped_df.to_string(index=False))
    logger.info("Missing missions by humans: " + ", ".join(missing_by_human))
    logger.info("Missing missions by LLM: " + ", ".join(missing_by_llm))
    logger.info("Hallucination by LLM: " + ", ".join(set(hallucinated_missions)))

    threshold = config.llms.performance.threshold
    inspection = config.llms.performance.inspection

    # write the summary output
    if write_file:
        output = prepare_output(
            bibcode,
            threshold,
            inspection,
            grouped_df,
            mission_group,
            human_classes,
            missing_by_human,
            missing_by_llm,
            hallucinated_missions,
        )
        write_summary(output)

    # return the dataframe
    return grouped_df


def group_by_mission_papertype(df: pd.DataFrame):
    """Create a Pandas grouped_by data frame

    Group by distict mission and papertype and add the mean/std confidence by filling NaN with zeros
    for missing missions/papertypes and the count of llm_mission

    Parameters
    ----------
    df: pd.DataFrame
        paper ouput pandas data frame

    Returns
    -------
    pd.DataFrame
        pandas data frame grouped by mission and papertype

    """
    grouped_df = (
        df.fillna(0)
        .groupby(["mission", "papertype"])
        .agg(
            mean_llm_confidences=("llm_confidences", lambda x: np.round(np.mean(np.stack(x), axis=0), 3)),
            std_llm_confidences=("llm_confidences", lambda x: np.round(np.std(np.stack(x), axis=0), 3)),
            count=("mission", "size"),
        )
        .reset_index()
        .rename(columns={"mission": "llm_mission", "papertype": "llm_papertype"})
    )

    return grouped_df


def group_by_mission(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """Groups the dataframe by mission

    Groups the dataframe by mission and compute final confidence values and probabilities.
    Important relevant columns are the "prob_mission" and "total_weighted_conf" columns.
    Low "prob_mission" means the mission is likely hallucinated.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        the input grouped dataframe

    Returns
    -------
    pd.DataFrame
        the output grouped dataframe
    """

    # group by each mission and compute the total confidence values by mission
    df = grouped_df.groupby("llm_mission", as_index=False).agg(
        total_mission_conf=("weighted_confs", lambda x: x.sum().sum()),
        total_weighted_conf=("weighted_confs", lambda x: x.sum()),
    )

    # compute columns for the probability of mission and within each mission, probability of each papertype
    df["prob_mission"] = df["total_mission_conf"].apply(lambda x: (x / df["total_mission_conf"].sum()).round(3))
    df["prob_papertype"] = df.apply(lambda x: x["total_weighted_conf"] / x["total_mission_conf"], axis=1)
    return df


def get_human_classification(paper: dict | str):
    """Get human's mission and paper types

    Parameters
    ----------
    paper: dict | str
        dictionary or text (a row from the source dataset)

    Returns
    -------
    dict
        human's mission and paper type
    """
    human_classes = paper["class_missions"]
    formatted_output = "\n".join([f"{mission}: {info['papertype']}" for mission, info in human_classes.items()])
    logger.info(f"Human Classifications:\n {formatted_output}")
    return human_classes


def compute_consistency(paper: dict | str, grouped_df: pd.DataFrame, human_classes: dict):
    """Compare consistency between llm's and human's classification

    Compare llm classification with human's, get missing missions from human and llm,
    check if mission names are found in the text body.

    Parameters
    ----------
    paper: dict | str
        dictionary or text (a row from the source dataset)
    grouped_df: pd.DataFrame
        pandas data frame grouped by mission and papertype
    human_classes: dict
        human's mission and paper type

    Returns
    -------
    tuple
        missing missions by human and those by llm
    """
    # compute consistency of matches to human classification
    vv = [(k, v["papertype"]) for k, v in human_classes.items()]
    grouped_df["consistency"] = grouped_df.apply(
        lambda x: (x["count"] / x["n_runs"]) * 100 if (x["llm_mission"], x["llm_papertype"]) in vv else 0, axis=1
    )
    # whether the mission is in human classification
    grouped_df["in_human_class"] = grouped_df.apply(lambda x: (x["llm_mission"], x["llm_papertype"]) in vv, axis=1)

    # get missing missions
    missing_by_human = set(grouped_df["llm_mission"]) - set(human_classes)
    missing_by_llm = set(human_classes) - set(grouped_df["llm_mission"])

    # check if missions are in the paper text body
    text = f"{paper['title']}; {paper['abstract']}; {paper['body']}"
    in_text = identify_missions_in_text(grouped_df["llm_mission"], text)
    grouped_df["mission_in_text"] = in_text

    return missing_by_human, missing_by_llm


def check_hallucination(grouped_df):
    """Find missions by llm hallucination

    Parameters
    ----------
    grouped_df: pd.DataFrame
        pandas data frame grouped by mission and papertype

    Returns
    -------
    list
        list of hallucinated missions
    """
    grouped_df["hallucination_by_llm"] = [
        False if mission_in_text else True for mission_in_text in grouped_df["mission_in_text"]
    ]

    # Capture the hallucinated missions
    hallucinated_missions = [
        grouped_df["llm_mission"][index]
        for index, hallucination in enumerate(grouped_df["hallucination_by_llm"])
        if hallucination
    ]

    return hallucinated_missions


def prepare_output(
    bibcode: str,
    threshold: float,
    inspection: float,
    grouped_df: pd.DataFrame,
    mission_df: pd.DataFrame,
    human_classes: dict,
    missing_by_human: set,
    missing_by_llm: set,
    hallucinated_missions: list,
):
    """Prepare output and write summary

    Preparing output by gathering information.

    Parameters
    ----------
    bibcode: str
        paper bibcode
    grouped_df: pd.DataFrame
        pandas dataframe grouped by mission and papertype
    mission_df: pd.DataFrame
        grouped_df dataframe grouped by mission
    human classes: dict
        human classifed missions and papertypes
    missing_by_human: set
        set of missions missed by human
    missing_by_llm: set
        set of missions missed by llm

    Returns
    -------
    dict[str, dict[str, Any]]
        dictionary of paper, missions, and papertypes, pandas dataframe of llm assessment, and other

    """
    # reindex mission
    mm = mission_df.set_index("llm_mission")

    # pass its llm's classification if the maximum weighted-confidence value is higher than the threshold
    # the maximum value is used because the papertype's confidence is aligned with the maximum value
    llm = [
        {
            i["llm_mission"]: i["llm_papertype"],
            "confidence": mm.loc[i["llm_mission"]]["total_weighted_conf"].tolist(),
            "mission_probability": mm.loc[i["llm_mission"]]["prob_mission"],
        }
        for i in grouped_df.to_dict(orient="records")
        if max(i["weighted_confs"]) >= threshold
    ]

    # human inspection list for ambiguous classification
    inspection_missions = [
        {
            i["llm_mission"]: i["llm_papertype"],
            "confidence": mm.loc[i["llm_mission"]]["total_weighted_conf"].tolist(),
            "mission_probability": mm.loc[i["llm_mission"]]["prob_mission"],
        }
        for i in grouped_df.to_dict(orient="records")
        if max(i["weighted_confs"]) >= inspection and max(i["weighted_confs"]) < threshold
    ]

    output = {
        bibcode: {
            "human": {k: v["papertype"] for k, v in human_classes.items()},
            "threshold_acceptance": threshold,
            "threshold_inspection": inspection,
            "llm": llm,
            "inspection": inspection_missions,
            "missing_by_human": list(missing_by_human),
            "missing_by_llm": list(missing_by_llm),
            "hallucinated_missions": list(set(hallucinated_missions)),
            "df": grouped_df.to_dict(orient="records"),
            "mission_conf": mission_df.to_dict(orient="records"),
        }
    }

    return output


def identify_missions_in_text(missions: list, text: str) -> list:
    """Check if a mission is in the paper text

    Checks if a list of mission names are present in the title, abstract, and body of
    the paper text.  The text comes from the "body" field of the
    source dataset.  First, it loads the text into the bibcat Paper object,
    parses, and retrieves the paragraphs matching all the bibcat mission
    keywords.  Then it iterates over each item in the input mission list, e.g.
    all the missions from the LLM output response, identifies the correct
    keyword object, and checks if there is a corresponding paper paragraph.

    Parameters
    ----------
    missions : list
        a list of missions
    text : str
        the paper text body

    Returns
    -------
    list
        a list of boolean values indicating if the mission is in the text
    """
    # get the paper object
    # this is slow, only do this once for all missions
    paper = Paper(text, keyword_objs=params.all_kobjs, do_check_truematch=True)
    try:
        paper.process_paragraphs()
        paragraphs = paper.get_paragraphs()
    except NotImplementedError as ee:
        logger.warning("Error processing paper paragraphs: %s.", ee)
        paragraphs = None

    in_text = []
    for mission in missions:
        # if no paper paragraphs, just check if mission is in straight text
        if not paragraphs:
            in_text.append(mission in text)
            continue

        # get the relevant mission keyword
        try:
            keyword = op._fetch_keyword_object(mission)
        except ValueError:
            # if the keyword doesn't exist, just use the provided mission name
            keywd = mission
        else:
            keywd = keyword.get_name()

        # identify the keyword in the text
        in_text.append(True if paragraphs.get(keywd) else False)

    return in_text
