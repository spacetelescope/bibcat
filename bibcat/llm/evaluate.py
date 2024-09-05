import json
import pathlib

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


def evaluate_output(bibcode: str = None, index: int = None, threshold: float = 0.5) -> pd.DataFrame:
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
    threshold : float, optional
        the threshold for rejection, by default 0.5

    Returns
    -------
    pd.DataFrame
        an output pandas dataframe
    """
    # get the output
    paper = get_source(bibcode=bibcode, index=index)
    bibcode = paper["bibcode"]
    response = read_output(bibcode)

    # exit if no bibcode found in output
    if not response:
        logger.info(f"No output found for {bibcode}")
        return None

    n_runs = len(response)

    logger.info(f"Evaluating output for {bibcode}")
    logger.info(f"Number of runs: {n_runs}")

    # convert output to a dataframe
    df = pd.DataFrame(
        [(k, v[0], v[1]) for i in response for k, v in i.items()], columns=["mission", "papertype", "llm_confidence"]
    )
    df = df.sort_values("mission").reset_index(drop=True)

    ## TODO: The mean and std need to be recalculated so that missing confidence (NaN) should be treated as zero.
    # group by mission and paper type,
    # get the mean confidence and the count in each group
    grouped_df = (
        df.groupby(["mission", "papertype"])
        .agg(
            mean_llm_confidence=("llm_confidence", "mean"),
            std_llm_confidence=("llm_confidence", "std"),
            count=("llm_confidence", "size"),
        )
        .reset_index()
        .rename(columns={"mission": "llm_mission", "papertype": "llm_papertype"})
    )
    grouped_df["n_runs"] = n_runs

    # get the human paper classifications
    human_classes = paper["class_missions"]
    formatted_output = "\n".join([f"{mission}: {info['papertype']}" for mission, info in human_classes.items()])
    logger.info(f"Human Classifications:\n {formatted_output}")

    # compute consistency of matches to human classification
    vv = [(k, v["papertype"]) for k, v in human_classes.items()]
    grouped_df["consistency"] = grouped_df.apply(
        lambda x: (x["count"] / x["n_runs"]) * 100 if (x["llm_mission"], x["llm_papertype"]) in vv else 0, axis=1
    )
    grouped_df["in_human_class"] = grouped_df.apply(lambda x: (x["llm_mission"], x["llm_papertype"]) in vv, axis=1)

    # get missing missions
    missing_by_human = set(grouped_df["llm_mission"]) - set(human_classes)
    missing_by_llm = set(human_classes) - set(grouped_df["llm_mission"])

    # check if missions are in the paper text body
    in_text = identify_missions_in_text(
        grouped_df["llm_mission"], " ".join(paper["title"]) + " ".join(paper["abstract"]) + " ".join(paper["body"])
    )
    grouped_df["mission_in_text"] = in_text

    # log the output
    logger.info("Output Stats by LLM Mission and Paper Type:\n" + grouped_df.to_string(index=False))
    logger.info("Missing missions by humans: " + ", ".join(missing_by_human))
    logger.info("Missing missions by LLM: " + ", ".join(missing_by_llm))

    # write the summary output
    llm = [
        {i["llm_mission"]: i["llm_papertype"]}
        for i in grouped_df.to_dict(orient="records")
        if i["mean_llm_confidence"] >= threshold
    ]
    output = {
        bibcode: {
            "human": {k: v["papertype"] for k, v in human_classes.items()},
            "threshold": threshold,
            "llm": llm,
            "missing_by_human": list(missing_by_human),
            "missing_by_llm": list(missing_by_llm),
            "df": grouped_df.to_dict(orient="records"),
        }
    }
    write_summary(output)

    # return the dataframe
    return grouped_df


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
    paper.process_paragraphs()
    paragraphs = paper.get_paragraphs()

    in_text = []
    for mission in missions:
        # get the relevant mission keyword
        keyword = op._fetch_keyword_object(mission)

        # identify the keyword in the text
        in_text.append(True if paragraphs.get(keyword.get_name()) else False)

    return in_text
