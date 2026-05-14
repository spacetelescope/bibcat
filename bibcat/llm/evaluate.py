import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from bibcat import config
from bibcat.core import parameters as params
from bibcat.core.keyword import Keyword
from bibcat.core.paper import Paper
from bibcat.llm.llm_io import get_source, read_output, write_summary
from bibcat.utils.logger_config import setup_logger

# set up logger
logger = setup_logger(__name__, level=config.logging.level)


def _filter_valid_responses(response_runs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Filter successful run responses with mission outputs.

    Parameters
    ----------
    response_runs : list[dict[str, Any]] or None
        Run-level LLM outputs for a paper.

    Returns
    -------
    list[dict[str, Any]]
        Responses that do not contain ``error`` and include non-empty
        ``missions`` entries.
    """
    if not response_runs:
        return []
    return [item for item in response_runs if "error" not in item and item.get("missions")]


def evaluate_output_from_runs(
    paper: dict[str, Any], response_runs: list[dict[str, Any]] | None, summary_log_level: int = logging.INFO
) -> tuple[pd.DataFrame | None, dict]:
    """Evaluate in-memory LLM run outputs for a paper.

    Evaluate one or more run outputs using the same workflow used by
    :func:`evaluate_output`, but without writing summary output files.

    Parameters
    ----------
    paper : dict[str, Any]
        Source paper record, including ``bibcode`` and ``class_missions``.
    response_runs : list[dict[str, Any]] or None
        Run-level LLM outputs associated with ``paper``.
    summary_log_level : int, optional
        Logging level used for evaluation summary messages emitted by this
        function and :func:`get_human_classification`. Defaults to
        ``logging.INFO`` so direct :func:`evaluate_output` calls keep their
        current verbosity, while aggregate callers can demote per-bibcode
        summaries to ``logging.DEBUG``.

    Returns
    -------
    tuple[pd.DataFrame or None, dict]
        Two-item tuple containing:

        - grouped_df: grouped evaluation dataframe, or ``None`` when no valid
          mission output exists.
        - output_item: in-memory summary dictionary with the same shape as a
          single bibcode entry produced by :func:`prepare_output`, or an
          ``error`` payload when output is missing.
    """
    bibcode = paper["bibcode"]

    #  valid response is structured as:
    # - notes: str
    # - missions: [{mission: str, papertype: str, confidence: list[float], reason: str, quotes: list[str]}]
    valid_responses = _filter_valid_responses(response_runs)

    if not valid_responses:
        logger.warning(f"No mission output found for {bibcode}")
        human_classes = get_human_classification(paper, summary_log_level=summary_log_level)
        return None, {
            "error": f"No mission output found for {bibcode}.",
            "human": {k: v["papertype"] for k, v in human_classes.items()},
        }

    n_runs = len(valid_responses)

    logger.log(summary_log_level, "Evaluating output for %s", bibcode)
    logger.log(summary_log_level, "Number of runs: %s", n_runs)

    df = pd.DataFrame([j | {"notes": i["notes"]} for i in valid_responses for j in i["missions"]])
    df = df.rename(columns={"confidence": "llm_confidences"})
    df = df.sort_values("mission").reset_index(drop=True)

    grouped_df = group_by_mission_papertype(df)
    grouped_df["n_runs"] = n_runs

    # weight the mean confidences by the frequency of occurrence
    # these represent a combined measure of frequency and confidence across multiple independent trials and categories
    grouped_df["weighted_confs"] = grouped_df.apply(
        lambda row: (row["mean_llm_confidences"] * (row["count"] / row["n_runs"])).round(3), axis=1
    )

    mission_group = group_by_mission(grouped_df)
    human_classes = get_human_classification(paper, summary_log_level=summary_log_level)
    missing_by_human, missing_by_llm = compute_consistency(paper, grouped_df, human_classes)
    hallucinated_missions = check_hallucination(grouped_df)

    # Skip building the dataframe summary string when this log level is disabled.
    if logger.isEnabledFor(summary_log_level):
        logger.log(
            summary_log_level, "Output Stats by LLM Mission and Paper Type:\n%s", grouped_df.to_string(index=False)
        )
    logger.log(summary_log_level, "Missing missions by humans: %s", ", ".join(missing_by_human))
    logger.log(summary_log_level, "Missing missions by LLM: %s", ", ".join(missing_by_llm))
    logger.log(summary_log_level, "Hallucination by LLM: %s", ", ".join(set(hallucinated_missions)))

    output = prepare_output(
        bibcode,
        config.llms.performance.threshold,
        config.llms.performance.inspection,
        grouped_df,
        mission_group,
        human_classes,
        missing_by_human,
        missing_by_llm,
        hallucinated_missions,
    )
    return grouped_df, output[bibcode]


# To be used in metrics.py and roc.py for evaluation of multiple runs and ROC/AUC computation
def build_eval_data_for_run(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    run_index: int,
    bibcodes: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build in-memory evaluation data for one run index.

    Build a summary-like mapping for one run across bibcodes. The returned
    mapping mirrors ``summary_output`` entries and is used in memory only.

    Parameters
    ----------
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    run_index : int
        Zero-based run index to evaluate.
    bibcodes : list[str], optional
        Subset of bibcodes to evaluate. If not provided, all bibcodes in
        ``llm_runs_data`` are evaluated.

    Returns
    -------
    dict[str, dict[str, Any]]
        Evaluation-style dictionary keyed by bibcode. Each value is either an
        evaluation summary item (including ``human`` and ``llm`` fields) or an
        ``error`` item for missing source/output cases.
    """
    eval_data: dict[str, dict[str, Any]] = {}
    target_bibcodes = bibcodes if bibcodes is not None else list(llm_runs_data.keys())

    for bibcode in target_bibcodes:
        paper = get_source(bibcode=bibcode)
        if not paper:
            eval_data[bibcode] = {"error": "No paper source found"}
            continue

        llm_runs = llm_runs_data.get(bibcode, [])
        run_item = llm_runs[run_index] if run_index < len(llm_runs) else {"missions": []}
        _, output_item = evaluate_output_from_runs(paper, [run_item], summary_log_level=logging.DEBUG)
        eval_data[bibcode] = output_item

    return eval_data


def evaluate_output(
    bibcode: str = None, index: int = None, write_file: bool = False, base_path: str = None
) -> pd.DataFrame:
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
    base_path : str, optional
        Optional base directory path for input and output files

    Returns
    -------
    pd.DataFrame
        an output pandas dataframe
    """
    input_path = (
        pathlib.Path(base_path)
        if base_path
        else pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}"
    )
    paper_output = input_path / f"{config.llms.prompt_output_file}"

    paper = get_source(bibcode=bibcode, index=index)
    if not paper:
        logger.warning(f"No paper source found for {bibcode}")
        if write_file:
            write_summary({bibcode: {"error": "No paper source found"}})
        return None

    bibcode = paper["bibcode"]
    response = read_output(bibcode=bibcode, filename=paper_output)

    # Prevent iteration error when bibcode doesn't exist in paper_output
    if response is None:
        response = []

    grouped_df, output_item = evaluate_output_from_runs(paper, response)

    if grouped_df is None:
        if write_file:
            write_summary({bibcode: output_item})
        return None

    if write_file:
        write_summary({bibcode: output_item}, output_path=base_path)

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


def get_human_classification(paper: dict | str, summary_log_level: int = logging.INFO):
    """Get human's mission and paper types

    Parameters
    ----------
    paper: dict or str
        dictionary or text (a row from the source dataset)
    summary_log_level : int, optional
        Logging level used for the human-classification summary message.

    Returns
    -------
    dict
        human's mission and paper type
    """
    human_classes = {key.upper(): value for key, value in paper.get("class_missions", {}).items()}
    # Skip formatting the human classification summary when this log level is disabled.
    if logger.isEnabledFor(summary_log_level):
        formatted_output = "\n".join([f"{mission}: {info['papertype']}" for mission, info in human_classes.items()])
        logger.log(summary_log_level, "Human Classifications:\n%s", formatted_output)
    return human_classes


def compute_consistency(paper: dict | str, grouped_df: pd.DataFrame, human_classes: dict):
    """Compare consistency between llm's and human's classification

    Compare llm classification with human's, get missing missions from human and llm,
    check if mission names are found in the text body.

    Parameters
    ----------
    paper: dict or str
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
    grouped_df["in_human_class"] = grouped_df.apply(
        lambda x: (x["llm_mission"].upper(), x["llm_papertype"]) in vv, axis=1
    )

    # get missing missions
    missing_by_human = set(grouped_df["llm_mission"].str.upper()) - set(human_classes)
    missing_by_llm = set(human_classes) - set(grouped_df["llm_mission"].str.upper())

    # check if missions are in the paper text body
    text = f"{paper['title'][0]}; {paper.get('abstract', '')}; {paper['body']}"
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
        grouped_df["llm_mission"][index].upper()
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
    # Capitalize mission names for output to make consistent with human class mission names
    mission_df["llm_mission"] = mission_df["llm_mission"].str.upper()
    grouped_df["llm_mission"] = grouped_df["llm_mission"].str.upper()
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
            keyword = Keyword._fetch_keyword_object(params.all_kobjs, mission, verbose=config.logging.verbose)
        except ValueError:
            # if the keyword doesn't exist, just use the provided mission name
            keywd = mission
        else:
            keywd = keyword.get_name()

        # identify the keyword in the text
        in_text.append(True if paragraphs.get(keywd) else False)

    return in_text
