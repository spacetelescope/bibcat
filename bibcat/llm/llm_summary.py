import pathlib

import pandas as pd

from bibcat import config
from bibcat.llm.evaluate import group_by_mission, group_by_mission_papertype
from bibcat.llm.io import read_output, write_summary
from bibcat.utils.logger_config import setup_logger

# set up logger
logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def evaluate_output_llm_only(bibcode: str = None, write_file: bool = False, base_path: str = None) -> pd.DataFrame:
    """Evaluate the output from the LLM model

    For a given paper bibcode, reads in the output from the LLM model. It
    computes a cursory accuracy score based on the number of runs.

    Returns a pandas DataFrame grouped by LLM mission and paper type, with columns for
    its mean confidence score, the count of this entry in the LLM output, the total number
    of trial runs.

    Parameters
    ----------
    bibcode : str, optional
        the paper bibcode, by default None
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

    response = read_output(bibcode=bibcode, filename=paper_output)

    # Prevent iteration error when bibcode doesn't exist in paper_output
    if response is None:
        response = []

    # filter out any cases where the llm returns an error, or there is no missions in output
    response = [i for i in response if "error" not in i.keys() and i["missions"]]

    # response is structured as:
    # - notes: str
    # - missions: [{mission: str, papertype: str, confidence: list[float], reason: str, quotes: list[str]}]

    # exit if no bibcode found in output
    if not response:
        logger.warning(f"No mission output found for {bibcode}")
        if write_file:
            write_summary(
                {bibcode: {"error": f"No mission output found for {bibcode}.", "notes": "No human review is done"}}
            )
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

    # log the output
    logger.info("Output Stats by LLM Mission and Paper Type:\n" + grouped_df.to_string(index=False))

    threshold = config.llms.performance.threshold
    inspection = config.llms.performance.inspection

    # Capitalize mission names for output to make consistent with human class mission names
    mission_group["llm_mission"] = mission_group["llm_mission"].str.upper()
    grouped_df["llm_mission"] = grouped_df["llm_mission"].str.upper()
    # reindex mission
    mm = mission_group.set_index("llm_mission")

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

    # write the summary output
    if write_file:
        write_summary(
            {
                bibcode: {
                    "notes": "No human review is done.",
                    "threshold_acceptance": threshold,
                    "threshold_inspection": inspection,
                    "llm": llm,
                    "inspection": inspection_missions,
                    "df": grouped_df.to_dict(orient="records"),
                    "mission_conf": mission_group.to_dict(orient="records"),
                }
            }
        )

    # return the dataframe
    return grouped_df
