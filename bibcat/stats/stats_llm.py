import json
import os
import pathlib
from typing import Any, Dict, Set, Tuple

import pandas as pd

from bibcat import config
from bibcat.llm.io import read_output, write_summary
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import save_json_file

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def save_evaluation_stats(
    input_path: pathlib.Path, output_path: pathlib.Path, threshold_acceptance: float, threshold_inspection: float
):
    """Save the evaluation stats file and the inconsistent classification list file.  The stats file also includes the lists of the bibcodes for llm classification acceptance and for human inspection for egdy case confidence values. The second file includes the list of data where there are inconsistencies between human and llm's classifications caused by human's missing missions, llm's hallucination, or simply inconsistent classifications.

    Parameters
    ==========
    input_filepath: pathlib.Path
        input paper_output filename path for statistics
    output_filepath: pathlib.Path
        file name path to save the JSON file
    threshold_acceptance: float
        threshold value to accept llm papertype
    threshold_inspection: float
        threshold value to filter papers required for human inspection


    Returns
    =======
    """

    data = read_output(bibcode=None, filename=input_path)
    logger.debug(f"Loaded data: {data}")

    # Build DataFrame
    try:
        df = pd.DataFrame(
            [
                (
                    item["llm_mission"].lower(),  # mission
                    item["llm_papertype"].lower(),  # papertype
                    item["mean_llm_confidences"],
                    bibcode,
                    item["in_human_class"],
                    item["mission_in_text"],
                    item["consistency"],
                )
                for bibcode, eval_item in data.items()
                for index, item in enumerate(eval_item["df"])
            ],
            columns=[
                "mission",
                "papertype",
                "mean_llm_confidences",
                "bibcode",
                "in_human_class",
                "mission_in_text",
                "consistency",
            ],
        )
    except Exception as e:
        logger.error(f"Error during operation DataFrame creation: {e}")
        raise
    df = df.sort_values(["mission", "papertype"]).reset_index(drop=True)

    # grouping DF and aggregate other properies
    grouped_df = group_by_agg("mean_llm_confidences", threshold_acceptance, threshold_inspection, df)

    # Write the statistics summary
    write_stats(output_path, threshold_acceptance, threshold_inspection, grouped_df)

    # Filter data where human and llm classifications are inconsistent due to either hallucination or human classification is missing or classifications between human and llm differ
    inconsistent_classification_df = df[(df["consistency"] != 100.0) | (~df["mission_in_text"])]

    inconsistency_filename: pathlib.Path = (
        pathlib.Path(config.paths.output)
        / f"llms/openai_{config.llms.openai.model}/{config.llms.inconsistent_classifications_file}_t{config.llms.performance.threshold}.json"
    )

    inconsistent_classification_df.to_json(
        inconsistency_filename,
        orient="records",
        lines=True,
    )


def save_operation_stats(
    input_path: pathlib.Path, output_path: pathlib.Path, threshold_acceptance: float, threshold_inspection: float
):
    """Save the stats file from the operational llm classification. This file also includes the lists of the bibcodes for llm classification acceptance and for human inspection for egdy case confidence values.

    Parameters
    ==========
    input_filepath: pathlib.Path
        input paper_output filename path for statistics
    output_filepath: pathlib.Path
        file name path to save the JSON file
    threshold_acceptance: float
        threshold value to accept llm papertype
    threshold_inspection: float
        threshold value to filter papers required for human inspection

    Returns
    =======
    """

    data = read_output(bibcode=None, filename=input_path)

    logger.debug(f"Loaded data: {data}")

    # Validate data structure
    for bibcode, assessment in data.items():
        assert isinstance(assessment, list), f"Assessment for {bibcode} should be a list."
        for mission_item in assessment:
            assert isinstance(
                mission_item, dict
            ), f"Each mission_item should be a dict, got {type(mission_item)} for bibcode {bibcode}."

    # Build Pandas DataFrame
    try:
        df = pd.DataFrame(
            [
                (mission.lower(), classification[0].lower(), classification[1], bibcode)
                for bibcode, assessment in data.items()
                for mission_item in assessment
                for mission, classification in mission_item.items()
            ],
            columns=["mission", "papertype", "llm_confidences", "bibcode"],
        )

    except Exception as e:
        logger.error(f"Error during operation DataFrame creation: {e}")
        raise

    df = df.sort_values(["mission", "papertype"]).reset_index(drop=True)

    # grouping DF and aggregate other properies
    grouped_df = group_by_agg("llm_confidences", threshold_acceptance, threshold_inspection, df)

    # Write the statistics summary
    write_stats(output_path, threshold_acceptance, threshold_inspection, grouped_df)


def group_by_agg(confidence_name: str, threshold_acceptance: float, threshold_inspection: float, df: pd.DataFrame):
    """Group by mission and papertype and aggregate other properties"""

    def inspection_condition(confidence: list[float, float]):
        return (max(confidence) >= threshold_inspection) and (max(confidence) < threshold_acceptance)

    def acceptance_condition(confidence: list[float, float]):
        return max(confidence) >= threshold_acceptance

    grouped_df = (
        df.fillna(0)
        .groupby(["mission", "papertype"])
        .agg(
            total_count=("mission", "size"),
            accepted_count=(confidence_name, lambda x: sum(1 for i in x if max(i) >= threshold_acceptance)),
            accepted_bibcodes=(
                "bibcode",
                lambda x: list(
                    set(
                        [
                            df.loc[i, "bibcode"]
                            for i in range(len(x))
                            if acceptance_condition(df.loc[x.index[i], confidence_name])
                        ]
                    )
                ),
            ),
            inspection_count=(
                confidence_name,
                lambda x: sum(1 for i in x if inspection_condition(i)),
            ),
            inspection_bibcodes=(
                "bibcode",
                lambda x: list(
                    set(
                        [
                            df.loc[i, "bibcode"]
                            for i in range(len(x))
                            if inspection_condition(df.loc[x.index[i], confidence_name])
                        ]
                    )
                ),
            ),
        )
        .reset_index()
    )

    return grouped_df


def write_stats(output_path, threshold_acceptance, threshold_inspection, grouped_df):
    """
    Parameters
    ==========
    output_path: pathlib.Path
        filename path to save the stats results
    threshold_acceptance: float
        threshold value to accept llm papertype
    threshold_inspection: float
        threshold value to filter papers required for human inspection
    grouped_df: pd.DataFrame

    Returns
    =======
    """
    logger.info(
        "Production counts by LLM Mission and Paper Type:\n"
        + grouped_df[["mission", "papertype", "total_count", "accepted_count", "inspection_count"]].to_string(
            index=False
        )
    )

    list_of_dicts = grouped_df.to_dict(orient="records")
    list_of_dicts.insert(
        0, {"threshold_acceptance": threshold_acceptance, "threshold_inspection": threshold_inspection}
    )

    # writing the stats table JSON
    if not os.path.exists(output_path):
        save_json_file(
            output_path,
            list_of_dicts,
        )
    else:
        raise FileExistsError(
            f"{output_path} already exists. Are you sure you want to overwrite the file? Choose a different name for the output in 'bibcat_config.yaml', if you want to keep the existing file"
        )

    logger.info(f"bibcode lists for both acceptance and inspection were generated in {output_path}")
