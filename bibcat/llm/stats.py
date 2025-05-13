import pathlib

import pandas as pd

from bibcat import config
from bibcat.llm.io import read_output
from bibcat.utils.logger_config import setup_logger
from bibcat.utils.utils import save_json_file

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def inconsistent_classifications(input_path: str | pathlib.Path, output_path: str | pathlib.Path):
    """Save falsely classified bibcodes to a json file

    This code will check if llm classification is different from human classification
    or incorrectly ignore the mission and save the results to a json file.

    Parameters
    ----------
    input_path: str | pathlib.Path
        Input paper_output file name/path for statistics
    output_path: str | pathlib.Path
        File name/path to save the JSON file

    Returns
    =======
    None
    """

    data = read_output(bibcode=None, filename=input_path)
    logger.debug(f"Loaded data: {data}")

    results = {}
    matched_missions = 0
    for bibcode, item in data.items():
        if "error" not in item:
            human = item.get("human", {})
            llm_entries = item.get("llm", [])

            errors = {}
            for mission, human_label in human.items():
                # Check if mission exists at all in any LLM item
                mission_in_llm = any(mission in llm for llm in llm_entries)
                # Check if mission has the same label in any LLM item
                match_found = any(llm.get(mission) == human_label for llm in llm_entries if mission in llm)

                # Check if any LLM item assigns SCIENCE
                llm_science_assigned = any(llm.get(mission) == "SCIENCE" for llm in llm_entries if mission in llm)

                if not mission_in_llm:
                    if human_label == "SCIENCE":
                        errors[mission] = "false_negative_because_ignored"
                    else:
                        errors[mission] = "ignored"
                elif match_found:
                    matched_missions += 1
                elif not match_found:
                    if human_label == "SCIENCE":
                        errors[mission] = "false_negative"
                    elif llm_science_assigned:  # only count as false positive if LLM predicted SCIENCE
                        errors[mission] = "false_positive"
                # If matched, do not include

            if errors:
                results[bibcode] = {
                    "errors": errors,
                    "human": human,
                    "llm": llm_entries,
                    "missions_not_in_text": item.get("hallucinated_missions", []),
                }
    summary_counts = audit_summary(results)
    summary_counts = {"n_total_bibcodes": len(data), **summary_counts}
    # Add the summary to the top of the dictionary
    results_with_summary = {
        "summary_counts": summary_counts,
        "bibcodes": results,
    }

    save_json_file(output_path, results_with_summary, indent=2)


def audit_summary(audit_results: dict) -> dict[str, int]:
    """Create the summary of the inconsistent classifications

    Parameters
    ==========
    audit_results: dict
        the breakdown bibcode list of inconsistent llm classifications

    Returns
    =======
    summary_counts: dict[str, int]
        various count summary
    """

    # TODO - add stats grouped by mission
    summary_counts = {
        "mismatched_bibcodes": 0,
        "mismatched_missions": 0,
        "false_positive": 0,
        "false_negative": 0,
        "false_negative_because_ignored": 0,
        "ignored": 0,
    }

    for bibcode, entry in audit_results.items():
        error_dict = entry.get("errors", {})
        error_count = len(error_dict)

        if error_count > 0:
            summary_counts["mismatched_bibcodes"] += 1
            summary_counts["mismatched_missions"] += error_count

            for error_type in error_dict.values():
                if error_type in summary_counts:
                    summary_counts[error_type] += 1

    return summary_counts


def save_evaluation_stats(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    threshold_acceptance: float,
    threshold_inspection: float,
):
    """Generate acceptance and inspection statistics and identify classification inconsistencies between humans and the LLM for evaluation summary data

    This function performs the following actions:
         - **Creates a statistics file** containing:
            - **Accepted LLM Classifications**: Number of papers with classifications accepted by the LLM based on a specified threshold value for each combination of mission and paper type.
            - **Human Inspection Requirements**: Number of papers requiring human inspection
            - **Accepted Bibcodes**: Bibcodes corresponding to the accepted classifications.
            - **Inspection-Required Bibcodes**: Bibcodes that need human inspection due to ambiguous confidence values.

    Parameters
    ----------
    input_path: str | pathlib.Path
        Input paper_output file name/path for statistics
    output_path: str | pathlib.Path
        File name/path to save the JSON file
    threshold_acceptance: float
        Threshold value to accept LLM papertype
    threshold_inspection: float
        Threshold value to filter papers required for human inspection


    Returns
    -------
    None

    Raises
    ------
    Exception
        For any other exceptions that occur during DataFrame creation or file operations.

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
                if "df" in eval_item
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


def save_operation_stats(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    threshold_acceptance: float,
    threshold_inspection: float,
):
    """Generate acceptance and inspection statistics from operational classifications

    This function performs the following actions:
         - **Creates a statistics file** containing:
            - **Accepted LLM Classifications**: Number of papers with classifications accepted by the LLM based on a specified threshold value for each combination of mission and paper type.
            - **Human Inspection Requirements**: Number of papers requiring human inspection
            - **Accepted Bibcodes**: Bibcodes corresponding to the accepted classifications.
            - **Inspection-Required Bibcodes**: Bibcodes that need human inspection due to ambiguous confidence values.

    Parameters
    ----------
    input_path: str | pathlib.Path
        Input paper_output filename/path for statistics
    output_path: str | pathlib.Path
        File name/path to save the JSON file
    threshold_acceptance: float
        Threshold value to accept LLM papertype
    threshold_inspection: float
        Threshold value to filter papers required for human inspection

    Returns
    -------
    None

    Raises
    ------
    Exception
        For any other exceptions that occur during DataFrame creation or file operations.

    """

    data = read_output(bibcode=None, filename=input_path)
    logger.debug(f"The number of the loaded data: {len(data)}")

    # filter out bad data
    n_data = len(data)
    data = {b: a for b, a in data.items() for mi in a if "error" not in mi.keys() and mi["missions"]}
    logger.debug(f"Filtered {n_data - len(data)} bad data from {n_data} total entries.")

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
                [item["mission"].lower(), item["papertype"].lower(), item["confidence"], bibcode]
                for bibcode, assessment in data.items()
                for mission_item in assessment
                for item in mission_item["missions"]
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
    """Group DataFrame by mission and papertype and aggregate other properties.

    Parameters
    ----------
    confidence_name: str
        The Key name for LLM confidences, e.g, `"llm_confidences"` for `paper_output.json` or `"mean_llm_confidences"` for `summary_output.json`
    threshold_acceptance: float
        Threshold value to accept LLM papertype
    threshold_inspection: float
        Threshold value to filter papers required for human inspection
    df: pd.DataFrame
        Dataframe

    Returns
    -------
    pd.DataFrame
    """

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
    """Write the satistics into a JSON file.

    Parameters
    ----------
    output_path: pathlib.Path
        Filename path to save the stats results.
    threshold_acceptance: float
        Threshold value to accept LLM papertype.
    threshold_inspection: float
        Threshold value to filter papers required for human inspection.
    grouped_df: pd.DataFrame
        Grouped DataFrame

    Returns
    -------
    None
    """
    tsv_df = grouped_df[["mission", "papertype", "total_count", "accepted_count", "inspection_count"]]

    logger.info("Production counts by LLM Mission and Paper Type:\n" + tsv_df.to_string(index=False))
    # Write to an ascii file
    summary_file = (
        pathlib.Path(config.paths.output)
        / f"llms/openai_{config.llms.openai.model}/{config.llms.eval_stats_file}_t{config.llms.performance.threshold}.txt"
    )
    try:
        # Format and save to a text file with proper alignment
        with open(summary_file, "w") as f:
            f.write(tsv_df.to_string(index=False))
        print(f"Data successfully written to {summary_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # writing the stats table JSON
    list_of_dicts = grouped_df.to_dict(orient="records")
    list_of_dicts.insert(
        0, {"threshold_acceptance": threshold_acceptance, "threshold_inspection": threshold_inspection}
    )

    save_json_file(
        output_path,
        list_of_dicts,
    )

    logger.info(f"bibcode lists for both acceptance and inspection were generated in {output_path}")
