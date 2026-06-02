from dataclasses import dataclass
from typing import Any

from bibcat import config

POSITIVE_LABEL = config.llms.papertypes[0]  # SCIENCE
NEGATIVE_LABEL = config.llms.papertypes[1]  # NONSCIENCE
IGNORED_RAW_LABEL = "IGNORED"
LLM_METADATA_KEYS = {"confidence", "probability"}


@dataclass(frozen=True)
class LlmRunPrediction:
    """Normalized raw LLM prediction for one mission in one run."""

    papertype: str
    confidence: list[float] | None


@dataclass(frozen=True)
class RunPaperEvaluation:
    """Run-specific evaluation inputs for one bibcode."""

    bibcode: str
    human_labels: dict[str, str]
    llm_predictions: dict[str, LlmRunPrediction]
    has_source: bool
    has_output: bool


def normalize_mission(name: str | None) -> str | None:
    """Normalize a mission name to uppercase.

    Parameters
    ----------
    name : str or None
        Mission name to normalize.

    Returns
    -------
    str or None
        Uppercased mission name with surrounding whitespace removed, or
        ``None`` if ``name`` is ``None``.
    """
    if name is None:
        return None
    return name.strip().upper()


def normalize_missions(missions: list[str]) -> list[str]:
    """Normalize a list of mission names.

    Parameters
    ----------
    missions : list[str]
        Mission names to normalize.

    Returns
    -------
    list[str]
        Normalized mission names.
    """
    return [m for m in (normalize_mission(mission) for mission in missions) if m is not None]


def map_papertype(raw: str | None) -> str:
    """Map a raw papertype label to configured output label.

    Parameters
    ----------
    raw : str or None
        Raw papertype label.

    Returns
    -------
    str
        Uppercased mapped papertype. Defaults to nonscience mapping when
        ``raw`` is missing or unknown.
    """
    if raw is None:
        return NEGATIVE_LABEL
    mapped = config.llms.map_papertypes.get(str(raw).lower(), "nonscience")
    return str(mapped).upper()


def to_binary_from_raw(raw: str | None) -> str:
    """Convert a raw papertype to binary science/nonscience label.

    Parameters
    ----------
    raw : str or None
        Raw papertype label.

    Returns
    -------
    str
        ``POSITIVE_LABEL`` for science-class mapping, otherwise
        ``NEGATIVE_LABEL``.
    """
    mapped = map_papertype(raw)
    return POSITIVE_LABEL if mapped == POSITIVE_LABEL else NEGATIVE_LABEL


def normalize_human_labels(human: dict[str, str] | None) -> dict[str, str]:
    """Normalize human labels to uppercased mission keys.

    Parameters
    ----------
    human : dict[str, str] or None
        Human mission-to-papertype mapping.

    Returns
    -------
    dict[str, str]
        Human mapping with normalized mission keys.
    """
    out: dict[str, str] = {}
    for mission, label in (human or {}).items():
        normalized = normalize_mission(mission)
        if normalized is not None:
            out[normalized] = str(label)
    return out


def extract_llm_labels(llm_list: list[dict[str, Any]] | None) -> dict[str, str]:
    """Extract mission-to-papertype labels from llm summary entries.

    Parameters
    ----------
    llm_list : list[dict[str, Any]] or None
        LLM summary output list, where each item includes mission labels and
        metadata keys.

    Returns
    -------
    dict[str, str]
        Mapping of normalized mission names to raw LLM papertypes.
    """
    out: dict[str, str] = {}
    for item in llm_list or []:
        for key, value in item.items():
            if key not in LLM_METADATA_KEYS:
                mission = normalize_mission(key)
                if mission is not None:
                    out[mission] = str(value)
    return out


def extract_mission_confidence_map(mission_conf_rows: list[dict[str, Any]] | None) -> dict[str, list[float]]:
    """Extract mission confidence vectors by mission.

    Parameters
    ----------
    mission_conf_rows : list[dict[str, Any]] or None
        Rows from the ``mission_conf`` field in evaluation summary output.

    Returns
    -------
    dict[str, list[float]]
        Mapping of normalized mission names to confidence vectors ordered as
        ``[p_science, p_nonscience]``.
    """
    out: dict[str, list[float]] = {}
    for row in mission_conf_rows or []:
        mission = normalize_mission(row.get("llm_mission"))
        prob = row.get("prob_papertype")
        if mission is not None and isinstance(prob, list):
            out[mission] = list(prob)
    return out


def has_no_mission_output(item: dict[str, Any]) -> bool:
    """Check whether an eval item indicates missing mission output.

    Parameters
    ----------
    item : dict[str, Any]
        Eval-data entry for one bibcode.

    Returns
    -------
    bool
        ``True`` if the entry error indicates no mission output.
    """
    error = str(item.get("error") or "").strip()
    return "No mission output found" in error


def has_no_paper_source(item: dict[str, Any]) -> bool:
    """Check whether an eval item indicates missing paper source.

    Parameters
    ----------
    item : dict[str, Any]
        Eval-data entry for one bibcode.

    Returns
    -------
    bool
        ``True`` if the entry error indicates no source paper.
    """
    error = str(item.get("error") or "").strip()
    return "No paper source found" in error


def build_source_lookup(source_dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a source-paper lookup keyed by bibcode.

    Parameters
    ----------
    source_dataset : list[dict[str, Any]]
        Source paper dataset rows.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from bibcode to source paper row.
    """
    return {str(item["bibcode"]): item for item in source_dataset if "bibcode" in item}


def extract_human_labels_from_source(paper: dict[str, Any] | None) -> dict[str, str]:
    """Extract normalized human mission labels from one source paper.

    Parameters
    ----------
    paper : dict[str, Any] or None
        Source paper row.

    Returns
    -------
    dict[str, str]
        Mapping from normalized mission name to raw papertype.
    """
    out: dict[str, str] = {}
    class_missions = (paper or {}).get("class_missions") or {}
    for mission, value in class_missions.items():
        normalized = normalize_mission(mission)
        if normalized is None:
            continue
        if isinstance(value, dict):
            papertype = value.get("papertype")
        else:
            papertype = value
        if papertype is not None:
            out[normalized] = str(papertype)
    return out


def get_llm_run_or_empty(llm_runs: list[dict[str, Any]], run_index: int) -> dict[str, Any]:
    """Get one run item or an empty default run record.

    Parameters
    ----------
    llm_runs : list[dict[str, Any]]
        Run outputs for one bibcode.
    run_index : int
        Zero-based run index.

    Returns
    -------
    dict[str, Any]
        Run output at ``run_index`` if available, otherwise ``{"missions": []}``.
    """
    if run_index < len(llm_runs):
        return llm_runs[run_index]
    return {"missions": []}


def extract_llm_run_predictions(run_item: dict[str, Any]) -> dict[str, LlmRunPrediction]:
    """Extract normalized mission predictions from one run item.

    Parameters
    ----------
    run_item : dict[str, Any]
        One run response containing a ``missions`` list.

    Returns
    -------
    dict[str, LlmRunPrediction]
        Mapping of normalized mission names to prediction records.
    """
    predictions: dict[str, LlmRunPrediction] = {}
    for mission_item in run_item.get("missions", []) or []:
        mission = normalize_mission(mission_item.get("mission"))
        if mission is None:
            continue
        papertype = str(mission_item.get("papertype", IGNORED_RAW_LABEL))
        confidence = mission_item.get("confidence")
        conf = list(confidence) if isinstance(confidence, list) else None
        predictions[mission] = LlmRunPrediction(papertype=papertype, confidence=conf)
    return predictions


def build_run_paper_evaluations(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    bibcodes: list[str],
    run_index: int,
    source_lookup: dict[str, dict[str, Any]],
) -> list[RunPaperEvaluation]:
    """Build raw run-evaluation inputs for one run across bibcodes.

    Parameters
    ----------
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    bibcodes : list[str]
        Explicit bibcode roster to evaluate.
    run_index : int
        Zero-based run index.
    source_lookup : dict[str, dict[str, Any]]
        Source paper lookup keyed by bibcode.

    Returns
    -------
    list[RunPaperEvaluation]
        Run-specific evaluation inputs for the requested bibcodes.
    """
    evaluations: list[RunPaperEvaluation] = []
    for bibcode in bibcodes:
        paper = source_lookup.get(bibcode)
        if paper is None:
            evaluations.append(
                RunPaperEvaluation(
                    bibcode=bibcode,
                    human_labels={},
                    llm_predictions={},
                    has_source=False,
                    has_output=False,
                )
            )
            continue

        run_item = get_llm_run_or_empty(llm_runs_data.get(bibcode, []), run_index)
        llm_predictions = extract_llm_run_predictions(run_item)
        evaluations.append(
            RunPaperEvaluation(
                bibcode=bibcode,
                human_labels=extract_human_labels_from_source(paper),
                llm_predictions=llm_predictions,
                has_source=True,
                has_output=bool(llm_predictions),
            )
        )
    return evaluations


def compute_run_coverage(
    bibcodes: list[str],
    llm_runs_data: dict[str, list[dict[str, Any]]],
    n_runs: int,
    source_bibcodes: set[str],
) -> float:
    """Compute fraction of expected run outputs that are present.

    Parameters
    ----------
    bibcodes : list[str]
        Bibcode roster to evaluate.
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    n_runs : int
        Total number of runs considered.
    source_bibcodes : set[str]
        Bibcodes that are present in the source dataset.

    Returns
    -------
    float
        Ratio of present run outputs to expected run outputs for bibcodes with
        available source papers.
    """
    total = 0
    present = 0
    for bibcode in bibcodes:
        if bibcode not in source_bibcodes:
            continue
        llm_runs = llm_runs_data.get(bibcode, [])
        for run_index in range(n_runs):
            total += 1
            if run_index < len(llm_runs):
                present += 1
    return present / total if total else 0.0
