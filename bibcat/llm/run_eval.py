from dataclasses import dataclass
from typing import Any

from bibcat import config
from bibcat.data.build_dataset import load_source_dataset

POSITIVE_LABEL = config.llms.papertypes[0]  # SCIENCE
NEGATIVE_LABEL = config.llms.papertypes[1]  # NONSCIENCE
IGNORED_RAW_LABEL = "IGNORED"
LLM_METADATA_KEYS = {"confidence", "mission_probability"}


@dataclass(frozen=True)
class LlmPrediction:
    """Normalized raw LLM prediction for one mission in one run."""

    papertype: str
    confidence: list[float] | None


@dataclass(frozen=True)
class SingleRunEvaluation:
    """Evaluation inputs for one paper in a single run."""

    bibcode: str
    human_labels: dict[str, str]
    llm_predictions: dict[str, LlmPrediction]
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


def to_binary_from_raw(raw: str | None) -> str:
    """Convert a raw papertype to binary science/nonscience label.

    Maps raw labels through config.llms.map_papertypes, defaulting to nonscience
    for missing or unknown labels, then classifies as either POSITIVE_LABEL or
    NEGATIVE_LABEL.

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
    # Map raw label through config, default to nonscience if missing/unknown
    if raw is None:
        mapped = NEGATIVE_LABEL
    else:
        mapped = config.llms.map_papertypes.get(str(raw).lower(), "nonscience")
        mapped = str(mapped).upper()
    return POSITIVE_LABEL if mapped == POSITIVE_LABEL else NEGATIVE_LABEL


def has_no_paper_source(item: dict[str, Any]) -> bool:
    """Return True if the eval item has no paper source."""
    return "No paper source found" in str(item.get("error") or "")


def has_no_mission_output(item: dict[str, Any]) -> bool:
    """Return True if the eval item has no mission output."""
    return "No mission output found" in str(item.get("error") or "")


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


def build_source_lookup(source_dataset: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Build a bibcode-keyed lookup from a source paper dataset.

    Parameters
    ----------
    source_dataset : list[dict[str, Any]], optional
        Source paper dataset rows. If not provided, loads from disk via
        ``load_source_dataset()``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from bibcode to source paper row.
    """
    if source_dataset is None:
        source_dataset = load_source_dataset()
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


def extract_llm_run_predictions(run_item: dict[str, Any]) -> dict[str, LlmPrediction]:
    """Extract normalized mission predictions from one run item.

    Parameters
    ----------
    run_item : dict[str, Any]
        One run response containing a ``missions`` list.

    Returns
    -------
    dict[str, LlmPrediction]
        Mapping of normalized mission names to prediction records.
    """
    predictions: dict[str, LlmPrediction] = {}
    for mission_item in run_item.get("missions", []) or []:
        mission = normalize_mission(mission_item.get("mission"))
        if mission is None:
            continue
        papertype = str(mission_item.get("papertype", IGNORED_RAW_LABEL))
        confidence = mission_item.get("confidence")
        # Handle missing confidence data by returning None
        conf = list(confidence) if isinstance(confidence, list) else None
        predictions[mission] = LlmPrediction(papertype=papertype, confidence=conf)
    return predictions


def build_run_paper_evaluations(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    bibcodes: list[str],
    run_index: int,
    source_lookup: dict[str, dict[str, Any]],
) -> list[SingleRunEvaluation]:
    """Build raw run-evaluation inputs for one run across bibcodes.

    This is the main evaluation data assembly function. It constructs a list of
    ``SingleRunEvaluation`` objects by pairing LLM predictions from a specific run
    with human classifications from a source dataset. Each evaluation object
    contains normalized mission labels, papertype predictions, and metadata
    about data availability (source present, output present).

    The function handles three evaluation scenarios per bibcode:

    1. **Missing source**: Bibcode has no entry in the source dataset.
       Returns evaluation with ``has_source=False``, empty labels and predictions.

    2. **Missing LLM output**: Bibcode in source but no LLM run data for the
       requested ``run_index``. Returns evaluation with ``has_output=False``.

    3. **Complete**: Both source and LLM output available. Extracts and normalizes
       human mission labels and LLM run predictions for comparison.

    This output is typically used downstream by metrics computation functions
    (confusion matrix, ROC) to assess LLM classification performance.

    Parameters
    ----------
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode. Each value is a list of run
        responses, where each run contains mission predictions with confidence.
    bibcodes : list[str]
        Explicit bibcode list to evaluate. Only these bibcodes will be
        included in the output, in the order specified.
    run_index : int
        Zero-based run index to extract from ``llm_runs_data``. If the bibcode
        has fewer runs than requested, a placeholder with no missions is
        created, resulting in empty LLM predictions and ``has_output=False``.
    source_lookup : dict[str, dict[str, Any]]
        Source paper lookup keyed by bibcode. Typically built from the combined
        dataset and contains human mission classifications under the
        ``class_missions`` field.

    Returns
    -------
    list[SingleRunEvaluation]
        Run-specific evaluation inputs for the requested bibcodes. One entry per
        bibcode in order. Each entry contains normalized human labels and LLM
        predictions, plus flags indicating data availability (``has_source``,
        ``has_output``).

    Examples
    --------
    >>> source_dataset = [
    ...     {
    ...         "bibcode": "2023Natur.616..266L",
    ...         "class_missions": {"HST": "SCIENCE", "JWST": "MENTION"}
    ...     }
    ... ]
    >>> source_lookup = build_source_lookup(source_dataset)
    >>> llm_output = {
    ...     "2023Natur.616..266L": [
    ...         {
    ...             "missions": [
    ...                 {"mission": "HST", "papertype": "SCIENCE", "confidence": [0.9, 0.1]},
    ...                 {"mission": "JWST", "papertype": "MENTION", "confidence": [0.3, 0.7]}
    ...             ]
    ...         }
    ...     ]
    ... }
    >>> evals = build_run_paper_evaluations(
    ...     llm_output, ["2023Natur.616..266L"], run_index=0, source_lookup=source_lookup
    ... )
    >>> evals[0].has_source, evals[0].has_output
    (True, True)
    >>> evals[0].human_labels
    {'HST': 'SCIENCE', 'JWST': 'MENTION'}

    Notes
    -----
    Missing mission names (None after normalization) are skipped silently.
    Human label extraction is case-insensitive and mission keys are normalized
    to uppercase. LLM predictions use the same normalization for consistent
    comparison.
    """
    evaluations: list[SingleRunEvaluation] = []
    for bibcode in bibcodes:
        # Check if bibcode exists in source paper dataset
        paper = source_lookup.get(bibcode)
        if paper is None:
            # Missing source: record empty evaluation with has_source=False
            evaluations.append(
                SingleRunEvaluation(
                    bibcode=bibcode,
                    human_labels={},
                    llm_predictions={},
                    has_source=False,
                    has_output=False,
                )
            )
            continue

        # Extract LLM predictions for this run; return empty dict if run not available
        llm_runs = llm_runs_data.get(bibcode, [])
        run_item = llm_runs[run_index] if run_index < len(llm_runs) else {"missions": []}
        llm_predictions = extract_llm_run_predictions(run_item)

        evaluations.append(
            SingleRunEvaluation(
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
        # Skip bibcodes not in source dataset
        if bibcode not in source_bibcodes:
            continue
        llm_runs = llm_runs_data.get(bibcode, [])
        # Count expected vs. present run outputs for this bibcode
        for run_index in range(n_runs):
            total += 1
            if run_index < len(llm_runs):
                present += 1
    return present / total if total else 0.0
