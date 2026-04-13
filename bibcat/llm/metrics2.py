from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import auc, roc_curve

from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

POSITIVE_LABEL = "SCIENCE"
NEGATIVE_LABEL = "NONSCIENCE"
IGNORED_RAW_LABEL = "IGNORED"
LLM_METADATA_KEYS = {"confidence", "probability"}


@dataclass(frozen=True)
class MissionSample:
    """Flattened evaluation sample for one bibcode-mission pair.

    Parameters
    ----------
    bibcode : str
        Paper identifier.
    mission : str
        Normalized mission name.
    human_raw : str
        Raw human label before papertype mapping.
    llm_raw : str
        Raw LLM label before papertype mapping.
    human_label : str
        Binary mapped human label. Either ``SCIENCE`` or ``NONSCIENCE``.
    llm_label : str
        Binary mapped LLM label. Either ``SCIENCE`` or ``NONSCIENCE``.
    mission_in_text : bool
        Whether the mission was found in text according to the LLM-side
        ``df`` entry for this mission.
    """

    bibcode: str
    mission: str
    human_raw: str
    llm_raw: str
    human_label: str
    llm_label: str
    mission_in_text: bool


def normalize_mission(name: str | None) -> str | None:
    """Normalize a mission name to canonical uppercase form.

    Parameters
    ----------
    name : str or None
        Mission name.

    Returns
    -------
    str or None
        Normalized mission name, or ``None`` if the input is ``None``.

    Examples
    --------
    >>> normalize_mission(" tess ")
    'TESS'
    >>> normalize_mission(None) is None
    True
    """
    if name is None:
        return None
    return name.strip().upper()


def has_no_mission_output(item: dict[str, Any]) -> bool:
    """Return whether an evaluation item indicates missing mission output.

    Parameters
    ----------
    item : dict
        Evaluation record for one bibcode.

    Returns
    -------
    bool
        ``True`` when the record contains a ``No mission output found`` error.

    Notes
    -----
    The error string may include additional text such as the bibcode, so this
    function uses substring matching rather than exact equality.
    """
    error = str(item.get("error") or "").strip()
    return "No mission output found" in error


def has_no_paper_source(item: dict[str, Any]) -> bool:
    """Return whether an evaluation item indicates missing paper source.

    Parameters
    ----------
    item : dict
        Evaluation record for one bibcode.

    Returns
    -------
    bool
        ``True`` when the record contains a ``No paper source found`` error.
    """
    error = str(item.get("error") or "").strip()
    return "No paper source found" in error


def map_papertype(raw: str | None) -> str:
    """Map a raw papertype into a configured allowed papertype.

    Parameters
    ----------
    raw : str or None
        Raw papertype label from human or LLM output.

    Returns
    -------
    str
        Uppercase mapped papertype.

    Notes
    -----
    Missing labels and unknown labels are treated as the negative class by
    mapping to ``NONSCIENCE``.
    """
    if raw is None:
        return NEGATIVE_LABEL

    mapped = config.llms.map_papertypes.get(str(raw).lower(), "nonscience")
    return str(mapped).upper()


def to_binary_label(mapped_label: str) -> str:
    """Reduce a mapped papertype to the binary evaluation label.

    Parameters
    ----------
    mapped_label : str
        Mapped label after ``map_papertype``.

    Returns
    -------
    str
        ``SCIENCE`` if the mapped label is positive, otherwise
        ``NONSCIENCE``.
    """
    return POSITIVE_LABEL if mapped_label == POSITIVE_LABEL else NEGATIVE_LABEL


def to_binary_from_raw(raw: str | None) -> str:
    """Map a raw label directly to the binary evaluation label.

    Parameters
    ----------
    raw : str or None
        Raw human or LLM papertype.

    Returns
    -------
    str
        ``SCIENCE`` or ``NONSCIENCE``.
    """
    return to_binary_label(map_papertype(raw))


def normalize_human_labels(human: dict[str, str] | None) -> dict[str, str]:
    """Normalize human mission keys.

    Parameters
    ----------
    human : dict or None
        Human mission-to-label mapping.

    Returns
    -------
    dict
        Dictionary of normalized mission names to raw human labels.
    """
    out: dict[str, str] = {}
    for mission, label in (human or {}).items():
        normalized = normalize_mission(mission)
        if normalized is not None:
            out[normalized] = label
    return out


def extract_llm_labels(llm_list: list[dict[str, Any]] | None) -> dict[str, str]:
    """Extract accepted LLM mission labels from the ``llm`` list.

    Parameters
    ----------
    llm_list : list of dict or None
        Accepted LLM verdict list. Each element is expected to contain one
        mission key plus metadata keys such as ``confidence`` and
        ``probability``.

    Returns
    -------
    dict
        Mapping of normalized mission name to raw LLM label.

    Notes
    -----
    Any key not present in ``LLM_METADATA_KEYS`` is treated as a mission key.
    This function assumes there is at most one accepted entry per mission per
    bibcode.
    """
    out: dict[str, str] = {}

    for item in llm_list or []:
        for key, value in item.items():
            if key not in LLM_METADATA_KEYS:
                mission = normalize_mission(key)
                if mission is not None:
                    out[mission] = value

    return out


def extract_mission_in_text(df_rows: list[dict[str, Any]] | None) -> dict[str, bool]:
    """Extract ``mission_in_text`` flags by mission from ``df``.

    Parameters
    ----------
    df_rows : list of dict or None
        Rows from the ``df`` field.

    Returns
    -------
    dict
        Mapping of normalized mission name to ``mission_in_text``.

    Notes
    -----
    When a mission appears multiple times in ``df``, the last row wins.
    """
    out: dict[str, bool] = {}

    for row in df_rows or []:
        mission = normalize_mission(row.get("llm_mission"))
        if mission is not None:
            out[mission] = bool(row.get("mission_in_text", False))

    return out


def extract_mission_confidence_map(
    mission_conf_rows: list[dict[str, Any]] | None,
) -> dict[str, list[float]]:
    """Extract probability vectors from ``mission_conf`` by mission.

    Parameters
    ----------
    mission_conf_rows : list of dict or None
        Rows from the ``mission_conf`` field.

    Returns
    -------
    dict
        Mapping of normalized mission name to ``prob_papertype`` values.

    Notes
    -----
    This function assumes the probability vector order is fixed:
    ``prob_papertype[0]`` is the ``SCIENCE`` score and
    ``prob_papertype[1]`` is the ``NONSCIENCE`` score.
    """
    out: dict[str, list[float]] = {}

    for row in mission_conf_rows or []:
        mission = normalize_mission(row.get("llm_mission"))
        prob = row.get("prob_papertype")
        if mission is not None and isinstance(prob, list):
            out[mission] = prob

    return out


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers.

    Parameters
    ----------
    numerator : float
        Numerator.
    denominator : float
        Denominator.

    Returns
    -------
    float
        ``numerator / denominator`` when the denominator is non-zero, otherwise
        ``0.0``.
    """
    return numerator / denominator if denominator else 0.0


def build_entry(sample: MissionSample) -> dict[str, str]:
    """Build a confusion-bucket output entry.

    Parameters
    ----------
    sample : MissionSample
        Flattened evaluation sample.

    Returns
    -------
    dict
        Output dictionary containing bibcode, mission, and raw labels.
    """
    return {
        "bibcode": sample.bibcode,
        "mission": sample.mission,
        "human_raw": sample.human_raw,
        "llm_raw": sample.llm_raw,
    }


def extract_samples_and_summary(
    data: dict[str, dict[str, Any]],
    missions: list[str],
) -> tuple[list[MissionSample], dict[str, Any]]:
    """Flatten nested evaluation data and collect summary bookkeeping.

    Parameters
    ----------
    data : dict
        Evaluation data keyed by bibcode.
    missions : list of str
        Missions to evaluate.

    Returns
    -------
    samples : list of MissionSample
        Flattened mission-by-mission evaluation records.
    summary : dict
        Summary bookkeeping values used in the final JSON output.

    Notes
    -----
    Evaluation is performed per mission per bibcode. Missing labels are treated
    as ``IGNORED`` at the raw level and become ``NONSCIENCE`` after mapping.

    The ``n_human_callouts`` and ``n_llm_callouts`` values are restricted to
    missions in the provided ``missions`` list.
    """

    normalized_missions = [normalize_mission(mission) for mission in missions if normalize_mission(mission) is not None]
    mission_set = set(normalized_missions)

    samples: list[MissionSample] = []

    n_human_callouts = 0
    n_llm_callouts = 0
    n_missing_output_bibcodes = 0

    human_llm_missions_seen: set[str] = set()
    n_human_llm_mission_callouts = 0
    n_human_llm_hallucination = 0

    for bibcode, item in data.items():
        if has_no_paper_source(item):
            logger.debug("Skipping bibcode %s because no paper source was found.", bibcode)
            continue

        human = normalize_human_labels(item.get("human"))
        n_human_callouts += sum(1 for mission in human if mission in mission_set)

        if has_no_mission_output(item):
            n_missing_output_bibcodes += 1
            llm = {}
            mission_in_text_map = {}
        else:
            llm = extract_llm_labels(item.get("llm"))
            mission_in_text_map = extract_mission_in_text(item.get("df"))
            n_llm_callouts += sum(1 for mission in llm if mission in mission_set)

        for mission in normalized_missions:
            human_has_mission = mission in human
            llm_has_mission = mission in llm

            if human_has_mission and llm_has_mission:
                human_llm_missions_seen.add(mission)
                n_human_llm_mission_callouts += 1

                if not mission_in_text_map.get(mission, False):
                    n_human_llm_hallucination += 1

            human_raw = human.get(mission, IGNORED_RAW_LABEL)
            llm_raw = llm.get(mission, IGNORED_RAW_LABEL)

            samples.append(
                MissionSample(
                    bibcode=bibcode,
                    mission=mission,
                    human_raw=human_raw,
                    llm_raw=llm_raw,
                    human_label=to_binary_from_raw(human_raw),
                    llm_label=to_binary_from_raw(llm_raw),
                    mission_in_text=mission_in_text_map.get(mission, False),
                )
            )

    summary = {
        "threshold": config.llms.performance.threshold,
        "n_bibcodes": len(data),
        "n_human_callouts": n_human_callouts,
        "n_llm_callouts": n_llm_callouts,
        "n_missing_output_bibcodes": n_missing_output_bibcodes,
        "human_llm_missions": sorted(human_llm_missions_seen),
        "n_human_llm_mission_callouts": n_human_llm_mission_callouts,
        "n_human_llm_hallucination": n_human_llm_hallucination,
    }

    return samples, summary


def compute_confusion(samples: list[MissionSample]) -> dict[str, Any]:
    """Compute binary confusion counts and bucketed sample entries.

    Parameters
    ----------
    samples : list of MissionSample
        Flattened binary evaluation samples.

    Returns
    -------
    dict
        Confusion counts and entry lists for each quadrant.
    """
    tp = tn = fp = fn = 0
    tp_cases: list[dict[str, str]] = []
    tn_cases: list[dict[str, str]] = []
    fp_cases: list[dict[str, str]] = []
    fn_cases: list[dict[str, str]] = []

    for sample in samples:
        true_is_positive = sample.human_label == POSITIVE_LABEL
        pred_is_positive = sample.llm_label == POSITIVE_LABEL
        entry = build_entry(sample)

        if true_is_positive and pred_is_positive:
            tp += 1
            tp_cases.append(entry)
        elif not true_is_positive and not pred_is_positive:
            tn += 1
            tn_cases.append(entry)
        elif not true_is_positive and pred_is_positive:
            fp += 1
            fp_cases.append(entry)
        else:
            fn += 1
            fn_cases.append(entry)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp_cases": tp_cases,
        "tn_cases": tn_cases,
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
    }


def compute_metrics(confusion: dict[str, Any]) -> dict[str, float | int]:
    """Compute confusion-matrix-derived binary metrics.

    Parameters
    ----------
    confusion : dict
        Confusion counts as returned by ``compute_confusion``.

    Returns
    -------
    dict
        Binary summary metrics including precision, recall, F1, and accuracy.
    """
    tp = int(confusion["tp"])
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    tnr = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, tn + fp)
    fnr = safe_divide(fn, tp + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": recall,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
    }


def extract_eval_data(data: dict[str, dict[str, Any]], missions: list[str]) -> dict[str, Any]:
    """Extract evaluation data and compute the binary summary output.

    Parameters
    ----------
    data : dict
        Evaluation data keyed by bibcode.
    missions : list of str
        Missions to evaluate.

    Returns
    -------
    dict
        Final binary metrics payload.

    Notes
    -----
    This is the main public summary API for the metrics module.
    """
    samples, summary = extract_samples_and_summary(data, missions)
    confusion = compute_confusion(samples)
    metrics = compute_metrics(confusion)

    output = {
        **summary,
        "metrics": metrics,
        "fp_bibcodes": confusion["fp_cases"],
        "fn_bibcodes": confusion["fn_cases"],
        "tp_bibcodes": confusion["tp_cases"],
        "tn_bibcodes": confusion["tn_cases"],
    }

    return output


def save_metrics_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save metrics output to a JSON file.

    Parameters
    ----------
    data : dict
        Metrics dictionary returned by ``extract_eval_data``.
    output_path : str or `~pathlib.Path`
        Destination path.

    Returns
    -------
    None
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def extract_roc_data(
    data: dict[str, dict[str, Any]],
    missions: list[str],
) -> tuple[list[str], list[list[float]], list[str]]:
    """Extract binary labels and confidence vectors for ROC/AUC analysis.

    Parameters
    ----------
    data : dict
        Evaluation data keyed by bibcode.
    missions : list of str
        Missions to evaluate.

    Returns
    -------
    human_labels : list of str
        Binary human labels for ROC evaluation.
    llm_confidences : list of list of float
        Probability vectors aligned with ``human_labels``.
    human_llm_missions : list of str
        Sorted missions called out by both human and LLM.

    Notes
    -----
    This function preserves the per-mission-per-bibcode flattening logic.
    For missing LLM output, the confidence vector defaults to ``[0.0, 1.0]``.
    """
    normalized_missions = [normalize_mission(mission) for mission in missions if normalize_mission(mission) is not None]

    human_labels: list[str] = []
    llm_confidences: list[list[float]] = []
    human_llm_missions_seen: set[str] = set()

    for bibcode, item in data.items():
        if has_no_paper_source(item):
            logger.debug("Skipping ROC extraction for %s due to missing paper source.", bibcode)
            continue

        human = normalize_human_labels(item.get("human"))

        if has_no_mission_output(item):
            llm = {}
            mission_conf_map = {}
        else:
            llm = extract_llm_labels(item.get("llm"))
            mission_conf_map = extract_mission_confidence_map(item.get("mission_conf"))

        for mission in normalized_missions:
            human_has_mission = mission in human
            llm_has_mission = mission in llm

            if human_has_mission and llm_has_mission:
                human_llm_missions_seen.add(mission)

            human_raw = human.get(mission, IGNORED_RAW_LABEL)
            human_labels.append(to_binary_from_raw(human_raw))

            if mission in mission_conf_map:
                llm_confidences.append(mission_conf_map[mission])
            else:
                llm_confidences.append([0.0, 1.0])

    return human_labels, llm_confidences, sorted(human_llm_missions_seen)


def prepare_roc_inputs(
    human_labels: list[str],
    llm_confidences: list[list[float]],
) -> tuple[list[int], list[list[float]], int, int]:
    """Prepare ROC inputs for binary evaluation.

    Parameters
    ----------
    human_labels : list of str
        Binary human labels.
    llm_confidences : list of list of float
        Confidence vectors aligned with ``human_labels``.

    Returns
    -------
    y_true : list of int
        Binary ground-truth vector where ``SCIENCE`` is 1 and
        ``NONSCIENCE`` is 0.
    llm_confidences : list of list of float
        Unmodified confidence vectors.
    n_classes : int
        Number of classes, always 2 for the binary summary path.
    n_samples : int
        Number of ROC samples.
    """
    y_true = [1 if label == POSITIVE_LABEL else 0 for label in human_labels]
    n_classes = 2
    n_samples = len(y_true)
    return y_true, llm_confidences, n_classes, n_samples


def get_roc_metrics(
    llm_confidences: list[list[float]],
    y_true: list[int],
    n_classes: int,
) -> tuple[list[float], list[float], list[float], float]:
    """Compute binary ROC curve and AUC.

    Parameters
    ----------
    llm_confidences : list of list of float
        Probability vectors for each sample.
    y_true : list of int
        Binary ground-truth vector.
    n_classes : int
        Number of classes. Must be 2.

    Returns
    -------
    fpr : list of float
        False positive rates.
    tpr : list of float
        True positive rates.
    thresholds : list of float
        ROC thresholds.
    roc_auc : float
        Area under the ROC curve.

    Raises
    ------
    ValueError
        If ``n_classes`` is not 2.
    """
    if n_classes != 2:
        raise ValueError(f"Binary ROC is required for this module, got n_classes={n_classes}.")

    science_scores = [float(conf[0]) for conf in llm_confidences]
    fpr, tpr, thresholds = roc_curve(y_true, science_scores)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)
