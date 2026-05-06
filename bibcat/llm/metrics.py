from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from bibcat import config
from bibcat.llm.evaluate import build_eval_data_for_run
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

POSITIVE_LABEL = config.llms.papertypes[0]  # SCIENCE
NEGATIVE_LABEL = config.llms.papertypes[1]  # NONSCIENCE
IGNORED_RAW_LABEL = "IGNORED"
LLM_METADATA_KEYS = {"confidence", "probability"}


@dataclass(frozen=True)
class MissionSample:
    bibcode: str
    mission: str
    human_raw: str
    llm_raw: str
    human_label: str
    llm_label: str
    mission_in_text: bool


@dataclass(frozen=True)
class LlmRunPrediction:
    papertype: str
    confidence: list[float] | None


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
            out[normalized] = label
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


def extract_mission_in_text(df_rows: list[dict[str, Any]] | None) -> dict[str, bool]:
    """Extract mission-in-text flags by mission.

    Parameters
    ----------
    df_rows : list[dict[str, Any]] or None
        Rows from the ``df`` field in evaluation summary output.

    Returns
    -------
    dict[str, bool]
        Mapping of normalized mission names to mission-in-text flags.
    """
    out: dict[str, bool] = {}
    for row in df_rows or []:
        mission = normalize_mission(row.get("llm_mission"))
        if mission is not None:
            out[mission] = bool(row.get("mission_in_text", False))
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
            out[mission] = prob
    return out


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers.

    Parameters
    ----------
    numerator : float
        Numerator value.
    denominator : float
        Denominator value.

    Returns
    -------
    float
        ``numerator / denominator`` when denominator is non-zero, else ``0.0``.
    """
    return numerator / denominator if denominator else 0.0


def build_entry(sample: MissionSample) -> dict[str, str]:
    """Build a case-entry payload (confusion matrix entry) from a mission sample.

    Parameters
    ----------
    sample : MissionSample
        Mission-level sample record.

    Returns
    -------
    dict[str, str]
        Minimal dictionary used in TP/TN/FP/FN case lists.
    """
    return {
        "bibcode": sample.bibcode,
        "mission": sample.mission,
        "human_raw": sample.human_raw,
        "llm_raw": sample.llm_raw,
    }


def extract_samples_and_summary(
    data: dict[str, dict[str, Any]], missions: list[str]
) -> tuple[list[MissionSample], dict[str, Any]]:
    """Extract mission samples and summary counts from evaluation data.

    Parameters
    ----------
    data : dict[str, dict[str, Any]]
        Evaluation data keyed by bibcode.
    missions : list[str]
        Missions to evaluate.

    Returns
    -------
    tuple[list[MissionSample], dict[str, Any]]
        Mission-level samples and high-level summary counts used by metrics.
    """
    normalized_missions = normalize_missions(missions)
    mission_set = set(normalized_missions)

    samples: list[MissionSample] = []
    n_human_callouts = 0
    n_llm_callouts = 0
    n_missing_paper_sources = 0
    n_missing_output_bibcodes = 0
    human_llm_missions_seen: set[str] = set()
    n_human_llm_mission_callouts = 0
    n_human_llm_hallucination = 0

    for bibcode, item in data.items():
        if has_no_paper_source(item):
            n_missing_paper_sources += 1
            continue

        human = normalize_human_labels(item.get("human"))
        n_human_callouts += sum(
            1 for mission in human if mission in mission_set and to_binary_from_raw(human[mission]) == POSITIVE_LABEL
        )

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
        "n_missing_paper_sources": n_missing_paper_sources,
        "n_missing_output_bibcodes": n_missing_output_bibcodes,
        "human_llm_missions": sorted(human_llm_missions_seen),
        "n_human_llm_mission_callouts": n_human_llm_mission_callouts,
        "n_human_llm_hallucination": n_human_llm_hallucination,
    }
    return samples, summary


def compute_confusion(samples: list[MissionSample]) -> dict[str, Any]:
    """Compute confusion-matrix counts and case lists.

    Parameters
    ----------
    samples : list[MissionSample]
        Mission-level binary-labeled samples.

    Returns
    -------
    dict[str, Any]
        Confusion counts (``tp``, ``tn``, ``fp``, ``fn``) and corresponding
        case-entry lists.
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
    """Compute binary classification metrics from confusion counts.

    Parameters
    ----------
    confusion : dict[str, Any]
        Confusion payload containing ``tp``, ``tn``, ``fp``, and ``fn``.

    Returns
    -------
    dict[str, float or int]
        Confusion counts and derived scores including TPR/recall, precision,
        F1, and accuracy.
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
        "f1": f1,
        "accuracy": accuracy,
    }


def extract_eval_data(data: dict[str, dict[str, Any]], missions: list[str]) -> dict[str, Any]:
    """Extract confusion-matrix inputs and summary metrics for one eval set.

    Parameters
    ----------
    data : dict[str, dict[str, Any]]
        Evaluation data keyed by bibcode.
    missions : list[str]
        Missions to evaluate.

    Returns
    -------
    dict[str, Any]
        Combined summary containing labels, confusion-case payloads, and
        computed metric values.
    """
    samples, summary = extract_samples_and_summary(data, missions)
    confusion = compute_confusion(samples)
    metrics = compute_metrics(confusion)
    return {
        **summary,
        "human_labels": [sample.human_label for sample in samples],
        "llm_labels": [sample.llm_label for sample in samples],
        "metrics": metrics,
        "fp_bibcodes": confusion["fp_cases"],
        "fn_bibcodes": confusion["fn_cases"],
        "tp_bibcodes": confusion["tp_cases"],
        "tn_bibcodes": confusion["tn_cases"],
    }


def get_llm_run_or_empty(llm_runs: list[dict[str, Any]], run_index: int) -> dict[str, Any]:
    """Get one run item or an empty default payload.

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
        Mapping of normalized mission names to prediction payloads.
    """
    predictions: dict[str, LlmRunPrediction] = {}
    for mission_item in run_item.get("missions", []) or []:
        mission = normalize_mission(mission_item.get("mission"))
        if mission is None:
            continue
        papertype = str(mission_item.get("papertype", IGNORED_RAW_LABEL))
        confidence = mission_item.get("confidence")
        conf = confidence if isinstance(confidence, list) else None
        predictions[mission] = LlmRunPrediction(papertype=papertype, confidence=conf)
    return predictions


def build_samples_for_run(
    eval_data: dict[str, dict[str, Any]],
    missions: list[str],
) -> list[MissionSample]:
    """Build mission samples for a single run-specific evaluation snapshot.

    Parameters
    ----------
    eval_data : dict[str, dict[str, Any]]
        Run-specific evaluation data keyed by bibcode. Expected to follow the same
        shape as summary output entries produced during evaluation.
    missions : list[str]
        Missions to evaluate.

    Returns
    -------
    list[MissionSample]
        Flattened mission-level samples used for confusion-matrix metric
        calculations.
    """
    normalized_missions = normalize_missions(missions)
    samples: list[MissionSample] = []

    for bibcode, item in eval_data.items():
        if has_no_paper_source(item):
            continue

        human = normalize_human_labels(item.get("human"))

        if has_no_mission_output(item):
            llm_labels: dict[str, str] = {}
        else:
            llm_labels = extract_llm_labels(item.get("llm"))

        for mission in normalized_missions:
            human_raw = human.get(mission, IGNORED_RAW_LABEL)
            llm_raw = llm_labels.get(mission, IGNORED_RAW_LABEL)

            samples.append(
                MissionSample(
                    bibcode=bibcode,
                    mission=mission,
                    human_raw=human_raw,
                    llm_raw=llm_raw,
                    human_label=to_binary_from_raw(human_raw),
                    llm_label=to_binary_from_raw(llm_raw),
                    mission_in_text=False,
                )
            )

    return samples


def aggregate_metrics_across_runs(per_run_metrics: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    """Aggregate metric mean and standard deviation across runs.

    Parameters
    ----------
    per_run_metrics : list[dict[str, float or int]]
        Per-run metric dictionaries.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from metric name to ``{"mean": float, "std": float}``.
    """
    if not per_run_metrics:
        return {}
    metric_names = list(per_run_metrics[0].keys())
    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(run_metrics[metric_name]) for run_metrics in per_run_metrics]
        aggregated[metric_name] = {"mean": float(mean(values)), "std": float(pstdev(values))}
    return aggregated


def compute_run_coverage(
    eval_data: dict[str, dict[str, Any]],
    llm_runs_data: dict[str, list[dict[str, Any]]],
    n_runs: int,
) -> float:
    """Compute fraction of expected run outputs that are present.

    Parameters
    ----------
    eval_data : dict[str, dict[str, Any]]
        Evaluation data keyed by bibcode.
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    n_runs : int
        Total number of runs considered.

    Returns
    -------
    float
        Ratio of present run outputs to expected run outputs for bibcodes with
        available source papers.
    """
    total = 0
    present = 0
    for bibcode, item in eval_data.items():
        if has_no_paper_source(item):
            continue
        llm_runs = llm_runs_data.get(bibcode, [])
        for run_index in range(n_runs):
            total += 1
            if run_index < len(llm_runs):
                present += 1
    return safe_divide(present, total)


def evaluate_multiple_llm_runs(
    eval_data: dict[str, dict[str, Any]],
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
) -> dict[str, Any]:
    """Evaluate aggregate confusion-matrix metrics across LLM runs.

    For each run index, this function builds an in-memory run-specific evaluation
    snapshot and computes confusion-matrix metrics from that run only.

    Parameters
    ----------
    eval_data : dict[str, dict[str, Any]]
        Existing evaluation data keyed by bibcode. This is used to define the
        bibcode set for evaluation and run-coverage calculation.
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    missions : list[str]
        Missions to evaluate.

    Returns
    -------
    dict[str, Any]
        Aggregate multi-run metric summary including run count, run coverage,
        per-run metrics, and mean/std aggregate metrics.
    """
    n_runs = max((len(runs) for runs in llm_runs_data.values()), default=0)
    per_run_metrics: list[dict[str, float | int]] = []
    eval_bibcodes = list(eval_data.keys())

    for run_index in range(n_runs):
        logger.info(f"Compute confusion matrix metrics for run {run_index + 1}/{n_runs}...")
        eval_data_for_run = build_eval_data_for_run(
            llm_runs_data=llm_runs_data,
            run_index=run_index,
            bibcodes=eval_bibcodes,
        )
        samples = build_samples_for_run(
            eval_data=eval_data_for_run,
            missions=missions,
        )
        confusion = compute_confusion(samples)
        metrics = compute_metrics(confusion)
        per_run_metrics.append(metrics)

    return {
        "n_runs": n_runs,
        "run_coverage": compute_run_coverage(eval_data, llm_runs_data, n_runs),
        "aggregate_metrics": aggregate_metrics_across_runs(per_run_metrics),
        "per_run_metrics": per_run_metrics,
    }
