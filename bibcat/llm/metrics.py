from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from bibcat import config
from bibcat.data.build_dataset import load_source_dataset
from bibcat.llm.evaluate import build_eval_data_for_run
from bibcat.llm.run_eval import (
    IGNORED_RAW_LABEL,
    build_run_paper_evaluations,
    build_source_lookup,
    compute_run_coverage,
    extract_llm_labels,
    has_no_mission_output,
    has_no_paper_source,
    normalize_human_labels,
    normalize_mission,
    normalize_missions,
    to_binary_from_raw,
)
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__, level=config.logging.level)
POSITIVE_LABEL = config.llms.papertypes[0]  # SCIENCE
NEGATIVE_LABEL = config.llms.papertypes[1]  # NONSCIENCE


@dataclass(frozen=True)
class MissionSample:
    bibcode: str
    mission: str
    human_raw: str
    llm_raw: str
    human_label: str
    llm_label: str
    mission_in_text: bool


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
    """Build a confusion-matrix case entry from a mission sample.

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
        Confusion data containing ``tp``, ``tn``, ``fp``, and ``fn``.

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
        Combined summary containing labels, confusion-case entries, and
        computed metric values.
    """
    samples, summary = extract_samples_and_summary(data, missions)
    confusion = compute_confusion(samples)
    metrics = compute_metrics(confusion)
    return {
        **summary,
        "metrics": metrics,
        "fp_bibcodes": confusion["fp_cases"],
        "fn_bibcodes": confusion["fn_cases"],
        "tp_bibcodes": confusion["tp_cases"],
        "tn_bibcodes": confusion["tn_cases"],
        "human_labels": [sample.human_label for sample in samples],
        "llm_labels": [sample.llm_label for sample in samples],
    }


def extract_eval_data_for_run(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
    bibcodes: list[str],
    run_index: int = 0,
) -> dict[str, Any]:
    """Extract confusion-matrix metrics for one run from raw LLM output.

    Parameters
    ----------
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    missions : list[str]
        Missions to evaluate.
    bibcodes : list[str]
        Explicit bibcode roster to evaluate.
    run_index : int, optional
        Zero-based run index to evaluate, by default 0.

    Returns
    -------
    dict[str, Any]
        Single-run confusion-matrix metrics data matching ``extract_eval_data``.
    """
    eval_data = build_eval_data_for_run(
        llm_runs_data=llm_runs_data,
        run_index=run_index,
        bibcodes=bibcodes,
    )
    return extract_eval_data(data=eval_data, missions=missions)


def _append_cm_mission_samples(
    samples: list[MissionSample],
    bibcode: str,
    missions: list[str],
    human_labels: dict[str, str],
    llm_labels: dict[str, str],
) -> None:
    """Append flattened confusion-matrix mission samples for one bibcode.

    Parameters
    ----------
    samples : list[MissionSample]
        Target list receiving mission-level rows.
    bibcode : str
        Bibcode for the source paper.
    missions : list[str]
        Normalized missions to flatten.
    human_labels : dict[str, str]
        Normalized human mission labels for this bibcode.
    llm_labels : dict[str, str]
        Normalized LLM mission labels for this bibcode.
    """
    for mission in missions:
        human_raw = human_labels.get(mission, IGNORED_RAW_LABEL)
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


def build_cm_mission_samples_from_eval_data(
    eval_data: dict[str, dict[str, Any]],
    missions: list[str],
) -> list[MissionSample]:
    """Flatten summary-shaped eval data into confusion-matrix mission samples.

    Parameters
    ----------
    eval_data : dict[str, dict[str, Any]]
        Run-specific evaluation data keyed by bibcode, using the same shape as
        evaluation summary entries.
    missions : list[str]
        Missions to flatten into mission-level rows.

    Returns
    -------
    list[MissionSample]
        Mission-level rows consumed by confusion-matrix counting.
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

        _append_cm_mission_samples(
            samples=samples,
            bibcode=bibcode,
            missions=normalized_missions,
            human_labels=human,
            llm_labels=llm_labels,
        )

    return samples


def build_cm_mission_samples_from_run_evaluations(
    run_evaluations: list[Any],
    missions: list[str],
) -> list[MissionSample]:
    """Flatten raw run evaluations into confusion-matrix mission samples.

    Parameters
    ----------
    run_evaluations : list[Any]
        Run-specific evaluation inputs for the selected bibcodes.
    missions : list[str]
        Missions to flatten into mission-level rows.

    Returns
    -------
    list[MissionSample]
        Mission-level rows consumed by confusion-matrix counting.
    """
    normalized_missions = normalize_missions(missions)
    samples: list[MissionSample] = []

    for evaluation in run_evaluations:
        if not evaluation.has_source:
            continue

        human = evaluation.human_labels
        llm_labels = {mission: prediction.papertype for mission, prediction in evaluation.llm_predictions.items()}

        _append_cm_mission_samples(
            samples=samples,
            bibcode=evaluation.bibcode,
            missions=normalized_missions,
            human_labels=human,
            llm_labels=llm_labels,
        )

    return samples


def build_samples_for_run(
    eval_data: dict[str, dict[str, Any]],
    missions: list[str],
) -> list[MissionSample]:
    """Compatibility wrapper for summary-shaped confusion-matrix sample building."""
    return build_cm_mission_samples_from_eval_data(eval_data=eval_data, missions=missions)


def build_samples_for_run_evaluations(
    run_evaluations: list[Any],
    missions: list[str],
) -> list[MissionSample]:
    """Compatibility wrapper for raw run confusion-matrix sample building."""
    return build_cm_mission_samples_from_run_evaluations(run_evaluations=run_evaluations, missions=missions)


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


def evaluate_multiple_llm_runs(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
    bibcodes: list[str],
    source_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate aggregate confusion-matrix metrics across LLM runs.

    For each run index, this function builds an in-memory run-specific evaluation
    snapshot and computes confusion-matrix metrics from that run only.

    Parameters
    ----------
    llm_runs_data : dict[str, list[dict[str, Any]]]
        Multi-run LLM output data keyed by bibcode.
    missions : list[str]
        Missions to evaluate.
    bibcodes : list[str]
        Explicit bibcode roster to evaluate.

    Returns
    -------
    dict[str, Any]
        Aggregate multi-run metric summary including run count, run coverage,
        per-run metrics, and mean/std aggregate metrics.
    """
    if source_lookup is None:
        source_lookup = build_source_lookup(load_source_dataset(do_verbose=False))

    n_runs = max((len(runs) for runs in llm_runs_data.values()), default=0)
    per_run_metrics: list[dict[str, float | int]] = []

    for run_index in range(n_runs):
        logger.info(f"Compute confusion matrix metrics for run {run_index + 1}/{n_runs}...")
        run_evaluations = build_run_paper_evaluations(
            llm_runs_data=llm_runs_data,
            bibcodes=bibcodes,
            run_index=run_index,
            source_lookup=source_lookup,
        )
        samples = build_cm_mission_samples_from_run_evaluations(
            run_evaluations=run_evaluations,
            missions=missions,
        )
        confusion = compute_confusion(samples)
        metrics = compute_metrics(confusion)
        per_run_metrics.append(metrics)

    return {
        "n_runs": n_runs,
        "run_coverage": compute_run_coverage(bibcodes, llm_runs_data, n_runs, set(source_lookup)),
        "aggregate_metrics": aggregate_metrics_across_runs(per_run_metrics),
        "per_run_metrics": per_run_metrics,
    }
