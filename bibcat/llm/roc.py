from statistics import mean, pstdev
from typing import Any

from sklearn.metrics import auc, roc_curve

from bibcat import config
from bibcat.data.build_dataset import load_source_dataset
from bibcat.llm.run_eval import (
    IGNORED_RAW_LABEL,
    POSITIVE_LABEL,
    build_run_paper_evaluations,
    build_source_lookup,
    compute_run_coverage,
    normalize_missions,
    to_binary_from_raw,
)


def get_roc_metrics(
    llm_confidences: list[list[float]],
    y_true: list[int],
) -> tuple[list[float], list[float], list[float], float]:
    """Compute binary ROC curve and AUC.

    Parameters
    ----------
    llm_confidences : list of list of float
        Probability vectors ordered as ``[p_science, p_nonscience]``.
    y_true : list of int
        Binary ground-truth vector.

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
        If any confidence vector is malformed.
    """
    if any(len(conf) != 2 for conf in llm_confidences):
        raise ValueError("Each confidence vector must have exactly two values: [p_science, p_nonscience].")

    science_scores = [float(conf[0]) for conf in llm_confidences]
    fpr, tpr, thresholds = roc_curve(y_true, science_scores)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def build_roc_inputs_for_run_evaluations(
    run_evaluations: list[Any],
    missions: list[str],
) -> tuple[list[int], list[list[float]], list[str]]:
    """Build ROC inputs from raw run-evaluation inputs.

    Parameters
    ----------
    run_evaluations : list[Any]
        Run-specific evaluation inputs for the selected bibcodes.
    missions : list[str]
        Missions to evaluate.

    Returns
    -------
    tuple[list[int], list[list[float]], list[str]]
        Binary ground truth, confidence vectors, and missions called out by both
        human and LLM in the selected run.
    """
    normalized_missions = normalize_missions(missions)
    y_true: list[int] = []
    confidences: list[list[float]] = []
    human_llm_missions_seen: set[str] = set()

    for evaluation in run_evaluations:
        if not evaluation.has_source:
            continue

        human = evaluation.human_labels

        for mission in normalized_missions:
            prediction = evaluation.llm_predictions.get(mission)
            if mission in human and prediction is not None:
                human_llm_missions_seen.add(mission)

            human_raw = human.get(mission, IGNORED_RAW_LABEL)
            human_label = to_binary_from_raw(human_raw)
            y_true.append(1 if human_label == POSITIVE_LABEL else 0)

            if prediction is not None and prediction.confidence is not None and len(prediction.confidence) == 2:
                confidences.append(list(prediction.confidence))
            else:
                confidences.append([0.0, 1.0])

    return y_true, confidences, sorted(human_llm_missions_seen)


def extract_roc_metrics_for_run(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
    bibcodes: list[str],
    run_index: int = 0,
    source_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract ROC metrics for one run from raw LLM output.

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
        Compact ROC payload for the selected run.
    """
    if source_lookup is None:
        source_lookup = build_source_lookup(load_source_dataset())

    run_evaluations = build_run_paper_evaluations(
        llm_runs_data=llm_runs_data,
        bibcodes=bibcodes,
        run_index=run_index,
        source_lookup=source_lookup,
    )
    y_true, llm_confidences, human_llm_missions = build_roc_inputs_for_run_evaluations(
        run_evaluations=run_evaluations,
        missions=missions,
    )
    n_verdicts = len(y_true)
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(llm_confidences, y_true)

    return {
        "threshold": config.llms.performance.threshold,
        "missions": normalize_missions(missions),
        "human_llm_missions": human_llm_missions,
        "n_verdicts": n_verdicts,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
    }


def evaluate_multiple_llm_runs_with_roc(
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
    bibcodes: list[str],
    source_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate multiple LLM runs and compute ROC/AUC per run.

    For each run index, this function builds an in-memory run-specific
    evaluation snapshot and computes ROC/AUC from that run only.

    Parameters
    ----------
    llm_runs_data : dict
        Multi-run LLM results keyed by bibcode.
    missions : list of str
        Missions to evaluate.
    bibcodes : list[str]
        Explicit bibcode roster to evaluate.

    Returns
    -------
    dict
        Multi-run summary augmented with per-run ROC/AUC values and aggregate
        AUC mean/std.

    Notes
    -----
    This function returns ROC-focused aggregate output only. Confusion-matrix
    aggregate payload fields are intentionally excluded from this return value.
    """
    if source_lookup is None:
        source_lookup = build_source_lookup(load_source_dataset())

    n_runs = max((len(runs) for runs in llm_runs_data.values()), default=0)

    per_run_roc: list[dict[str, Any]] = []
    auc_values: list[float] = []

    for run_index in range(n_runs):
        run_evaluations = build_run_paper_evaluations(
            llm_runs_data=llm_runs_data,
            bibcodes=bibcodes,
            run_index=run_index,
            source_lookup=source_lookup,
        )
        y_true, confidences, _ = build_roc_inputs_for_run_evaluations(
            run_evaluations=run_evaluations,
            missions=missions,
        )
        fpr, tpr, thresholds, roc_auc = get_roc_metrics(
            llm_confidences=confidences,
            y_true=y_true,
        )
        per_run_roc.append(
            {
                "run_index": run_index,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "roc_auc": roc_auc,
            }
        )
        auc_values.append(roc_auc)

    return {
        "n_runs": n_runs,
        "run_coverage": compute_run_coverage(bibcodes, llm_runs_data, n_runs, set(source_lookup)),
        "aggregate_auc": {
            "mean": float(mean(auc_values)) if auc_values else 0.0,
            "std": float(pstdev(auc_values)) if auc_values else 0.0,
        },
        "per_run_roc": per_run_roc,
    }
