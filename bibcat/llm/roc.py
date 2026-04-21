from statistics import mean, pstdev
from typing import Any

from sklearn.metrics import auc, roc_curve

from bibcat.llm.metrics import (
    IGNORED_RAW_LABEL,
    POSITIVE_LABEL,
    compute_run_coverage,
    extract_llm_labels,
    extract_llm_run_predictions,
    extract_mission_confidence_map,
    get_llm_run_or_empty,
    has_no_mission_output,
    has_no_paper_source,
    normalize_human_labels,
    normalize_missions,
    to_binary_from_raw,
)


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
    For missing LLM output, the confidence vector defaults to ``[0.0, 1.0]``.
    """
    normalized_missions = normalize_missions(missions)

    human_labels: list[str] = []
    llm_confidences: list[list[float]] = []
    human_llm_missions_seen: set[str] = set()

    for item in data.values():
        if has_no_paper_source(item):
            continue

        human = normalize_human_labels(item.get("human"))

        if has_no_mission_output(item):
            llm = {}
            mission_conf_map = {}
        else:
            llm = extract_llm_labels(item.get("llm"))
            mission_conf_map = extract_mission_confidence_map(item.get("mission_conf"))

        for mission in normalized_missions:
            if mission in human and mission in llm:
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
) -> tuple[list[int], list[list[float]], int]:
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
    n_samples : int
        Number of ROC samples.
    """
    y_true = [1 if label == POSITIVE_LABEL else 0 for label in human_labels]
    n_samples = len(y_true)
    return y_true, llm_confidences, n_samples


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


def build_roc_inputs_for_run(
    eval_data: dict[str, dict[str, Any]],
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
    run_index: int,
) -> tuple[list[int], list[list[float]]]:
    """Build binary ROC inputs for one LLM run.

    Parameters
    ----------
    eval_data : dict
        Existing evaluation data keyed by bibcode. The ``human`` field is used
        as the ground truth.
    llm_runs_data : dict
        Multi-run LLM results keyed by bibcode.
    missions : list of str
        Missions to evaluate.
    run_index : int
        Zero-based run index.

    Returns
    -------
    y_true : list of int
        Binary ground-truth vector where science is 1 and nonscience is 0.
    confidences : list of list of float
        Confidence vectors ordered as ``[p_science, p_nonscience]``.

    Notes
    -----
    Missing runs and missing mission predictions default to ``[0.0, 1.0]``.
    """
    normalized_missions = normalize_missions(missions)
    y_true: list[int] = []
    confidences: list[list[float]] = []

    for bibcode, item in eval_data.items():
        if has_no_paper_source(item):
            continue

        human = normalize_human_labels(item.get("human"))
        llm_runs = llm_runs_data.get(bibcode, [])
        run_item = get_llm_run_or_empty(llm_runs, run_index)
        llm_predictions = extract_llm_run_predictions(run_item)

        for mission in normalized_missions:
            human_raw = human.get(mission, IGNORED_RAW_LABEL)
            human_label = to_binary_from_raw(human_raw)
            y_true.append(1 if human_label == POSITIVE_LABEL else 0)

            if mission in llm_predictions and llm_predictions[mission].confidence is not None:
                confidences.append(llm_predictions[mission].confidence or [0.0, 1.0])
            else:
                confidences.append([0.0, 1.0])

    return y_true, confidences


def evaluate_multiple_llm_runs_with_roc(
    eval_data: dict[str, dict[str, Any]],
    llm_runs_data: dict[str, list[dict[str, Any]]],
    missions: list[str],
) -> dict[str, Any]:
    """Evaluate multiple LLM runs and compute ROC/AUC per run.

    Parameters
    ----------
    eval_data : dict
        Existing evaluation data keyed by bibcode.
    llm_runs_data : dict
        Multi-run LLM results keyed by bibcode.
    missions : list of str
        Missions to evaluate.

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
    n_runs = max((len(runs) for runs in llm_runs_data.values()), default=0)

    per_run_roc: list[dict[str, Any]] = []
    auc_values: list[float] = []

    for run_index in range(n_runs):
        y_true, confidences = build_roc_inputs_for_run(
            eval_data=eval_data,
            llm_runs_data=llm_runs_data,
            missions=missions,
            run_index=run_index,
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
        "run_coverage": compute_run_coverage(eval_data, llm_runs_data, n_runs),
        "aggregate_auc": {
            "mean": float(mean(auc_values)) if auc_values else 0.0,
            "std": float(pstdev(auc_values)) if auc_values else 0.0,
        },
        "per_run_roc": per_run_roc,
    }
