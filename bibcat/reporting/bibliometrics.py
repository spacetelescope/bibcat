from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any, DefaultDict, Iterable

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for filtering and aggregating bibliometric records."""

    missions: list[str]
    flagship_missions: list[str]
    paper_type: str
    start_year: int
    end_year: int
    publishers_include: list[str] | None = None
    publishers_exclude: list[str] | None = None
    exclude_publishers_as_substring: bool = True


@dataclass
class PlotData:
    """Aggregated data products used by plotting functions."""

    records: list[dict[str, Any]]
    missions: list[str]
    missions_with_data: list[str]
    flagship_missions: list[str]
    mission_active_ranges: dict[str, tuple[int, int]]
    total_records_by_year: dict[int, int]
    positives: dict[str, dict[int, dict[str, int]]]
    sum_counts: dict[int, dict[str, int]]
    sum_counts_excl_flagship: dict[int, dict[str, int]]
    mission_to_color: dict[str, Any]
    publisher_label: str
    start_year: int
    end_year: int
    paper_type: str


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load bibliometric records from a JSON file.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the JSON file.

    Returns
    -------
    list of dict
        Loaded record dictionaries.

    Raises
    ------
    ValueError
        If the top-level JSON object is not a list.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Top-level JSON must be a list of records")

    return [record for record in records if isinstance(record, dict)]


def norm(value: Any) -> str | None:
    """Normalize a value for case-insensitive string comparison.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str or None
        Stripped and case-folded string if the input is a string, otherwise
        ``None``.
    """
    if isinstance(value, str):
        return value.strip().casefold()
    return None


def normalized_nonnull(values: Iterable[object]) -> set[str]:
    """Return normalized non-null string values.

    Parameters
    ----------
    values : iterable of object
        Values to normalize.

    Returns
    -------
    set of str
        Normalized values with nulls removed.
    """
    result: set[str] = set()
    for value in values:
        normalized = norm(value)
        if normalized is not None:
            result.add(normalized)
    return result


def parse_year(date_str: object) -> int | None:
    """Extract the calendar year from an ISO-format date or datetime string.

    Parameters
    ----------
    date_str : object
        publication date.

    Returns
    -------
    int | None
        Parsed year if successful, otherwise ``None``.
    """
    if not isinstance(date_str, str):
        return None

    try:
        return datetime.fromisoformat(date_str).year
    except Exception:
        return None


def publisher_label(
    publishers_include: list[str] | None,
    publishers_exclude: list[str] | None,
    include_norm: set[str] | None,
    exclude_norm: set[str],
) -> str:
    """Build a human-readable label for publisher filters.

    Parameters
    ----------
    publishers_include : list[str] | None
        Included publisher names.
    publishers_exclude : list[str] | None
        Excluded publisher names.
    include_norm : set[str] | None
        Normalized included publishers.
    exclude_norm : set[str]
        Normalized excluded publishers.

    Returns
    -------
    str
        Summary label for publisher filtering.
    """
    if include_norm is not None:
        included = ", ".join(publishers_include or [])
        if exclude_norm:
            excluded = ", ".join(publishers_exclude or [])
            return f"Include: [{included}] | Exclude: [{excluded}]"
        return f"Include: [{included}]"

    if exclude_norm:
        excluded = ", ".join(publishers_exclude or [])
        return f"All publishers except: [{excluded}]"

    return "All publishers"


def make_unique_mission_colors(
    missions: Iterable[str],
    cmap_name: str = "tab20",
) -> dict[str, Any]:
    """Generate a unique color mapping for missions.

    Parameters
    ----------
    missions : Iterable[str]
        Mission names.
    cmap_name : str, optional
        Matplotlib colormap name.

    Returns
    -------
    dict
        Mapping from mission name to color.
    """
    mission_list = list(missions)
    n = len(mission_list)
    cmap = mpl.colormaps.get_cmap(cmap_name)

    if hasattr(cmap, "N") and cmap.N >= n and cmap_name.lower().startswith("tab"):
        colors = [cmap(i) for i in range(n)]
    else:
        if n == 1:
            colors = [cmap(0.5)]
        else:
            xs = [0.08 + (0.84 * i / (n - 1)) for i in range(n)]
            colors = [cmap(x) for x in xs]

    return dict(zip(mission_list, colors))


def wilson_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of positive outcomes.
    n : int
        Number of trials.
    confidence : float, optional
        Confidence level.

    Returns
    -------
    tuple[float, float]
        Lower and upper interval bounds.
    """
    n = int(n)
    if n <= 0:
        return (np.nan, np.nan)

    k = max(0, min(int(k), n))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * ((phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def errors_wilson_ci_from_arrays(
    positives: NDArray[np.float64] | list[float],
    totals: NDArray[np.float64] | list[float],
    confidence: float = 0.95,
) -> NDArray[np.float64]:
    """Convert Wilson proportion intervals into count-space asymmetric errors.

    Parameters
    ----------
    positives : array-like
        Positive counts.
    totals : array-like
        Total counts.
    confidence : float, optional
        Confidence level.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2, N)`` containing lower and upper errors for use
        with matplotlib ``yerr``.
    """
    positives_arr = np.asarray(positives, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)

    lows = np.full_like(positives_arr, np.nan, dtype=float)
    highs = np.full_like(positives_arr, np.nan, dtype=float)

    for i, (pos, tot) in enumerate(zip(positives_arr, totals_arr)):
        if np.isnan(tot) or tot <= 0 or np.isnan(pos):
            continue

        tot_i = int(tot)
        pos_i = max(0, min(int(pos), tot_i))
        lo_p, hi_p = wilson_interval(pos_i, tot_i, confidence=confidence)
        lows[i] = lo_p * tot_i
        highs[i] = hi_p * tot_i

    err_low = np.where(np.isnan(positives_arr), np.nan, np.maximum(0.0, positives_arr - lows))
    err_high = np.where(np.isnan(positives_arr), np.nan, np.maximum(0.0, highs - positives_arr))
    return np.vstack([err_low, err_high])


def earliest_year_with_any_positive(
    year_range: Iterable[int],
    human_pos: NDArray[np.float64],
    llm_pos: NDArray[np.float64],
) -> int | None:
    """Find the earliest year with any positive human or LLM count.

    Parameters
    ----------
    year_range : Iterable[int]
        Year values corresponding to the arrays.
    human_pos : numpy.ndarray
        Human count series.
    llm_pos : numpy.ndarray
        LLM count series.

    Returns
    -------
    int | None
        Earliest year with a positive non-NaN count, or ``None``.
    """
    for year, human, llm in zip(year_range, human_pos, llm_pos):
        if (not np.isnan(human) and human > 0) or (not np.isnan(llm) and llm > 0):
            return year
    return None


def build_publisher_allowed(
    config: FilterConfig,
) -> tuple[set[str] | None, set[str], callable]:
    """Build publisher filter state and predicate.

    Parameters
    ----------
    config : FilterConfig
        Aggregation/filter configuration.

    Returns
    -------
    tuple
        ``(include_norm, exclude_norm, predicate)`` where ``predicate`` is a
        callable accepting a publisher value and returning a boolean.
    """
    include_norm = None
    if config.publishers_include:
        include_norm = normalized_nonnull(config.publishers_include)

    exclude_norm: set[str] = set()
    if config.publishers_exclude:
        exclude_norm = normalized_nonnull(config.publishers_exclude)

    def publisher_allowed(publisher_value: Any) -> bool:
        pnorm = norm(publisher_value)

        if pnorm is None:
            return True

        if include_norm is not None and pnorm not in include_norm:
            return False

        if exclude_norm:
            if config.exclude_publishers_as_substring:
                if any(ex in pnorm for ex in exclude_norm):
                    return False
            else:
                if pnorm in exclude_norm:
                    return False

        return True

    return include_norm, exclude_norm, publisher_allowed


def build_plot_data(
    records: list[dict[str, Any]],
    config: FilterConfig,
    mission_active_ranges: dict[str, tuple[int, int]],
    overlay_colormap: str = "tab20",
) -> PlotData:
    """Aggregate record data into plot-ready structures.

    Parameters
    ----------
    records : list[dict]
        Input bibliometric records.
    config : FilterConfig
        Aggregation and filtering configuration.
    mission_active_ranges : dict
        Mapping of mission name to active year ranges.
    overlay_colormap : str, optional
        Matplotlib colormap used for mission line colors.

    Returns
    -------
    PlotData
        Aggregated data bundle used by plotting functions.
    """
    # sanitizing mission names
    missions_norm_to_canon = {
        normalized: mission for mission in config.missions if (normalized := norm(mission)) is not None
    }
    missions_norm = set(missions_norm_to_canon)
    flagship_missions_norm = normalized_nonnull(config.flagship_missions)

    # publishers in-/exclusion
    include_norm, exclude_norm, publisher_allowed = build_publisher_allowed(config)

    # Counting totals and positives
    total_records_by_year: DefaultDict[int, int] = defaultdict(int)
    positives: DefaultDict[str, DefaultDict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"human": 0, "llm": 0})
    )

    for rec in records:
        if not publisher_allowed(rec.get("publisher_name")):
            continue

        year = parse_year(rec.get("publication_date"))
        if year is None or year < config.start_year or year > config.end_year:
            continue

        total_records_by_year[year] += 1

        cm_list = rec.get("class_missions", [])
        if isinstance(cm_list, list):
            for cm in cm_list:
                if not isinstance(cm, dict):
                    continue

                mission_norm = norm(cm.get("mission"))
                if mission_norm not in missions_norm:
                    continue

                if norm(cm.get("paper_type")) == norm(config.paper_type):
                    mission_canon = missions_norm_to_canon[mission_norm]
                    positives[mission_canon][year]["human"] += 1

        scores = rec.get("scores", [])
        latest_score: dict[str, Any] | None = None
        latest_dt: datetime | None = None

        if isinstance(scores, list) and scores:
            for score in scores:
                if not isinstance(score, dict):
                    continue

                try:
                    score_date = score.get("score_date")
                    dt = datetime.fromisoformat(score_date) if isinstance(score_date, str) else None
                except Exception:
                    dt = None

                if latest_dt is None or (dt is not None and dt > latest_dt):
                    latest_dt = dt
                    latest_score = score

        if isinstance(latest_score, dict):
            missions_obj = latest_score.get("missions", {})
            if isinstance(missions_obj, dict):
                for mission_name, payload in missions_obj.items():
                    mission_norm = norm(mission_name)
                    if mission_norm not in missions_norm:
                        continue

                    papertype = payload.get("papertype") if isinstance(payload, dict) else None
                    if norm(papertype) == norm(config.paper_type):
                        mission_canon = missions_norm_to_canon[mission_norm]
                        positives[mission_canon][year]["llm"] += 1

    missions_with_data = [
        mission for mission in config.missions if mission in positives and len(positives[mission]) > 0
    ]
    # unique mission colors for plotting
    mission_to_color = make_unique_mission_colors(missions_with_data, cmap_name=overlay_colormap)

    # summing counts by years
    sum_counts: DefaultDict[int, dict[str, int]] = defaultdict(lambda: {"human": 0, "llm": 0})
    for mission in missions_with_data:
        for year, values in positives[mission].items():
            sum_counts[year]["human"] += values["human"]
            sum_counts[year]["llm"] += values["llm"]
    # summing counts by years except flagship missions
    sum_counts_excl_flagship: DefaultDict[int, dict[str, int]] = defaultdict(lambda: {"human": 0, "llm": 0})
    for mission in missions_with_data:
        if norm(mission) in flagship_missions_norm:
            continue

        for year, values in positives[mission].items():
            sum_counts_excl_flagship[year]["human"] += values["human"]
            sum_counts_excl_flagship[year]["llm"] += values["llm"]
    # publisher label for annotations
    pub_label = publisher_label(
        publishers_include=config.publishers_include,
        publishers_exclude=config.publishers_exclude,
        include_norm=include_norm,
        exclude_norm=exclude_norm,
    )

    return PlotData(
        records=records,
        missions=config.missions,
        missions_with_data=missions_with_data,
        flagship_missions=config.flagship_missions,
        mission_active_ranges=mission_active_ranges,
        total_records_by_year=dict(total_records_by_year),
        positives={mission: dict(years) for mission, years in positives.items()},
        sum_counts=dict(sum_counts),
        sum_counts_excl_flagship=dict(sum_counts_excl_flagship),
        mission_to_color=mission_to_color,
        publisher_label=pub_label,
        start_year=config.start_year,
        end_year=config.end_year,
        paper_type=config.paper_type,
    )
