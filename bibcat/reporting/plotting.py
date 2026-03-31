from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

from .bibliometrics import PlotData, earliest_year_with_any_positive, errors_wilson_ci_from_arrays, norm

OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddishpurple": "#CC79A7",
    "gray": "#6E6E6E",
}


@dataclass(frozen=True)
class PlotStyle:
    """Style and output configuration for bibliometric plots."""

    wilson_confidence: float = 0.95
    individual_series_alpha: float = 0.8
    review_stop_year: int = 2022
    review_stop_text: str = "Human Review Stopped for non-Flagship"
    use_hatch_region: bool = True
    hatch_pattern: str = "///"
    overlay_markersize: float = 2.8
    overlay_colormap: str = "tab20"
    missions_textbox_max: int = 10
    save_figs: bool = False
    save_dir: str = "bibliometrics_plots"
    save_format: str = "png"
    human_color: str = "#153A4B"  # MAST teal color
    llm_color: str = "#F2A155"  # MAST orange color


def apply_matplotlib_style() -> None:
    """Apply global matplotlib style settings."""

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "figure.figsize": (7, 4.6),
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 6,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 8.0,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": True,
            "legend.borderpad": 0.6,
            "figure.constrained_layout.use": False,
        }
    )


def savefig_if_enabled(fig: Figure, filename_stub: str, style: PlotStyle) -> None:
    """Save a figure if saving is enabled."""

    if not style.save_figs:
        return

    output_dir = Path(style.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_stub}.{style.save_format}"
    fig.savefig(path, bbox_inches="tight")


def format_year_axis(
    ax: Axes,
    years: list[int],
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    """Apply consistent formatting to a year-based x-axis."""

    if not years:
        return

    min_year = start_year if start_year is not None else min(years)
    max_year = end_year if end_year is not None else max(years)
    ax.set_xlim(min_year, max_year + 1)

    span = max_year - min_year
    if span <= 10:
        major_step = 1
    elif span <= 20:
        major_step = 2
    elif span <= 40:
        major_step = 5
    else:
        major_step = 10

    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.7)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(which="major", linestyle="-", alpha=0.25)
    ax.grid(which="minor", linestyle=":", alpha=0.18)

    ax.spines["top"].set_alpha(0.35)
    ax.spines["right"].set_alpha(0.35)


def set_nice_legend(ax: Axes, ncol: int = 1, loc: str = "best") -> None:
    """Create and lightly style a legend."""

    legend = ax.legend(ncol=ncol, loc=loc)
    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("#999999")
    frame.set_facecolor("white")


def add_review_stop_region(
    ax: Axes,
    style: PlotStyle,
    start_year: int | None,
    end_year: int | None,
) -> None:
    """Add the review-stop region to a plot."""

    if end_year is None:
        return

    shade_start = max(style.review_stop_year, start_year) if start_year is not None else style.review_stop_year
    if shade_start > end_year:
        return

    if style.use_hatch_region:
        ax.axvspan(
            shade_start,
            end_year + 1,
            facecolor="none",
            edgecolor=OKABE_ITO["gray"],
            hatch=style.hatch_pattern,
            linewidth=0.0,
            zorder=0,
        )
    else:
        ax.axvspan(
            shade_start,
            end_year + 1,
            alpha=0.25,
            color=OKABE_ITO["gray"],
            zorder=0,
        )

    y0, y1 = ax.get_ylim()
    y_text = y0 + 0.06 * (y1 - y0)
    x_text = end_year + 0.8

    ax.text(
        x_text,
        y_text,
        style.review_stop_text,
        fontsize=10,
        color=OKABE_ITO["gray"],
        va="bottom",
        ha="right",
        rotation=90,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=OKABE_ITO["gray"],
            alpha=0.85,
        ),
        zorder=5,
    )


def missions_summary_text(
    missions_with_data: list[str],
    all_requested_missions: list[str],
    max_list: int,
) -> str | None:
    """Build summary text for the missions textbox."""

    n = len(missions_with_data)
    if n == 0:
        return None
    if n <= max_list:
        return "Missions:\n" + ", ".join(missions_with_data)
    if n == len(all_requested_missions):
        return "Missions: All Missions"
    return f"Missions: Many Missions ({n})"


def add_missions_textbox_rule(
    ax: Axes,
    missions_with_data: list[str],
    all_requested_missions: list[str],
    style: PlotStyle,
) -> None:
    """Add the missions textbox to a plot."""

    msg = missions_summary_text(
        missions_with_data=missions_with_data,
        all_requested_missions=all_requested_missions,
        max_list=style.missions_textbox_max,
    )
    if not msg:
        return

    msg = "\n".join(textwrap.wrap(msg, width=60))
    ax.text(
        0.3,
        0.94,
        msg,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92),
        zorder=6,
    )


def add_mission_annotation_bars_axes(
    ax: Axes,
    mission_ranges: dict[str, tuple[int, int]],
    mission_to_color: dict[str, object],
    y_top: float = 0.62,
    y_bottom: float = 0.08,
    bar_height: float = 0.04,
    alpha: float = 0.18,
    text_size: int = 10,
    text_weight: str = "semibold",
) -> None:
    """Draw mission active range bars using data coordinates in x and axes coordinates in y."""

    missions = [mission for mission in mission_ranges if mission in mission_to_color]
    if not missions:
        return

    band = max(1e-6, (y_top - y_bottom))
    step = band / max(1, len(missions))
    height = min(bar_height, step * 0.75)

    transform = blended_transform_factory(ax.transData, ax.transAxes)

    for i, mission in enumerate(missions):
        x0, x1 = mission_ranges[mission]
        y = y_bottom + i * step + (step - height) / 2
        color = mission_to_color[mission]

        ax.add_patch(
            Rectangle(
                (x0, y),
                x1 - x0,
                height,
                transform=transform,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=1,
                clip_on=False,
            )
        )

        ax.text(
            x0 + (x1 - x0) / 2,
            y + height / 2,
            mission,
            transform=transform,
            ha="center",
            va="center",
            fontsize=text_size,
            weight=text_weight,
            color="black",
            zorder=2,
            clip_on=False,
        )


def plot_per_mission(
    data: PlotData,
    style: PlotStyle,
    auto_start_mission_at_first_count_year: bool = True,
) -> None:
    """Plot per-mission human and LLM counts with Wilson confidence intervals."""

    for mission in data.missions:
        if mission not in data.positives:
            continue

        year_range = list(range(data.start_year, data.end_year + 1))
        totals = np.array([data.total_records_by_year.get(year, 0) for year in year_range], dtype=float)

        human_pos = np.array(
            [data.positives[mission].get(year, {}).get("human", 0) for year in year_range],
            dtype=float,
        )
        llm_pos = np.array(
            [data.positives[mission].get(year, {}).get("llm", 0) for year in year_range],
            dtype=float,
        )

        human_plot = np.where(totals <= 0, np.nan, human_pos)
        llm_plot = np.where(totals <= 0, np.nan, llm_pos)

        mission_start = data.start_year
        if auto_start_mission_at_first_count_year:
            first_pos_year = earliest_year_with_any_positive(year_range, human_plot, llm_plot)
            if first_pos_year is not None:
                mission_start = max(data.start_year, first_pos_year)

        start_idx = mission_start - data.start_year
        years = np.array(year_range[start_idx:], dtype=int)
        totals_trim = totals[start_idx:]
        human_trim = human_plot[start_idx:]
        llm_trim = llm_plot[start_idx:]

        if np.all(np.isnan(human_trim)) and np.all(np.isnan(llm_trim)):
            continue

        human_yerr = errors_wilson_ci_from_arrays(
            human_trim,
            totals_trim,
            confidence=style.wilson_confidence,
        )
        llm_yerr = errors_wilson_ci_from_arrays(
            llm_trim,
            totals_trim,
            confidence=style.wilson_confidence,
        )

        fig, ax = plt.subplots()
        ax.errorbar(
            years,
            human_trim,
            yerr=human_yerr,
            fmt="o-",
            color=style.human_color,
            alpha=style.individual_series_alpha,
            elinewidth=1.2,
            capsize=2.5,
            label=f"Human ({data.paper_type})",
            zorder=3,
        )
        ax.errorbar(
            years,
            llm_trim,
            yerr=llm_yerr,
            fmt="s-",
            color=style.llm_color,
            alpha=style.individual_series_alpha,
            elinewidth=1.2,
            capsize=2.5,
            label=f"LLM ({data.paper_type})",
            zorder=3,
        )

        ax.set_xlabel("Publication year")
        ax.set_ylabel("Publication Count")
        ax.set_title(
            f"{mission}: Human vs LLM counts (Wilson {int(style.wilson_confidence * 100)}% CI)\n"
            f"Years {mission_start}–{data.end_year} | {data.publisher_label}"
        )

        format_year_axis(ax, years.tolist(), mission_start - 1, data.end_year)
        add_review_stop_region(ax, style=style, start_year=mission_start, end_year=data.end_year)
        set_nice_legend(ax, ncol=2, loc="best")

        fig.tight_layout()
        savefig_if_enabled(fig, f"{mission}_human_vs_llm_wilson_{mission_start}_{data.end_year}", style)
        plt.show()


def plot_sum(data: PlotData, style: PlotStyle) -> None:
    """Plot summed human and LLM counts across all configured missions."""

    year_range = list(range(data.start_year, data.end_year + 1))

    totals_records = np.array([data.total_records_by_year.get(year, 0) for year in year_range], dtype=float)
    totals_pairs = totals_records * float(len(data.missions))

    sum_human = np.array([data.sum_counts.get(year, {}).get("human", 0) for year in year_range], dtype=float)
    sum_llm = np.array([data.sum_counts.get(year, {}).get("llm", 0) for year in year_range], dtype=float)

    sum_human_plot = np.where(totals_pairs <= 0, np.nan, sum_human)
    sum_llm_plot = np.where(totals_pairs <= 0, np.nan, sum_llm)

    human_yerr = errors_wilson_ci_from_arrays(
        sum_human_plot,
        totals_pairs,
        confidence=style.wilson_confidence,
    )
    llm_yerr = errors_wilson_ci_from_arrays(
        sum_llm_plot,
        totals_pairs,
        confidence=style.wilson_confidence,
    )

    fig, ax = plt.subplots()
    ax.errorbar(
        year_range,
        sum_human_plot,
        yerr=human_yerr,
        fmt="o-",
        color=style.human_color,
        alpha=0.8,
        elinewidth=1.2,
        capsize=2.5,
        label="SUM Human",
        zorder=3,
    )
    ax.errorbar(
        year_range,
        sum_llm_plot,
        yerr=llm_yerr,
        fmt="s-",
        color=style.llm_color,
        alpha=0.8,
        elinewidth=1.2,
        capsize=2.5,
        label="SUM LLM",
        zorder=3,
    )

    ax.set_xlabel("Publication year")
    ax.set_ylabel(f"Count {data.paper_type} labels")
    ax.set_title(
        f"Total {data.paper_type} labels across missions\n"
        f"Years {data.start_year}–{data.end_year} | {data.publisher_label}"
    )

    format_year_axis(ax, year_range, data.start_year, data.end_year)
    add_review_stop_region(ax, style=style, start_year=data.start_year, end_year=data.end_year)
    add_missions_textbox_rule(ax, data.missions_with_data, data.missions, style)

    add_mission_annotation_bars_axes(
        ax,
        mission_ranges=data.mission_active_ranges,
        mission_to_color=data.mission_to_color,
        y_top=0.70,
        y_bottom=0.10,
        bar_height=0.05,
        alpha=0.16,
    )
    set_nice_legend(ax, ncol=2, loc="upper left")

    fig.tight_layout()
    savefig_if_enabled(fig, f"SUM_paircount_human_vs_llm_wilson_{data.start_year}_{data.end_year}", style)
    plt.show()


def plot_sum_excl_flagship(data: PlotData, style: PlotStyle) -> None:
    """Plot summed counts excluding flagship missions."""
    year_range = list(range(data.start_year, data.end_year + 1))
    flagship_norm = {norm(mission) for mission in data.flagship_missions if norm(mission) is not None}
    flagships_in_missions = [mission for mission in data.missions if norm(mission) in flagship_norm]
    n_pairs_missions = len(data.missions) - len(flagships_in_missions)

    if n_pairs_missions <= 0:
        print("Skipping SUM excl. flagship: no non-flagship missions left after exclusion.")
        return

    totals_records = np.array([data.total_records_by_year.get(year, 0) for year in year_range], dtype=float)
    totals_pairs_excl = totals_records * float(n_pairs_missions)

    sum_human_excl = np.array(
        [data.sum_counts_excl_flagship.get(year, {}).get("human", 0) for year in year_range],
        dtype=float,
    )
    sum_llm_excl = np.array(
        [data.sum_counts_excl_flagship.get(year, {}).get("llm", 0) for year in year_range],
        dtype=float,
    )

    sum_human_excl_plot = np.where(totals_pairs_excl <= 0, np.nan, sum_human_excl)
    sum_llm_excl_plot = np.where(totals_pairs_excl <= 0, np.nan, sum_llm_excl)

    human_yerr_excl = errors_wilson_ci_from_arrays(
        sum_human_excl_plot,
        totals_pairs_excl,
        confidence=style.wilson_confidence,
    )
    llm_yerr_excl = errors_wilson_ci_from_arrays(
        sum_llm_excl_plot,
        totals_pairs_excl,
        confidence=style.wilson_confidence,
    )

    fig, ax = plt.subplots()
    ax.errorbar(
        year_range,
        sum_human_excl_plot,
        yerr=human_yerr_excl,
        fmt="o-",
        color=style.human_color,
        alpha=0.8,
        elinewidth=1.2,
        capsize=2.5,
        label="SUM Human (excl. flagship)",
        zorder=3,
    )
    ax.errorbar(
        year_range,
        sum_llm_excl_plot,
        yerr=llm_yerr_excl,
        fmt="s-",
        color=style.llm_color,
        alpha=0.8,
        elinewidth=1.2,
        capsize=2.5,
        label="SUM LLM (excl. flagship)",
        zorder=3,
    )

    ax.set_xlabel("Publication year")
    ax.set_ylabel(f"Count {data.paper_type} labels")
    ax.set_title(
        f"Total {data.paper_type} labels across missions without Flagship Missions\n"
        f"Years {data.start_year}–{data.end_year} | {data.publisher_label}"
    )

    format_year_axis(ax, year_range, data.start_year, data.end_year)
    add_review_stop_region(ax, style=style, start_year=data.start_year, end_year=data.end_year)
    add_mission_annotation_bars_axes(
        ax,
        mission_ranges=data.mission_active_ranges,
        mission_to_color=data.mission_to_color,
        y_top=0.80,
        y_bottom=0.10,
        bar_height=0.05,
        alpha=0.16,
    )

    ax.text(
        0.3,
        0.9,
        f"Excluded: {', '.join(flagships_in_missions)}\nMissions counted: {n_pairs_missions}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92),
        zorder=6,
    )

    set_nice_legend(ax, ncol=1, loc="upper left")
    fig.tight_layout()
    savefig_if_enabled(
        fig,
        f"SUM_paircount_EXCL_FLAGSHIP_human_vs_llm_wilson_{data.start_year}_{data.end_year}",
        style,
    )
    plt.show()


def plot_overlay(
    data: PlotData,
    style: PlotStyle,
    kind: Literal["human", "llm"],
    sum_mode: Literal["all", "excl_flagship"] = "all",
) -> None:
    """Plot mission overlay lines with either full or non-flagship summed series."""
    if not data.missions_with_data:
        return

    flagship_norm = {norm(mission) for mission in data.flagship_missions if norm(mission) is not None}

    fig, ax = plt.subplots()

    for mission in data.missions_with_data:
        years = sorted(data.positives[mission].keys())
        values = [data.positives[mission][year][kind] for year in years]
        linestyle = "--" if norm(mission) in flagship_norm else "-"

        ax.plot(
            years,
            values,
            marker="o",
            linestyle=linestyle,
            linewidth=1.8,
            markersize=style.overlay_markersize,
            alpha=0.95,
            color=data.mission_to_color[mission],
            label=mission,
            zorder=2,
        )

    if sum_mode == "excl_flagship":
        sum_dict = data.sum_counts_excl_flagship
        sum_label = "MAST/non-flagship"
    else:
        sum_dict = data.sum_counts
        sum_label = "All missions"

    sum_years = sorted(sum_dict.keys())
    sum_vals = [sum_dict[year][kind] for year in sum_years]

    ax.plot(
        sum_years,
        sum_vals,
        marker="s",
        linestyle="-",
        linewidth=2.8,
        markersize=5.2,
        color=OKABE_ITO["black"],
        label=sum_label,
        zorder=4,
    )

    ax.set_xlabel("Publication year")
    ax.set_ylabel(f"Count {data.paper_type} labels")

    kind_label = "Human" if kind == "human" else "LLM"
    ax.set_title(
        f"{kind_label} {data.paper_type} counts by mission\n"
        f"Years {data.start_year}–{data.end_year} | {data.publisher_label}"
    )

    full_years = list(range(data.start_year, data.end_year + 1))
    format_year_axis(ax, full_years, data.start_year, data.end_year)
    add_review_stop_region(ax, style=style, start_year=data.start_year, end_year=data.end_year)
    add_mission_annotation_bars_axes(
        ax,
        mission_ranges=data.mission_active_ranges,
        mission_to_color=data.mission_to_color,
        y_top=0.70,
        y_bottom=0.10,
        bar_height=0.05,
        alpha=0.16,
    )

    if len(data.missions_with_data) >= 6:
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_edgecolor("#999999")
    else:
        set_nice_legend(ax, ncol=2, loc="upper left")

    fig.tight_layout()
    out_tag = "ALL" if sum_mode == "all" else "EXCL_FLAGSHIP"
    savefig_if_enabled(fig, f"OVERLAY_{kind}_{out_tag}_{data.start_year}_{data.end_year}", style)
    plt.show()
    plt.show()
