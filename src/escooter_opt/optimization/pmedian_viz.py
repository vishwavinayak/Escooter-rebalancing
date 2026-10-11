"""Visualisations for Phase 4 P-Median Depot Location results.

Generates:
  1. Scatter map: demand zones (sized by weight), selected depots (starred),
     coloured by depot assignment.
  2. Bar chart: zones served and weighted distance per selected depot.
  3. Assignment distance histogram (unweighted).
  4. Elbow / objective-vs-p chart (if multi-p run data is available).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from escooter_opt.optimization.pmedian import PMedianInputs, PMedianResult

FIG_DPI: int = 150
_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]


def _use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        pass


def _color_for(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def plot_zone_depot_map(
    result: PMedianResult,
    inputs: PMedianInputs,
    out_dir: Path,
) -> None:
    """Scatter plot of demand zones and selected depots on a lat/lon plane.

    Demand zones are coloured by their assigned depot.  Marker size is
    proportional to demand weight (turnover).  Selected depot positions are
    overlaid as large star markers.
    """
    _use_style()

    depot_df = inputs.candidate_depots.set_index("ca_number")
    zone_df = inputs.demand_zones.set_index("ca_number")

    # Build depot-index → colour mapping
    depot_ca_to_idx = {sd.depot_ca: i for i, sd in enumerate(result.selected_depots)}

    fig, ax = plt.subplots(figsize=(11, 13))

    # Normalize weights for scatter size
    weights = inputs.demand_zones["weight"].to_numpy(dtype=float)
    min_s, max_s = 20, 300

    # Plot demand zones
    for assign in result.assignments:
        depot_idx = depot_ca_to_idx.get(assign.depot_ca, 0)
        color = _color_for(depot_idx)
        zone_row = zone_df.loc[assign.zone_ca]
        wi_norm = float(zone_df.loc[assign.zone_ca, "weight"]) / weights.max()
        size = min_s + wi_norm * (max_s - min_s)
        ax.scatter(
            zone_row["longitude"],
            zone_row["latitude"],
            s=size,
            c=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
        )

    # Draw assignment lines
    for assign in result.assignments:
        depot_idx = depot_ca_to_idx.get(assign.depot_ca, 0)
        color = _color_for(depot_idx)
        depot_row = depot_df.loc[assign.depot_ca]
        zone_row = zone_df.loc[assign.zone_ca]
        ax.plot(
            [zone_row["longitude"], depot_row["longitude"]],
            [zone_row["latitude"], depot_row["latitude"]],
            color=color,
            alpha=0.20,
            linewidth=0.6,
        )

    # Plot selected depots as stars
    for i, sd in enumerate(result.selected_depots):
        color = _color_for(i)
        ax.scatter(
            sd.longitude,
            sd.latitude,
            s=500,
            c=color,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
        ax.annotate(
            sd.depot_name[:14],
            xy=(sd.longitude, sd.latitude),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )

    # Legend
    handles = [
        mpatches.Patch(
            color=_color_for(i), label=f"Depot {sd.depot_ca}: {sd.depot_name[:18]}"
        )
        for i, sd in enumerate(result.selected_depots)
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        title="Selected Depots",
        title_fontsize=8,
    )

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(
        f"P-Median Depot Location: p={result.p} Selected Depots\n"
        f"(Distance method: {result.distance_method}  |  "
        f"Avg assignment: {result.average_assignment_distance_meters:.0f} m)\n"
        "DISCLAIMER: Modelled optimal candidates, not approved depot sites.",
        fontsize=10,
    )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "zone_depot_map.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_depot_summary_bars(
    result: PMedianResult,
    out_dir: Path,
) -> None:
    """Horizontal bar charts of zones served and weighted distance per depot."""
    _use_style()
    depots = result.selected_depots
    names = [f"{sd.depot_ca}: {sd.depot_name[:20]}" for sd in depots]
    zones_served = [sd.zones_served for sd in depots]
    wtd_km = [sd.weighted_distance_served / 1000 for sd in depots]
    colors = [_color_for(i) for i in range(len(depots))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(depots) * 0.6 + 2)))

    ax1.barh(names, zones_served, color=colors, alpha=0.85)
    ax1.set_xlabel("Zones Served", fontsize=10)
    ax1.set_title("Zones Served per Depot", fontsize=10)
    ax1.invert_yaxis()
    for bar, v in zip(ax1.patches, zones_served):
        ax1.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(v),
            va="center",
            fontsize=8,
        )

    ax2.barh(names, wtd_km, color=colors, alpha=0.85)
    ax2.set_xlabel("Weighted Distance (km-weight)", fontsize=10)
    ax2.set_title("Weighted Assignment Cost per Depot", fontsize=10)
    ax2.invert_yaxis()
    for bar, v in zip(ax2.patches, wtd_km):
        ax2.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.0f}",
            va="center",
            fontsize=8,
        )

    fig.suptitle(
        f"Per-Depot Summary (p={result.p})\n"
        "DISCLAIMER: Modelled optimal candidates, not approved depot sites.",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "depot_summary_bars.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_assignment_distance_hist(
    result: PMedianResult,
    out_dir: Path,
) -> None:
    """Histogram of zone-to-depot assignment distances (unweighted, km)."""
    _use_style()
    distances_km = [a.distance_km for a in result.assignments]
    if not distances_km:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(distances_km, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    mean_d = np.mean(distances_km)
    ax.axvline(
        mean_d,
        color="#DD8452",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_d:.2f} km",
    )
    ax.set_xlabel("Assignment Distance (km)", fontsize=11)
    ax.set_ylabel("Number of Zones", fontsize=11)
    ax.set_title(
        f"Zone-to-Depot Assignment Distance Distribution (p={result.p})\n"
        f"Distance method: {result.distance_method}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "assignment_distance_hist.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def plot_elbow_curve(
    p_values: list[int],
    objectives: list[float],
    out_dir: Path,
) -> None:
    """Elbow / objective-vs-p sensitivity chart.

    Parameters
    ----------
    p_values : list[int]
    objectives : list[float]  objectives in km-weight units
    out_dir : Path
    """
    _use_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_values, objectives, marker="o", linewidth=2, color="#4C72B0")
    ax.set_xlabel("Number of Depots (p)", fontsize=11)
    ax.set_ylabel("Weighted Objective (km-weight)", fontsize=11)
    ax.set_title(
        "P-Median Elbow Curve: Objective vs Number of Depots\n"
        "DISCLAIMER: Modelled optimal candidates, not approved depot sites.",
        fontsize=10,
    )
    ax.set_xticks(p_values)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "elbow_curve.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    result: PMedianResult,
    inputs: PMedianInputs,
    out_dir: Path,
) -> None:
    """Generate and save all Phase 4 visualisation figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_zone_depot_map(result, inputs, out_dir)
    plot_depot_summary_bars(result, out_dir)
    plot_assignment_distance_hist(result, out_dir)
