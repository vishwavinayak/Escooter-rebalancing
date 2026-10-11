"""Visualisations for Phase 3 Transportation Problem results.

Generates:
  - Allocation heatmap (surplus × deficit cost matrix with flow overlay)
  - Allocation bar chart (top transfers by total cost)
  - Supply vs demand comparison bar chart
  - Zone utilisation chart
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from escooter_opt.optimization.transport import (
        TransportProblemInputs,
        TransportProblemResult,
    )

# Use non-interactive backend to avoid display dependency
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DPI: int = 150
FIG_STYLE: str = "seaborn-v0_8-darkgrid"


def _use_style() -> None:
    try:
        plt.style.use(FIG_STYLE)
    except OSError:
        pass  # fallback to default


def plot_allocation_bar(
    result: TransportProblemResult,
    out_dir: Path,
    top_n: int = 25,
) -> None:
    """Horizontal bar chart of the top-N allocations by total transfer cost."""
    _use_style()
    allocs = result.allocations[:top_n]
    if not allocs:
        return

    labels = [f"{a.surplus_name[:16]}→{a.deficit_name[:16]}" for a in allocs]
    costs_km = [a.total_cost_km for a in allocs]
    units = [a.units for a in allocs]

    fig, ax1 = plt.subplots(figsize=(12, max(6, len(allocs) * 0.38)))
    bars = ax1.barh(
        labels, costs_km, color="#4C72B0", alpha=0.85, label="Total cost (km)"
    )
    ax1.set_xlabel("Estimated Total Transfer Cost (km)", fontsize=11)
    ax1.set_title(
        f"Top {top_n} Allocations by Total Estimated Transfer Cost\n"
        "(Cost = units × Manhattan distance between zone centroids)\n"
        "[NOTE] Quantities are estimated from historical trip-flow imbalance, not observed inventory",
        fontsize=10,
    )
    ax1.invert_yaxis()

    # Annotate with unit counts
    for bar, u in zip(bars, units):
        ax1.text(
            bar.get_width() + max(costs_km) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{u:,} units",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "allocation_bar_chart.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_supply_demand_comparison(
    result: TransportProblemResult,
    inputs: TransportProblemInputs,
    out_dir: Path,
) -> None:
    """Side-by-side bar chart comparing surplus supply and deficit demand per zone."""
    _use_style()
    surplus_df = inputs.surplus_zones.copy()
    deficit_df = inputs.deficit_zones.copy()

    # Compute units sent/received from allocation
    sent: dict[int, int] = {}
    recv: dict[int, int] = {}
    for a in result.allocations:
        sent[a.surplus_ca] = sent.get(a.surplus_ca, 0) + a.units
        recv[a.deficit_ca] = recv.get(a.deficit_ca, 0) + a.units

    surplus_df["units_sent"] = surplus_df["ca_number"].map(sent).fillna(0).astype(int)
    deficit_df["units_recv"] = deficit_df["ca_number"].map(recv).fillna(0).astype(int)

    # Top 20 each by magnitude
    top_surplus = surplus_df.nlargest(20, "supply")
    top_deficit = deficit_df.nlargest(20, "demand")

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(16, 8))

    # Surplus panel
    y_s = range(len(top_surplus))
    ax_s.barh(
        list(y_s),
        top_surplus["supply"],
        color="#4C72B0",
        alpha=0.6,
        label="Estimated supply",
    )
    ax_s.barh(
        list(y_s),
        top_surplus["units_sent"],
        color="#DD8452",
        alpha=0.9,
        label="Units dispatched",
    )
    ax_s.set_yticks(list(y_s))
    ax_s.set_yticklabels(top_surplus["ca_name"].tolist(), fontsize=8)
    ax_s.invert_yaxis()
    ax_s.set_xlabel("Estimated Units")
    ax_s.set_title("Top 20 Surplus Zones\n(Supply vs Dispatched)", fontsize=10)
    ax_s.legend(fontsize=8)

    # Deficit panel
    y_d = range(len(top_deficit))
    ax_d.barh(
        list(y_d),
        top_deficit["demand"],
        color="#55A868",
        alpha=0.6,
        label="Estimated demand",
    )
    ax_d.barh(
        list(y_d),
        top_deficit["units_recv"],
        color="#C44E52",
        alpha=0.9,
        label="Units received",
    )
    ax_d.set_yticks(list(y_d))
    ax_d.set_yticklabels(top_deficit["ca_name"].tolist(), fontsize=8)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Estimated Units")
    ax_d.set_title("Top 20 Deficit Zones\n(Demand vs Received)", fontsize=10)
    ax_d.legend(fontsize=8)

    fig.suptitle(
        "Estimated Supply & Demand vs Allocation Outcomes\n"
        "[NOTE] All quantities inferred from historical trip-flow imbalance, not observed inventory",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "supply_demand_comparison.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def plot_cost_matrix_heatmap(
    result: TransportProblemResult,
    inputs: TransportProblemInputs,
    out_dir: Path,
) -> None:
    """Heatmap of the cost matrix with allocated flows overlaid as annotations.

    Only shows the top-20 surplus and top-20 deficit zones by magnitude
    to keep the figure readable.
    """
    _use_style()

    surplus_df = inputs.surplus_zones.copy()
    deficit_df = inputs.deficit_zones.copy()

    # Pick top zones
    top_s = surplus_df.nlargest(20, "supply")["ca_number"].tolist()
    top_d = deficit_df.nlargest(20, "demand")["ca_number"].tolist()

    cost_sub = inputs.cost_matrix.loc[top_s, top_d].copy()
    cost_km = cost_sub * 1e-3  # meters → km

    s_labels = [
        surplus_df.loc[surplus_df["ca_number"] == ca, "ca_name"].iloc[0][:14]
        for ca in top_s
    ]
    d_labels = [
        deficit_df.loc[deficit_df["ca_number"] == ca, "ca_name"].iloc[0][:14]
        for ca in top_d
    ]

    # Build allocation overlay
    alloc_map: dict[tuple[int, int], int] = {}
    for a in result.allocations:
        alloc_map[(a.surplus_ca, a.deficit_ca)] = a.units

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(cost_km.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Distance (km)", shrink=0.6)

    ax.set_xticks(range(len(top_d)))
    ax.set_yticks(range(len(top_s)))
    ax.set_xticklabels(d_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(s_labels, fontsize=7)
    ax.set_xlabel("Deficit Zones (top 20)", fontsize=10)
    ax.set_ylabel("Surplus Zones (top 20)", fontsize=10)
    ax.set_title(
        "Cost Matrix Heatmap (km) with Allocated Flow Annotations\n"
        "Numbers = estimated units transferred; "
        "[NOTE] quantities are estimated proxies, not observed inventory",
        fontsize=10,
    )

    # Annotate allocated cells
    for si, s_ca in enumerate(top_s):
        for di, d_ca in enumerate(top_d):
            units = alloc_map.get((s_ca, d_ca), 0)
            if units > 0:
                ax.text(
                    di,
                    si,
                    f"{units:,}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white"
                    if cost_km.iloc[si, di] > cost_km.values.max() * 0.5
                    else "black",
                )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cost_matrix_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_route_distance_distribution(
    result: TransportProblemResult,
    out_dir: Path,
) -> None:
    """Histogram of transfer distances weighted by units."""
    _use_style()
    if not result.allocations:
        return

    distances = [a.distance_km for a in result.allocations]
    weights = [a.units for a in result.allocations]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        distances,
        weights=weights,
        bins=30,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Transfer Distance (km)", fontsize=11)
    ax.set_ylabel("Weighted Units", fontsize=11)
    ax.set_title(
        "Distribution of Transfer Distances (weighted by estimated units)\n"
        f"Distance method: {result.distance_method}  |  "
        "[NOTE] Estimated quantities — not observed inventory",
        fontsize=10,
    )
    mean_d = np.average(distances, weights=weights)
    ax.axvline(
        mean_d,
        color="#DD8452",
        linestyle="--",
        linewidth=1.5,
        label=f"Weighted mean: {mean_d:.2f} km",
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "route_distance_distribution.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def generate_all_figures(
    result: TransportProblemResult,
    inputs: TransportProblemInputs,
    out_dir: Path,
    top_n: int = 25,
) -> None:
    """Generate and save all Phase 3 visualisation figures.

    Parameters
    ----------
    result : TransportProblemResult
    inputs : TransportProblemInputs
    out_dir : Path
        Output directory (created if absent).
    top_n : int
        Number of top allocations for bar chart.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_allocation_bar(result, out_dir, top_n=top_n)
    plot_supply_demand_comparison(result, inputs, out_dir)
    plot_cost_matrix_heatmap(result, inputs, out_dir)
    plot_route_distance_distribution(result, out_dir)
