"""Visualisations for Phase 5 Scenario & Sensitivity Analysis.

Generates:
  1. P-Median elbow curve: weighted objective vs p, per distance method.
  2. P-Median average distance vs p bar chart.
  3. Transportation sensitivity: objective comparison bar chart.
  4. Transportation sensitivity: units/routes comparison bar chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from escooter_opt.scenarios.runner import ScenarioResult

FIG_DPI: int = 150
COLORS = {"manhattan": "#4C72B0", "haversine": "#DD8452"}
SCALE_COLORS = {1.0: "#4C72B0", 0.5: "#55A868"}


def _use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        pass


def plot_pmedian_elbow(
    results: list[ScenarioResult],
    out_dir: Path,
) -> None:
    """Line chart of P-Median weighted objective vs p (one line per distance method)."""
    _use_style()
    pm = [
        r
        for r in results
        if r.config.model == "pmedian" and r.pmedian_objective_km is not None
    ]
    if not pm:
        return

    # Group by distance method
    dm_groups: dict[str, list[ScenarioResult]] = {}
    for r in pm:
        dm_groups.setdefault(r.config.distance_method, []).append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    for dm, group in dm_groups.items():
        group_sorted = sorted(group, key=lambda r: r.config.p or 0)
        p_vals = [r.config.p for r in group_sorted]
        obj_vals = [r.pmedian_objective_km for r in group_sorted]
        ax.plot(
            p_vals,
            obj_vals,
            marker="o",
            label=dm,
            linewidth=2,
            color=COLORS.get(dm, "#333333"),
        )

    ax.set_xlabel("Number of Depots (p)", fontsize=11)
    ax.set_ylabel("Weighted Objective (km-weight)", fontsize=11)
    ax.set_title(
        "P-Median: Weighted Objective vs Depot Count\n"
        "[NOTE] Quantities estimated from historical trip-flow imbalance",
        fontsize=10,
    )
    ax.legend(title="Distance method", fontsize=9)
    if dm_groups:
        all_p = sorted({r.config.p for r in pm})
        ax.set_xticks(all_p)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pmedian_elbow_curve.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pmedian_avg_distance(
    results: list[ScenarioResult],
    out_dir: Path,
) -> None:
    """Bar chart of mean assignment distance (m) vs p."""
    _use_style()
    pm = [
        r
        for r in results
        if r.config.model == "pmedian" and r.pmedian_avg_dist_m is not None
    ]
    if not pm:
        return

    dm_groups: dict[str, list[ScenarioResult]] = {}
    for r in pm:
        dm_groups.setdefault(r.config.distance_method, []).append(r)

    all_p = sorted({r.config.p for r in pm})
    x = np.arange(len(all_p))
    n_dm = len(dm_groups)
    width = 0.8 / max(n_dm, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (dm, group) in enumerate(dm_groups.items()):
        p_to_r = {r.config.p: r for r in group}
        vals = [
            p_to_r.get(p).pmedian_avg_dist_m if p_to_r.get(p) else 0.0 for p in all_p
        ]  # type: ignore[union-attr]
        bars = ax.bar(
            x + offset * width - (n_dm - 1) * width / 2,
            vals,
            width=width * 0.9,
            label=dm,
            color=COLORS.get(dm, "#888888"),
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if v:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 50,
                    f"{v:.0f}",
                    ha="center",
                    fontsize=7.5,
                )

    ax.set_xlabel("Number of Depots (p)", fontsize=11)
    ax.set_ylabel("Mean Assignment Distance (m)", fontsize=11)
    ax.set_title(
        "P-Median: Mean Zone-to-Depot Distance vs Depot Count\n"
        "[NOTE] Centroid-level distances; not actual routing distances",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in all_p])
    ax.legend(title="Distance method", fontsize=9)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pmedian_avg_distance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_transport_objective(
    results: list[ScenarioResult],
    out_dir: Path,
) -> None:
    """Bar chart comparing transportation objective values across scenarios."""
    _use_style()
    tr = [
        r
        for r in results
        if r.config.model == "transport" and r.transport_objective_km is not None
    ]
    if not tr:
        return

    labels = [r.config.label for r in tr]
    objs = [r.transport_objective_km for r in tr]
    colors = [
        COLORS.get(r.config.distance_method, "#888888")
        if r.config.demand_scale == 1.0
        else "#C44E52"
        for r in tr
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, objs, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, objs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.005,
            f"{v:,.0f}",
            ha="center",
            fontsize=8,
        )

    ax.set_ylabel("Total Transfer Cost (km)", fontsize=11)
    ax.set_title(
        "Transportation Problem: Objective Sensitivity\n"
        "[NOTE] Estimated transfer cost; not observed redistribution cost",
        fontsize=10,
    )
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "transport_objective_comparison.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def plot_transport_units_routes(
    results: list[ScenarioResult],
    out_dir: Path,
) -> None:
    """Grouped bar chart: units transferred and routes per transport scenario."""
    _use_style()
    tr = [
        r
        for r in results
        if r.config.model == "transport" and r.transport_units is not None
    ]
    if not tr:
        return

    labels = [r.config.label for r in tr]
    units = [r.transport_units for r in tr]
    routes = [r.transport_routes for r in tr]

    x = np.arange(len(tr))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(
        x - w / 2,
        units,
        width=w,
        label="Units transferred",
        color="#4C72B0",
        alpha=0.85,
    )
    b2 = ax.bar(
        x + w / 2, routes, width=w, label="Non-zero routes", color="#DD8452", alpha=0.85
    )

    for bar, v in zip(b1, units):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{v:,}",
            ha="center",
            fontsize=7.5,
        )
    for bar, v in zip(b2, routes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(v),
            ha="center",
            fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Transportation Problem: Units & Routes Sensitivity\n"
        "[NOTE] Estimated units; demand_scale < 1 models partial-period imbalance proxy",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "transport_units_routes.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def generate_all_figures(
    results: list[ScenarioResult],
    out_dir: Path,
) -> None:
    """Generate and save all Phase 5 comparative figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pmedian_elbow(results, out_dir)
    plot_pmedian_avg_distance(results, out_dir)
    plot_transport_objective(results, out_dir)
    plot_transport_units_routes(results, out_dir)
