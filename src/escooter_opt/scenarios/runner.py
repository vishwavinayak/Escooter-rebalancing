"""Scenario and Sensitivity Analysis Runner for E-Scooter Optimization.

Phase 5 – Scenario Analysis
============================

Purpose
-------
Runs structured suites of scenarios over the Transportation Problem (Phase 3)
and P-Median Depot Location (Phase 4) models.  Each scenario is a named
parameter configuration; the appropriate underlying optimizer is invoked for
every scenario.  Results are collected into a tabular summary for comparison.

Design principles
-----------------
- Re-solve the model for every scenario.  Results are never simulated by
  scaling a baseline objective.
- The same eligible dataset is reused across all scenarios in a suite unless a
  scenario explicitly changes data scope.
- Failed or infeasible scenarios are recorded without substituting fake values.
- All sensitivity conclusions are clearly labelled as depending on historical
  data, distance approximations, and the proxy imbalance assumptions of the
  underlying models.

Scenario types
--------------
``"pmedian"``
    Varies the depot count ``p`` (P-Median model).

``"transport"``
    Varies the distance cost metric (manhattan vs haversine) or, where the
    data supports it, a demand weight scaling factor applied to the surplus /
    deficit magnitudes to model aggregation periods.

IMPORTANT
---------
All sensitivity findings depend on:
  - Historical 2020 trip-flow data (not live demand).
  - Zone-centroid distance approximations.
  - Proxy surplus/deficit imbalances, not observed scooter inventory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from escooter_opt.optimization.pmedian import (
    build_pmedian_inputs,
    solve_pmedian,
)
from escooter_opt.optimization.pmedian import (
    load_eda_outputs as pmedian_load_eda,
)
from escooter_opt.optimization.transport import (
    build_transport_inputs,
    solve_transport_problem,
)
from escooter_opt.optimization.transport import (
    load_eda_outputs as transport_load_eda,
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_EDA_DIR = Path("results/eda")
DEFAULT_RESULTS_DIR = Path("results/scenarios")

# Default P-Median depot counts to sweep
DEFAULT_P_VALUES: list[int] = [1, 2, 3, 4, 5]

# Transport scenarios: (label, distance_method, demand_scale_factor)
# demand_scale_factor scales all surplus and deficit magnitudes uniformly.
# A factor of 1.0 is the baseline.  Fractional values model off-peak periods.
DEFAULT_TRANSPORT_SCENARIOS: list[dict[str, Any]] = [
    {
        "label": "baseline_manhattan",
        "distance_method": "manhattan",
        "demand_scale": 1.0,
    },
    {
        "label": "baseline_haversine",
        "distance_method": "haversine",
        "demand_scale": 1.0,
    },
    {"label": "offpeak_manhattan", "distance_method": "manhattan", "demand_scale": 0.5},
    {"label": "offpeak_haversine", "distance_method": "haversine", "demand_scale": 0.5},
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario run.

    Both ``model`` and all parameter fields are recorded so the baseline can
    be reproduced exactly from the stored configuration.
    """

    scenario_id: str
    """Unique, human-readable identifier (e.g. 'pmedian_p3', 'transport_baseline_manhattan')."""

    model: str
    """Model type: 'pmedian' or 'transport'."""

    # P-Median specific
    p: int | None = None
    """Depot count; only used when model == 'pmedian'."""

    distance_method: str = "manhattan"
    """Distance metric: 'manhattan' or 'haversine'."""

    # Transport specific
    demand_scale: float = 1.0
    """Uniform scale factor applied to all surplus/deficit magnitudes.
    1.0 = full historical imbalance (baseline).
    Values < 1.0 model off-peak or partial-day aggregation periods.
    This is a sensitivity parameter, NOT a forecast or inventory adjustment."""

    label: str = ""
    """Optional human-readable label for tables and plots."""

    notes: str = ""
    """Free-text notes describing what this scenario tests."""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.scenario_id


@dataclass
class ScenarioResult:
    """Outcome of a single scenario solve.

    DISCLAIMER: All metric values reflect historical trip-flow proxies,
    not observed vehicle inventory, and depend on centroid distance
    approximations.
    """

    config: ScenarioConfig

    status: str
    """'optimal', 'infeasible', 'error', or other solver status."""

    runtime_seconds: float
    """Wall-clock time for the solve (excludes data loading)."""

    error_message: str = ""
    """Non-empty if status != 'optimal'."""

    # ----- P-Median metrics --------------------------------------------------
    pmedian_objective_km: float | None = None
    """Demand-weighted total zone-to-depot distance (km-weight units)."""

    pmedian_avg_dist_m: float | None = None
    """Mean unweighted zone-to-depot assignment distance (meters)."""

    pmedian_total_unweighted_km: float | None = None
    """Total unweighted assignment distance (km)."""

    pmedian_depots: list[str] = field(default_factory=list)
    """Names of selected depot zones."""

    # ----- Transport metrics -------------------------------------------------
    transport_objective_km: float | None = None
    """Total estimated transfer cost (km)."""

    transport_units: int | None = None
    """Total units transferred."""

    transport_supply_util_pct: float | None = None
    """Percentage of estimated supply dispatched."""

    transport_demand_sat_pct: float | None = None
    """Percentage of estimated demand satisfied."""

    transport_routes: int | None = None
    """Number of non-zero transfer routes."""

    transport_supply_feasible: bool | None = None
    transport_demand_feasible: bool | None = None

    solved_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_row(self) -> dict[str, Any]:
        """Flatten to a one-row dict suitable for a DataFrame."""
        return {
            "scenario_id": self.config.scenario_id,
            "label": self.config.label,
            "model": self.config.model,
            "distance_method": self.config.distance_method,
            "p": self.config.p,
            "demand_scale": self.config.demand_scale,
            "status": self.status,
            "runtime_s": round(self.runtime_seconds, 3),
            # P-Median
            "pm_obj_km_wt": self.pmedian_objective_km,
            "pm_avg_dist_m": self.pmedian_avg_dist_m,
            "pm_unweighted_km": self.pmedian_total_unweighted_km,
            "pm_depots": "; ".join(self.pmedian_depots)
            if self.pmedian_depots
            else None,
            # Transport
            "tr_obj_km": self.transport_objective_km,
            "tr_units": self.transport_units,
            "tr_supply_util_pct": self.transport_supply_util_pct,
            "tr_demand_sat_pct": self.transport_demand_sat_pct,
            "tr_routes": self.transport_routes,
            "tr_supply_feasible": self.transport_supply_feasible,
            "tr_demand_feasible": self.transport_demand_feasible,
            "error_message": self.error_message or None,
            "solved_at_utc": self.solved_at_utc,
        }


# ---------------------------------------------------------------------------
# Individual scenario runners
# ---------------------------------------------------------------------------


def run_pmedian_scenario(
    config: ScenarioConfig,
    zone_flows: pd.DataFrame,
    centroids: pd.DataFrame,
    dist_matrix: pd.DataFrame,
) -> ScenarioResult:
    """Run a single P-Median scenario.

    Parameters
    ----------
    config : ScenarioConfig
        Must have ``model == 'pmedian'`` and a valid ``p`` value.
    zone_flows, centroids, dist_matrix :
        Pre-loaded EDA data (reused across scenarios for fairness).

    Returns
    -------
    ScenarioResult
    """
    t0 = time.perf_counter()
    try:
        inputs = build_pmedian_inputs(
            zone_flows=zone_flows,
            centroids=centroids,
            dist_matrix=dist_matrix,
            p=config.p,  # type: ignore[arg-type]
            distance_method=config.distance_method,
        )
        result = solve_pmedian(inputs)
        elapsed = time.perf_counter() - t0

        if result.status == "optimal":
            return ScenarioResult(
                config=config,
                status=result.status,
                runtime_seconds=elapsed,
                pmedian_objective_km=result.objective_value_km,
                pmedian_avg_dist_m=result.average_assignment_distance_meters,
                pmedian_total_unweighted_km=result.total_unweighted_distance_km,
                pmedian_depots=[sd.depot_name for sd in result.selected_depots],
            )
        return ScenarioResult(
            config=config,
            status=result.status,
            runtime_seconds=elapsed,
            error_message=result.solver_message,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return ScenarioResult(
            config=config,
            status="error",
            runtime_seconds=elapsed,
            error_message=str(exc),
        )


def run_transport_scenario(
    config: ScenarioConfig,
    zone_flows: pd.DataFrame,
    dist_matrix: pd.DataFrame,
) -> ScenarioResult:
    """Run a single Transportation Problem scenario.

    The ``demand_scale`` factor is applied uniformly to all surplus and deficit
    magnitudes before solving.  This models sensitivity to the magnitude of
    the historical flow imbalance (e.g. partial-day aggregation).

    NOTE: Scaled values are estimated proxies; they do not represent measured
    inventory at any aggregation level.

    Parameters
    ----------
    config : ScenarioConfig
        Must have ``model == 'transport'``.
    zone_flows, dist_matrix :
        Pre-loaded EDA data.

    Returns
    -------
    ScenarioResult
    """
    t0 = time.perf_counter()
    try:
        # Apply demand scale to zone flows before building inputs
        scaled_flows = zone_flows.copy()
        if config.demand_scale != 1.0:
            # Scale net_flow; flow_category classification stays the same
            scaled_flows["net_flow"] = (
                (scaled_flows["net_flow"] * config.demand_scale).round().astype(int)
            )
            # Recompute turnover-related cols consistently
            scaled_flows["arrivals"] = (
                (scaled_flows["arrivals"] * config.demand_scale).round().astype(int)
            )
            scaled_flows["departures"] = (
                (scaled_flows["departures"] * config.demand_scale).round().astype(int)
            )

        inputs = build_transport_inputs(
            zone_flows=scaled_flows,
            dist_matrix=dist_matrix,
            distance_method=config.distance_method,
        )
        result = solve_transport_problem(inputs)
        elapsed = time.perf_counter() - t0

        if result.status == "optimal":
            return ScenarioResult(
                config=config,
                status=result.status,
                runtime_seconds=elapsed,
                transport_objective_km=result.objective_value_km,
                transport_units=result.total_units_transferred,
                transport_supply_util_pct=result.supply_utilisation_pct,
                transport_demand_sat_pct=result.demand_satisfaction_pct,
                transport_routes=len(result.allocations),
                transport_supply_feasible=result.rounded_supply_feasible,
                transport_demand_feasible=result.rounded_demand_feasible,
            )
        return ScenarioResult(
            config=config,
            status=result.status,
            runtime_seconds=elapsed,
            error_message=result.solver_message,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return ScenarioResult(
            config=config,
            status="error",
            runtime_seconds=elapsed,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Suite builder helpers
# ---------------------------------------------------------------------------


def build_pmedian_suite(
    p_values: list[int],
    distance_method: str = "manhattan",
) -> list[ScenarioConfig]:
    """Build a list of P-Median scenario configs for a range of p values.

    Parameters
    ----------
    p_values : list[int]
        Depot counts to evaluate.  Must be sorted ascending for sensible plots.
    distance_method : str
        Distance metric shared across all configs.

    Returns
    -------
    list[ScenarioConfig]
    """
    configs: list[ScenarioConfig] = []
    for p in p_values:
        cfg_id = f"pmedian_p{p}_{distance_method}"
        configs.append(
            ScenarioConfig(
                scenario_id=cfg_id,
                model="pmedian",
                p=p,
                distance_method=distance_method,
                label=f"p={p} ({distance_method})",
                notes=f"P-Median with {p} depot(s), {distance_method} distance.",
            )
        )
    return configs


def build_transport_suite(
    scenarios: list[dict[str, Any]] | None = None,
) -> list[ScenarioConfig]:
    """Build a list of Transportation Problem scenario configs.

    Parameters
    ----------
    scenarios : list[dict] | None
        Each dict may have keys: label, distance_method, demand_scale.
        Defaults to ``DEFAULT_TRANSPORT_SCENARIOS``.

    Returns
    -------
    list[ScenarioConfig]
    """
    raw = scenarios or DEFAULT_TRANSPORT_SCENARIOS
    configs: list[ScenarioConfig] = []
    for raw_cfg in raw:
        label = raw_cfg.get("label", "transport")
        dm = raw_cfg.get("distance_method", "manhattan")
        scale = float(raw_cfg.get("demand_scale", 1.0))
        cfg_id = f"transport_{label}"
        notes = (
            f"Transportation problem, {dm} distance, "
            f"demand_scale={scale} "
            f"({'baseline' if scale == 1.0 else 'sensitivity: scaled imbalance proxy'})."
        )
        configs.append(
            ScenarioConfig(
                scenario_id=cfg_id,
                model="transport",
                distance_method=dm,
                demand_scale=scale,
                label=label,
                notes=notes,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_scenario_suite(
    configs: list[ScenarioConfig],
    eda_dir: Path = DEFAULT_EDA_DIR,
) -> list[ScenarioResult]:
    """Execute a list of scenario configs and return results.

    Data is loaded once per (model, distance_method) pair and reused across
    all scenarios sharing those parameters, ensuring fair comparison.

    Parameters
    ----------
    configs : list[ScenarioConfig]
    eda_dir : Path
        Directory containing Phase 2 EDA CSV files.

    Returns
    -------
    list[ScenarioResult]
        One result per config, in input order.
    """
    # Pre-load EDA data for each distinct distance_method used
    pmedian_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    transport_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    results: list[ScenarioResult] = []

    for cfg in configs:
        dm = cfg.distance_method

        if cfg.model == "pmedian":
            if dm not in pmedian_cache:
                pmedian_cache[dm] = pmedian_load_eda(eda_dir, dm)
            zone_flows, centroids, dist_matrix = pmedian_cache[dm]
            sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)

        elif cfg.model == "transport":
            if dm not in transport_cache:
                transport_cache[dm] = transport_load_eda(eda_dir, dm)
            zone_flows, dist_matrix = transport_cache[dm]
            sr = run_transport_scenario(cfg, zone_flows, dist_matrix)

        else:
            sr = ScenarioResult(
                config=cfg,
                status="error",
                runtime_seconds=0.0,
                error_message=f"Unknown model type: '{cfg.model}'",
            )

        results.append(sr)
        _log_result(sr)

    return results


def _log_result(sr: ScenarioResult) -> None:
    """Print a single-line progress log for a completed scenario."""
    if sr.config.model == "pmedian":
        metric = (
            f"obj={sr.pmedian_objective_km:,.1f} km-wt  avg={sr.pmedian_avg_dist_m:.0f} m"
            if sr.pmedian_objective_km is not None
            else f"FAILED: {sr.error_message}"
        )
    else:
        metric = (
            f"obj={sr.transport_objective_km:,.1f} km  units={sr.transport_units}"
            if sr.transport_objective_km is not None
            else f"FAILED: {sr.error_message}"
        )
    print(
        f"  [{sr.config.scenario_id}]  {sr.status}  {metric}  ({sr.runtime_seconds:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Results table & I/O
# ---------------------------------------------------------------------------


def results_to_dataframe(results: list[ScenarioResult]) -> pd.DataFrame:
    """Convert a list of ScenarioResult objects to a flat DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per scenario.  Columns differ by model type; irrelevant
        columns contain NaN.
    """
    return pd.DataFrame([r.to_row() for r in results])


def save_suite_results(
    results: list[ScenarioResult],
    out_dir: Path,
    suite_name: str = "scenarios",
) -> None:
    """Save scenario suite results to CSV and JSON.

    Files produced:
      - {suite_name}_results.csv  — flat results table
      - {suite_name}_results.json — full structured results
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    df.to_csv(out_dir / f"{suite_name}_results.csv", index=False)

    full = {
        "suite_name": suite_name,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_scenarios": len(results),
        "scenarios": [
            {
                "config": asdict(r.config),
                "result_row": r.to_row(),
            }
            for r in results
        ],
    }
    with open(out_dir / f"{suite_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str)


def format_suite_report(
    results: list[ScenarioResult], title: str = "Scenario Suite"
) -> str:
    """Format a human-readable comparison report.

    IMPORTANT: All metrics reflect historical trip-flow proxies,
    centroid distance approximations, and estimated imbalances.
    They do not represent measured inventory or guaranteed outcomes.
    """
    pmedian_results = [r for r in results if r.config.model == "pmedian"]
    transport_results = [r for r in results if r.config.model == "transport"]

    lines = [
        "=" * 72,
        f"Chicago E-Scooter Phase 5: {title}",
        "=" * 72,
        "",
        "DISCLAIMER: All metrics reflect historical 2020 trip-flow proxies,",
        "centroid distance approximations, and estimated demand imbalances.",
        "They do NOT represent measured inventory or approved depot sites.",
        "",
    ]

    if pmedian_results:
        lines += [
            "--- P-Median Depot Location: Sensitivity to Depot Count (p) ---",
            f"{'Scenario':<30}  {'Status':<8}  {'Obj(km-wt)':>14}  {'Avg dist(m)':>11}  {'t(s)':>5}",
            "-" * 74,
        ]
        for r in pmedian_results:
            obj = (
                f"{r.pmedian_objective_km:>14,.1f}"
                if r.pmedian_objective_km is not None
                else f"{'N/A':>14}"
            )
            avg = (
                f"{r.pmedian_avg_dist_m:>11.0f}"
                if r.pmedian_avg_dist_m is not None
                else f"{'N/A':>11}"
            )
            lines.append(
                f"{r.config.label:<30}  {r.status:<8}  {obj}  {avg}  {r.runtime_seconds:>5.2f}"
            )

        # Marginal reductions per distance method
        dm_groups: dict[str, list[ScenarioResult]] = {}
        for r in pmedian_results:
            if r.pmedian_objective_km is not None:
                dm_groups.setdefault(r.config.distance_method, []).append(r)

        has_multi = any(len(g) >= 2 for g in dm_groups.values())
        if has_multi:
            lines += ["", "  Marginal reduction in objective (pairwise):"]
            for group in dm_groups.values():
                sorted_group = sorted(group, key=lambda x: x.config.p or 0)
                if len(sorted_group) >= 2:
                    for prev, curr in itertools.pairwise(sorted_group):
                        if prev.pmedian_objective_km and curr.pmedian_objective_km:
                            delta = (
                                prev.pmedian_objective_km - curr.pmedian_objective_km
                            )
                            pct = delta / prev.pmedian_objective_km * 100
                            lines.append(
                                f"    {prev.config.label} → {curr.config.label}: −{delta:,.1f} ({pct:.1f}%)"
                            )
        lines.append("")

    if transport_results:
        lines += [
            "--- Transportation Problem: Cost Assumption Sensitivity ---",
            f"{'Scenario':<30}  {'Status':<8}  {'Obj(km)':>12}  {'Units':>6}  {'Routes':>6}  {'t(s)':>5}",
            "-" * 74,
        ]
        for r in transport_results:
            obj = (
                f"{r.transport_objective_km:>12,.1f}"
                if r.transport_objective_km is not None
                else f"{'N/A':>12}"
            )
            units = (
                f"{r.transport_units:>6}"
                if r.transport_units is not None
                else f"{'N/A':>6}"
            )
            routes = (
                f"{r.transport_routes:>6}"
                if r.transport_routes is not None
                else f"{'N/A':>6}"
            )
            lines.append(
                f"{r.config.label:<30}  {r.status:<8}  {obj}  {units}  {routes}  {r.runtime_seconds:>5.2f}"
            )
        lines.append("")

    lines += ["=" * 72]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Phase 5 scenario and sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5: Scenario & Sensitivity Analysis for E-Scooter Optimization.\n\n"
            "DISCLAIMER: Results reflect historical trip-flow proxies and estimated "
            "imbalances, not observed scooter inventory."
        )
    )
    parser.add_argument(
        "--eda-dir",
        type=Path,
        default=DEFAULT_EDA_DIR,
        help=f"Phase 2 EDA output directory (default: {DEFAULT_EDA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Output directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--p-values",
        type=int,
        nargs="+",
        default=DEFAULT_P_VALUES,
        help="Depot counts for P-Median sweep (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--distance-method",
        choices=["manhattan", "haversine", "both"],
        default="manhattan",
        help="Distance metric for P-Median sweep (default: manhattan).",
    )
    parser.add_argument(
        "--no-transport",
        action="store_true",
        help="Skip transportation scenario suite.",
    )
    parser.add_argument(
        "--no-pmedian",
        action="store_true",
        help="Skip P-Median scenario suite.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("Chicago E-Scooter: Phase 5 Scenario & Sensitivity Analysis")
    print("=" * 72)
    print(
        "\nDISCLAIMER: All results reflect historical 2020 trip-flow proxies,\n"
        "distance approximations, and estimated imbalances.\n"
    )

    all_configs: list[ScenarioConfig] = []

    # P-Median suite(s)
    if not args.no_pmedian:
        dm_list = (
            ["manhattan", "haversine"]
            if args.distance_method == "both"
            else [args.distance_method]
        )
        for dm in dm_list:
            all_configs.extend(build_pmedian_suite(args.p_values, distance_method=dm))

    # Transportation suite
    if not args.no_transport:
        all_configs.extend(build_transport_suite())

    print(f"Running {len(all_configs)} scenarios ...\n")
    results = run_scenario_suite(all_configs, eda_dir=args.eda_dir)

    print()
    print(format_suite_report(results, title="Scenario & Sensitivity Analysis"))

    save_suite_results(results, args.results_dir, suite_name="scenarios")
    print(f"\nResults saved to: {args.results_dir.resolve()}")

    if not args.no_figures:
        try:
            from escooter_opt.scenarios.visualize import generate_all_figures

            fig_dir = args.results_dir / "figures"
            generate_all_figures(results, fig_dir)
            print(f"Figures saved to:  {fig_dir.resolve()}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: figure generation failed — {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
