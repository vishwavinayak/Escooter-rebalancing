"""Transportation Problem for E-Scooter Rebalancing.

Phase 3 – Transportation Model
================================
Formulation
-----------
Given:
  S = set of potential surplus zones  (I_i > 0,  s_i = I_i)
  D = set of potential deficit zones  (I_i < 0,  d_j = -I_j)

Decision variable:
  x_{ij} ∈ Z⁺  — integer units transferred from surplus zone i to deficit zone j.

Objective (minimise total estimated transfer cost):
  min  Σ_{i∈S} Σ_{j∈D}  c_{ij} · x_{ij}

Subject to:
  Σ_j x_{ij} ≤ s_i        ∀ i ∈ S          (supply ≤ surplus)
  Σ_i x_{ij} ≥ d_j        ∀ j ∈ D          (demand ≥ deficit)
  x_{ij} ≥ 0, integer     ∀ i, j           (non-negative integers)

Cost matrix c_{ij}:
  Default: Manhattan grid distance in meters (derived from observed centroid coordinates,
  documented in Phase 2).  A haversine variant is available for sensitivity testing.

IMPORTANT LIMITATIONS
---------------------
The imbalance quantities s_i and d_j are estimated from historical trip-flow
accumulation.  They are NOT measured scooter inventory.  Any rebalancing quantities
produced here are therefore *estimated*, not observed, and should be treated as
operational heuristics subject to ground-truth validation.

Feasibility handling:
  Total supply Σ s_i may not equal total demand Σ d_j (net flow sums to 0 across
  all zones, but after filtering to surplus/deficit subsets the aggregate may differ
  if the LP is not perfectly balanced).  We add a slack surplus or slack demand zone
  with zero cost to balance the problem before solving, then strip the slack from
  the output.  The original imbalance magnitudes are preserved.

Solver:
  scipy.optimize.linprog with the revised-simplex method.
  Integer constraints are enforced by rounding the continuous LP solution (the
  continuous optimum is typically integral for balanced transportation problems with
  integer supply/demand vectors; post-rounding feasibility is verified).
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDA_DIR = Path("results/eda")
DEFAULT_RESULTS_DIR = Path("results/transport")
DEFAULT_DISTANCE_METHOD = "manhattan"

# Conversion: meters → km for cost display
M_TO_KM: float = 1 / 1000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TransportProblemInputs:
    """Structured inputs for the transportation problem.

    All quantities are estimated proxies from historical trip flows.
    """

    surplus_zones: pd.DataFrame
    """Zones with positive net flow.  Columns: ca_number, ca_name, net_flow (supply s_i)."""

    deficit_zones: pd.DataFrame
    """Zones with negative net flow.  Columns: ca_number, ca_name, net_flow, demand d_j."""

    cost_matrix: pd.DataFrame
    """|S| × |D| DataFrame of pairwise transfer costs c_{ij} (meters).
    Index = surplus ca_number, columns = deficit ca_number."""

    distance_method: str = DEFAULT_DISTANCE_METHOD
    """Distance metric used: 'manhattan' or 'haversine'."""

    total_supply: int = 0
    """Sum of surplus magnitudes Σ s_i."""

    total_demand: int = 0
    """Sum of deficit magnitudes Σ d_j."""


@dataclass
class TransportAllocation:
    """Single transfer recommendation."""

    surplus_ca: int
    surplus_name: str
    deficit_ca: int
    deficit_name: str
    units: int
    distance_meters: float
    cost_unit_meters: float  # c_{ij} (same as distance_meters; listed for clarity)
    total_cost_meters: float  # units × distance_meters

    @property
    def distance_km(self) -> float:
        return round(self.distance_meters * M_TO_KM, 3)

    @property
    def total_cost_km(self) -> float:
        return round(self.total_cost_meters * M_TO_KM, 3)


@dataclass
class TransportProblemResult:
    """Full result of the transportation LP solve.

    LIMITATION: All quantities are estimated from historical trip-flow imbalance,
    not observed vehicle inventory.
    """

    status: str
    """'optimal', 'infeasible', 'unbounded', or 'error'."""

    solver_message: str
    """Human-readable solver status message."""

    objective_value_meters: float
    """Optimal (continuous) objective value in meter-units (Σ c_{ij} x_{ij})."""

    objective_value_km: float
    """Optimal objective in km-units."""

    total_supply: int
    total_demand: int
    surplus_zone_count: int
    deficit_zone_count: int

    allocations: list[TransportAllocation]
    """Non-zero integer transfer recommendations, sorted by descending total cost."""

    total_units_transferred: int
    supply_utilisation_pct: float
    demand_satisfaction_pct: float

    # Feasibility verification after integer rounding
    rounded_supply_feasible: bool
    rounded_demand_feasible: bool

    distance_method: str
    solved_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert allocation dataclasses to plain dicts
        d["allocations"] = [asdict(a) for a in self.allocations]
        return d


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def build_transport_inputs(
    zone_flows: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    distance_method: str = DEFAULT_DISTANCE_METHOD,
) -> TransportProblemInputs:
    """Build structured Transportation Problem inputs from EDA outputs.

    Parameters
    ----------
    zone_flows : pd.DataFrame
        Output of ``calculate_zone_flows()`` — must contain columns
        ``ca_number``, ``ca_name``, ``net_flow``, ``flow_category``.
    dist_matrix : pd.DataFrame
        Square distance matrix (meters) indexed and columned by ``ca_number``.
        Output of ``compute_zone_distance_matrix()``.
    distance_method : str
        Label for the distance metric ('manhattan' or 'haversine').

    Returns
    -------
    TransportProblemInputs
    """
    surplus_mask = zone_flows["flow_category"] == "potential_surplus"
    deficit_mask = zone_flows["flow_category"] == "potential_deficit"

    surplus = zone_flows[surplus_mask][["ca_number", "ca_name", "net_flow"]].copy()
    deficit = zone_flows[deficit_mask][["ca_number", "ca_name", "net_flow"]].copy()

    surplus = surplus.reset_index(drop=True)
    deficit = deficit.reset_index(drop=True)

    # Supply s_i = net_flow (positive); demand d_j = |net_flow| (positive)
    surplus["supply"] = surplus["net_flow"].astype(int)
    deficit["demand"] = deficit["net_flow"].abs().astype(int)

    total_supply = int(surplus["supply"].sum())
    total_demand = int(deficit["demand"].sum())

    # Slice cost matrix: rows = surplus zones, cols = deficit zones
    s_cas = surplus["ca_number"].tolist()
    d_cas = deficit["ca_number"].tolist()

    # Align index/column types
    dist_index = dist_matrix.index.astype(int)
    dist_cols = dist_matrix.columns.astype(int)
    dist_matrix.index = dist_index
    dist_matrix.columns = dist_cols

    # Validate all zones have coordinates in distance matrix
    missing_s = [ca for ca in s_cas if ca not in dist_matrix.index]
    missing_d = [ca for ca in d_cas if ca not in dist_matrix.columns]
    if missing_s or missing_d:
        raise KeyError(
            f"Distance matrix missing zones — surplus: {missing_s}, deficit: {missing_d}"
        )

    cost_matrix = dist_matrix.loc[s_cas, d_cas].copy()

    return TransportProblemInputs(
        surplus_zones=surplus,
        deficit_zones=deficit,
        cost_matrix=cost_matrix,
        distance_method=distance_method,
        total_supply=total_supply,
        total_demand=total_demand,
    )


def _balance_problem(
    supply: np.ndarray, demand: np.ndarray, cost: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance supply and demand for the transportation LP.

    Adds a slack surplus or slack deficit node with zero cost so that
    Σ supply = Σ demand, which is required for the standard balanced
    transportation problem formulation.

    Returns balanced (supply, demand, cost).
    """
    ts = int(supply.sum())
    td = int(demand.sum())

    if ts == td:
        return supply.copy(), demand.copy(), cost.copy()

    if ts > td:
        # Add a dummy deficit zone absorbing the excess supply at zero cost
        slack = ts - td
        demand_bal = np.append(demand, slack)
        cost_bal = np.hstack([cost, np.zeros((len(supply), 1))])
        return supply.copy(), demand_bal, cost_bal
    else:
        # Add a dummy surplus zone providing the deficit shortfall at zero cost
        slack = td - ts
        supply_bal = np.append(supply, slack)
        cost_bal = np.vstack([cost, np.zeros((1, len(demand)))])
        return supply_bal, demand.copy(), cost_bal


def solve_transport_problem(
    inputs: TransportProblemInputs,
) -> TransportProblemResult:
    """Solve the transportation problem via scipy.optimize.linprog.

    Uses the revised-simplex method on the continuous LP relaxation.
    For balanced transportation problems with integer supply and demand,
    the continuous optimum is typically integral (unimodularity of the
    constraint matrix).  Post-solve, the solution is rounded to integers
    and feasibility is re-verified.

    Parameters
    ----------
    inputs : TransportProblemInputs
        Structured inputs from ``build_transport_inputs()``.

    Returns
    -------
    TransportProblemResult
        Full result with allocation recommendations.
    """
    surplus_df = inputs.surplus_zones
    deficit_df = inputs.deficit_zones
    cost_mat = inputs.cost_matrix.to_numpy(dtype=float)

    m = len(surplus_df)  # number of surplus zones
    n = len(deficit_df)  # number of deficit zones

    supply_vec = surplus_df["supply"].to_numpy(dtype=float)
    demand_vec = deficit_df["demand"].to_numpy(dtype=float)

    # Balance the problem
    supply_bal, demand_bal, cost_bal = _balance_problem(
        supply_vec, demand_vec, cost_mat
    )
    m_bal = len(supply_bal)
    n_bal = len(demand_bal)

    # Flatten cost matrix (row-major) for linprog c vector
    c = cost_bal.flatten()

    # Constraints:
    # Σ_j x_{ij} ≤ s_i  (supply upper bounds — inequality)
    # Σ_i x_{ij} ≥ d_j  (demand lower bounds — converted to ≤ by negation)
    # x_{ij} ≥ 0         (bounds)

    # Supply constraints: A_ub @ x ≤ b_ub
    A_supply = np.zeros((m_bal, m_bal * n_bal))
    for i in range(m_bal):
        for j in range(n_bal):
            A_supply[i, i * n_bal + j] = 1.0
    b_supply = supply_bal

    # Demand constraints: -Σ_i x_{ij} ≤ -d_j
    A_demand = np.zeros((n_bal, m_bal * n_bal))
    for j in range(n_bal):
        for i in range(m_bal):
            A_demand[j, i * n_bal + j] = -1.0
    b_demand = -demand_bal

    A_ub = np.vstack([A_supply, A_demand])
    b_ub = np.concatenate([b_supply, b_demand])

    # All variables ≥ 0
    bounds = [(0, None)] * (m_bal * n_bal)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",  # HiGHS is faster and more robust than revised simplex
            options={"disp": False},
        )

    # Map scipy status codes
    status_map = {
        0: "optimal",
        1: "iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_difficulty",
    }
    status = status_map.get(result.status, "error")

    if status not in ("optimal",):
        return TransportProblemResult(
            status=status,
            solver_message=result.message,
            objective_value_meters=float("nan"),
            objective_value_km=float("nan"),
            total_supply=inputs.total_supply,
            total_demand=inputs.total_demand,
            surplus_zone_count=m,
            deficit_zone_count=n,
            allocations=[],
            total_units_transferred=0,
            supply_utilisation_pct=0.0,
            demand_satisfaction_pct=0.0,
            rounded_supply_feasible=False,
            rounded_demand_feasible=False,
            distance_method=inputs.distance_method,
        )

    # Reshape solution
    x_bal = result.x.reshape(m_bal, n_bal)

    # Strip slack columns/rows (keep only original m × n block)
    x_orig = x_bal[:m, :n]

    # Round to integers (transportation polytope is TDI → LP optimum is integral)
    x_int = np.round(x_orig).astype(int)
    x_int = np.maximum(x_int, 0)  # guard against -0 from floating point

    # Verify rounded feasibility
    supply_used = x_int.sum(axis=1)
    demand_met = x_int.sum(axis=0)
    rounded_supply_feasible = bool(np.all(supply_used <= supply_vec))
    rounded_demand_feasible = bool(np.all(demand_met >= demand_vec))

    # Build allocation list (non-zero transfers only)
    allocations: list[TransportAllocation] = []
    s_list = surplus_df[["ca_number", "ca_name"]].values.tolist()
    d_list = deficit_df[["ca_number", "ca_name"]].values.tolist()

    for i in range(m):
        for j in range(n):
            units = int(x_int[i, j])
            if units > 0:
                s_ca, s_name = s_list[i]
                d_ca, d_name = d_list[j]
                dist_m = float(cost_mat[i, j])
                allocations.append(
                    TransportAllocation(
                        surplus_ca=int(s_ca),
                        surplus_name=str(s_name),
                        deficit_ca=int(d_ca),
                        deficit_name=str(d_name),
                        units=units,
                        distance_meters=round(dist_m, 1),
                        cost_unit_meters=round(dist_m, 1),
                        total_cost_meters=round(units * dist_m, 1),
                    )
                )

    allocations.sort(key=lambda a: -a.total_cost_meters)

    obj_value = float(np.sum(cost_mat * x_int))
    total_transferred = int(x_int.sum())

    supply_util = (
        (total_transferred / inputs.total_supply * 100) if inputs.total_supply else 0.0
    )
    demand_sat = (
        min(total_transferred / inputs.total_demand * 100, 100.0)
        if inputs.total_demand
        else 0.0
    )

    return TransportProblemResult(
        status=status,
        solver_message=result.message,
        objective_value_meters=round(obj_value, 2),
        objective_value_km=round(obj_value * M_TO_KM, 3),
        total_supply=inputs.total_supply,
        total_demand=inputs.total_demand,
        surplus_zone_count=m,
        deficit_zone_count=n,
        allocations=allocations,
        total_units_transferred=total_transferred,
        supply_utilisation_pct=round(supply_util, 2),
        demand_satisfaction_pct=round(demand_sat, 2),
        rounded_supply_feasible=rounded_supply_feasible,
        rounded_demand_feasible=rounded_demand_feasible,
        distance_method=inputs.distance_method,
    )


def format_solution_report(result: TransportProblemResult, top_n: int = 20) -> str:
    """Format a human-readable solution report string.

    Parameters
    ----------
    result : TransportProblemResult
    top_n : int
        Number of highest-cost allocations to list.
    """
    lines = [
        "=" * 70,
        "Chicago E-Scooter Phase 3: Transportation Problem Solution",
        "=" * 70,
        "",
        "IMPORTANT: All quantities are estimated from historical trip-flow",
        "imbalance, NOT from observed scooter inventory.  These are heuristic",
        "rebalancing recommendations subject to ground-truth validation.",
        "",
        f"Status:          {result.status}",
        f"Solver message:  {result.solver_message}",
        f"Distance method: {result.distance_method}",
        f"Solved at (UTC): {result.solved_at_utc}",
        "",
        "--- Problem Scale ---",
        f"Surplus zones:          {result.surplus_zone_count}",
        f"Deficit zones:          {result.deficit_zone_count}",
        f"Total supply (est.):    {result.total_supply:,} units",
        f"Total demand (est.):    {result.total_demand:,} units",
        "",
        "--- Solution Summary ---",
        f"Objective value:        {result.objective_value_km:,.1f} km  ({result.objective_value_meters:,.0f} m)",
        f"Units transferred:      {result.total_units_transferred:,}",
        f"Supply utilisation:     {result.supply_utilisation_pct:.1f}%",
        f"Demand satisfaction:    {result.demand_satisfaction_pct:.1f}%",
        f"Supply feasible (int):  {result.rounded_supply_feasible}",
        f"Demand feasible (int):  {result.rounded_demand_feasible}",
        "",
        f"--- Top {top_n} Allocations by Total Transfer Cost ---",
        f"{'Surplus Zone':<26} {'Deficit Zone':<26} {'Units':>6} {'Dist(km)':>9} {'Cost(km)':>10}",
        "-" * 82,
    ]

    for a in result.allocations[:top_n]:
        lines.append(
            f"{a.surplus_name:<26} {a.deficit_name:<26} "
            f"{a.units:>6,} {a.distance_km:>9.2f} {a.total_cost_km:>10.1f}"
        )

    if len(result.allocations) > top_n:
        lines.append(
            f"  ... and {len(result.allocations) - top_n} more allocations "
            f"(see allocations_detail.csv)"
        )

    lines += [
        "",
        f"Total non-zero transfer routes: {len(result.allocations)}",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_eda_outputs(
    eda_dir: Path, distance_method: str = DEFAULT_DISTANCE_METHOD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load zone_flows and distance matrix from Phase 2 EDA results directory.

    Parameters
    ----------
    eda_dir : Path
        Directory containing EDA CSV outputs (results/eda).
    distance_method : str
        'manhattan' or 'haversine'.

    Returns
    -------
    (zone_flows, dist_matrix)
    """
    flows_path = eda_dir / "zone_flows_and_imbalance.csv"
    if not flows_path.is_file():
        raise FileNotFoundError(
            f"Zone flows file not found: {flows_path.resolve()}\n"
            "Run Phase 2 EDA first: uv run escooter-eda"
        )

    if distance_method == "haversine":
        dist_path = eda_dir / "zone_distances_haversine_meters.csv"
    else:
        dist_path = eda_dir / "zone_distances_manhattan_meters.csv"

    if not dist_path.is_file():
        raise FileNotFoundError(
            f"Distance matrix not found: {dist_path.resolve()}\n"
            "Run Phase 2 EDA first: uv run escooter-eda"
        )

    zone_flows = pd.read_csv(flows_path)
    dist_matrix = pd.read_csv(dist_path, index_col=0)
    dist_matrix.index = dist_matrix.index.astype(int)
    dist_matrix.columns = dist_matrix.columns.astype(int)

    return zone_flows, dist_matrix


def save_results(result: TransportProblemResult, out_dir: Path) -> None:
    """Save transportation problem results to CSV and JSON files.

    Parameters
    ----------
    result : TransportProblemResult
    out_dir : Path
        Output directory (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON result
    result_dict = result.to_dict()
    with open(out_dir / "transport_solution.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    # Allocations detail CSV
    if result.allocations:
        rows = [
            {
                "surplus_ca": a.surplus_ca,
                "surplus_name": a.surplus_name,
                "deficit_ca": a.deficit_ca,
                "deficit_name": a.deficit_name,
                "units": a.units,
                "distance_meters": a.distance_meters,
                "distance_km": a.distance_km,
                "total_cost_meters": a.total_cost_meters,
                "total_cost_km": a.total_cost_km,
            }
            for a in result.allocations
        ]
        pd.DataFrame(rows).to_csv(out_dir / "allocations_detail.csv", index=False)

    # Summary metrics CSV
    summary = {
        "status": result.status,
        "objective_value_meters": result.objective_value_meters,
        "objective_value_km": result.objective_value_km,
        "total_supply": result.total_supply,
        "total_demand": result.total_demand,
        "surplus_zone_count": result.surplus_zone_count,
        "deficit_zone_count": result.deficit_zone_count,
        "total_units_transferred": result.total_units_transferred,
        "supply_utilisation_pct": result.supply_utilisation_pct,
        "demand_satisfaction_pct": result.demand_satisfaction_pct,
        "rounded_supply_feasible": result.rounded_supply_feasible,
        "rounded_demand_feasible": result.rounded_demand_feasible,
        "non_zero_routes": len(result.allocations),
        "distance_method": result.distance_method,
        "solved_at_utc": result.solved_at_utc,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "transport_summary.csv", index=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Phase 3 transportation problem solver."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: Solve the E-Scooter Rebalancing Transportation Problem.\n\n"
            "IMPORTANT: Supply/demand quantities are estimated from historical trip-flow "
            "imbalance, not observed scooter inventory."
        )
    )
    parser.add_argument(
        "--eda-dir",
        type=Path,
        default=DEFAULT_EDA_DIR,
        help="Directory containing Phase 2 EDA CSV outputs (default: results/eda).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write Phase 3 output files (default: results/transport).",
    )
    parser.add_argument(
        "--distance-method",
        choices=["manhattan", "haversine"],
        default=DEFAULT_DISTANCE_METHOD,
        help="Distance metric for cost matrix (default: manhattan).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top allocations to display in the console report (default: 20).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip generating visualisation figures.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Chicago E-Scooter: Phase 3 Transportation Problem")
    print("=" * 70)
    print(
        "\nIMPORTANT: Supply/demand quantities are estimated from historical\n"
        "trip-flow imbalance, NOT observed vehicle inventory.\n"
    )

    # Load EDA outputs
    print(f"Loading EDA outputs from: {args.eda_dir.resolve()}")
    zone_flows, dist_matrix = load_eda_outputs(args.eda_dir, args.distance_method)
    print(f"  Zone flows loaded: {len(zone_flows)} zones")
    print(
        f"  Distance matrix: {dist_matrix.shape[0]}×{dist_matrix.shape[1]} ({args.distance_method})"
    )

    # Build inputs
    inputs = build_transport_inputs(zone_flows, dist_matrix, args.distance_method)
    print(f"\nSurplus zones (S): {len(inputs.surplus_zones)}")
    print(f"Deficit zones (D): {len(inputs.deficit_zones)}")
    print(f"Total supply:      {inputs.total_supply:,}")
    print(f"Total demand:      {inputs.total_demand:,}")

    # Solve
    print("\nSolving transportation LP ...")
    result = solve_transport_problem(inputs)

    # Print report
    print()
    print(format_solution_report(result, top_n=args.top_n))

    # Save outputs
    save_results(result, args.results_dir)
    print(f"\nResults saved to: {args.results_dir.resolve()}")

    # Generate figures
    if not args.no_figures:
        try:
            from escooter_opt.optimization.visualize import generate_all_figures

            fig_dir = args.results_dir / "figures"
            generate_all_figures(result, inputs, fig_dir)
            print(f"Figures saved to:  {fig_dir.resolve()}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: figure generation failed — {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
