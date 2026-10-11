"""P-Median Facility Location Model for E-Scooter Depot Siting.

Phase 4 – P-Median Integer Program
=====================================

Formulation
-----------
Given:
  I  = set of eligible demand zones (all 77 Chicago Community Areas with valid centroids)
  J  = set of candidate depot locations (by default = I, i.e. every zone is a candidate)
  w_i ≥ 0  = demand weight for zone i  (turnover = arrivals + departures, from Phase 2 EDA)
  d_{ij}   = distance from demand zone i to candidate depot j  (meters, Manhattan or haversine)
  p        = number of depots to select  (configurable; 1 ≤ p ≤ |J|)

Decision variables:
  y_j ∈ {0, 1}  — 1 if candidate depot j is selected, 0 otherwise
  z_{ij} ∈ {0, 1} — 1 if demand zone i is assigned to depot j, 0 otherwise

Objective (minimise demand-weighted total assignment distance):
  min  Σ_{i∈I} Σ_{j∈J}  w_i · d_{ij} · z_{ij}

Subject to:
  Σ_j z_{ij} = 1          ∀ i ∈ I          (C1) each zone assigned to exactly one depot
  z_{ij} ≤ y_j            ∀ i ∈ I, j ∈ J   (C2) assign only to a selected depot
  Σ_j y_j = p                               (C3) exactly p depots selected
  y_j ∈ {0,1}             ∀ j ∈ J
  z_{ij} ∈ {0,1}          ∀ i ∈ I, j ∈ J

IMPORTANT LIMITATIONS
---------------------
- Candidate depot locations are Community Area centroids derived from observed
  GPS coordinates in the dataset.  They are **modelled optimal locations**, not
  approved or physically available depot properties.
- Demand weights (turnover) are derived from historical trip flows, not live demand.
- Distances use zone centroids as proxies for within-zone facility positions.
- No time dimension; the model uses aggregate historical data.
- No vehicle capacity, fleet size, or operational constraints.

Solver
------
Uses scipy.optimize.milp (HiGHS backend) with binary integer constraints.
Variable ordering: y_0 … y_{|J|-1}, z_{00}, z_{01}, …, z_{i,j}, … z_{|I|-1,|J|-1}
(row-major indexing for z).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDA_DIR = Path("results/eda")
DEFAULT_RESULTS_DIR = Path("results/pmedian")
DEFAULT_DISTANCE_METHOD = "manhattan"
DEFAULT_P = 5

# Demand weight column used from zone flows
DEMAND_WEIGHT_COL = "turnover"  # arrivals + departures; always > 0 for active zones


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PMedianInputs:
    """Validated inputs for the P-Median integer program."""

    demand_zones: pd.DataFrame
    """One row per eligible demand zone.
    Columns: ca_number, ca_name, weight, latitude, longitude."""

    candidate_depots: pd.DataFrame
    """One row per candidate depot location.
    Columns: ca_number, ca_name, latitude, longitude."""

    cost_matrix: pd.DataFrame
    """|I| × |J| DataFrame.  Rows = demand zone ca_number, cols = depot ca_number.
    Values are distances in meters."""

    p: int
    """Number of depots to site."""

    distance_method: str = DEFAULT_DISTANCE_METHOD

    @property
    def n_zones(self) -> int:
        return len(self.demand_zones)

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_depots)

    @property
    def n_vars(self) -> int:
        """Total decision-variable count: |J| y-vars + |I|·|J| z-vars."""
        return self.n_candidates + self.n_zones * self.n_candidates


@dataclass
class ZoneAssignment:
    """Single zone-to-depot assignment."""

    zone_ca: int
    zone_name: str
    depot_ca: int
    depot_name: str
    weight: float
    distance_meters: float
    weighted_distance: float  # weight * distance_meters

    @property
    def distance_km(self) -> float:
        return round(self.distance_meters / 1000.0, 3)

    @property
    def weighted_distance_km(self) -> float:
        return round(self.weighted_distance / 1000.0, 3)


@dataclass
class SelectedDepot:
    """A depot location chosen by the model."""

    depot_ca: int
    depot_name: str
    latitude: float
    longitude: float
    zones_served: int
    total_weight_served: float
    total_distance_served_meters: float
    weighted_distance_served: float


@dataclass
class PMedianResult:
    """Full result of the P-Median solve.

    DISCLAIMER: Selected depot locations are modelled optimal candidates derived
    from Community Area centroids.  They are NOT approved or available depot sites.
    """

    status: str
    """'optimal', 'infeasible', 'iteration_limit', 'numerical_difficulty', or 'error'."""

    solver_message: str

    p: int
    n_zones: int
    n_candidates: int

    objective_value: float
    """Optimal weighted objective: Σ w_i · d_{ij*} · z_{ij*} (meter-weight units)."""

    objective_value_km: float
    """Objective in km-weight units (meters / 1000)."""

    total_unweighted_distance_meters: float
    """Σ d_{ij*} (unweighted, meters)."""

    total_unweighted_distance_km: float
    """Unweighted total distance in km."""

    average_assignment_distance_meters: float
    """Mean assignment distance per zone (unweighted, meters)."""

    selected_depots: list[SelectedDepot]
    """Exactly p selected depot locations."""

    assignments: list[ZoneAssignment]
    """One entry per eligible demand zone."""

    distance_method: str
    solved_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selected_depots"] = [asdict(sd) for sd in self.selected_depots]
        d["assignments"] = [asdict(a) for a in self.assignments]
        return d


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    demand_zones: pd.DataFrame,
    candidate_depots: pd.DataFrame,
    cost_matrix: pd.DataFrame,
    p: int,
) -> None:
    """Raise ValueError with a clear message if any input is invalid."""
    if not isinstance(p, int) or isinstance(p, bool):
        raise TypeError(f"p must be a plain integer; got {type(p).__name__}.")

    n_candidates = len(candidate_depots)

    if p < 1 or p > n_candidates:
        raise ValueError(
            f"p={p} is out of range; must satisfy 1 ≤ p ≤ {n_candidates} "
            f"(number of eligible candidate sites)."
        )

    if demand_zones.empty:
        raise ValueError("demand_zones is empty; no eligible demand zones to assign.")

    if candidate_depots.empty:
        raise ValueError("candidate_depots is empty; no candidate depot locations.")

    required_dz = {"ca_number", "weight"}
    missing_dz = required_dz - set(demand_zones.columns)
    if missing_dz:
        raise ValueError(f"demand_zones is missing required columns: {missing_dz}")

    if (demand_zones["weight"] < 0).any():
        raise ValueError("demand_zones contains negative weight values.")

    if demand_zones["weight"].isna().any():
        raise ValueError("demand_zones contains NaN weight values.")

    if cost_matrix.empty:
        raise ValueError("cost_matrix is empty.")

    if cost_matrix.shape != (len(demand_zones), n_candidates):
        raise ValueError(
            f"cost_matrix shape {cost_matrix.shape} does not match "
            f"({len(demand_zones)}, {n_candidates})."
        )

    if (cost_matrix < 0).any().any():
        raise ValueError("cost_matrix contains negative distance values.")

    if cost_matrix.isna().any().any():
        raise ValueError("cost_matrix contains NaN values.")


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def build_pmedian_inputs(
    zone_flows: pd.DataFrame,
    centroids: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    p: int = DEFAULT_P,
    distance_method: str = DEFAULT_DISTANCE_METHOD,
    candidate_ca_numbers: list[int] | None = None,
) -> PMedianInputs:
    """Build P-Median inputs from Phase 2 EDA outputs.

    Parameters
    ----------
    zone_flows : pd.DataFrame
        Output of ``calculate_zone_flows()`` with columns ca_number, ca_name,
        turnover, latitude, longitude.
    centroids : pd.DataFrame
        Output of ``derive_zone_centroids()`` with ca_number, ca_name,
        latitude, longitude.
    dist_matrix : pd.DataFrame
        Square distance matrix indexed and columned by ca_number (meters).
    p : int
        Number of depots to site.
    distance_method : str
        Label for the distance metric used.
    candidate_ca_numbers : list[int] | None
        Subset of ca_numbers to use as candidate depots.  If None, all zones
        with valid centroids are candidates.

    Returns
    -------
    PMedianInputs
    """
    # --- Build demand zones --------------------------------------------------
    # Drop zones with invalid coordinates or zero turnover
    valid_zones = zone_flows.dropna(subset=["latitude", "longitude"]).copy()
    valid_zones = valid_zones[valid_zones[DEMAND_WEIGHT_COL] > 0].copy()
    valid_zones = valid_zones.reset_index(drop=True)

    demand_zones = valid_zones[
        ["ca_number", "ca_name", DEMAND_WEIGHT_COL, "latitude", "longitude"]
    ].copy()
    demand_zones = demand_zones.rename(columns={DEMAND_WEIGHT_COL: "weight"})
    demand_zones["ca_number"] = demand_zones["ca_number"].astype(int)
    demand_zones = demand_zones.sort_values("ca_number").reset_index(drop=True)

    # --- Build candidate depots ----------------------------------------------
    centroid_valid = centroids.dropna(subset=["latitude", "longitude"]).copy()
    centroid_valid["ca_number"] = centroid_valid["ca_number"].astype(int)

    if candidate_ca_numbers is not None:
        centroid_valid = centroid_valid[
            centroid_valid["ca_number"].isin(candidate_ca_numbers)
        ]

    # Only keep candidates that are also demand zones (have observed centroids)
    eligible_demand_cas = set(demand_zones["ca_number"].tolist())
    centroid_valid = centroid_valid[
        centroid_valid["ca_number"].isin(eligible_demand_cas)
    ]
    candidate_depots = centroid_valid[
        ["ca_number", "ca_name", "latitude", "longitude"]
    ].copy()
    candidate_depots = candidate_depots.sort_values("ca_number").reset_index(drop=True)

    # --- Align distance matrix -----------------------------------------------
    dist_matrix = dist_matrix.copy()
    dist_matrix.index = dist_matrix.index.astype(int)
    dist_matrix.columns = dist_matrix.columns.astype(int)

    zone_cas = demand_zones["ca_number"].tolist()
    depot_cas = candidate_depots["ca_number"].tolist()

    missing_rows = [ca for ca in zone_cas if ca not in dist_matrix.index]
    missing_cols = [ca for ca in depot_cas if ca not in dist_matrix.columns]
    if missing_rows or missing_cols:
        raise KeyError(
            f"Distance matrix missing demand zones: {missing_rows}; "
            f"depot candidates: {missing_cols}"
        )

    cost_matrix = dist_matrix.loc[zone_cas, depot_cas].copy()

    # --- Validate ------------------------------------------------------------
    _validate_inputs(demand_zones, candidate_depots, cost_matrix, p)

    return PMedianInputs(
        demand_zones=demand_zones,
        candidate_depots=candidate_depots,
        cost_matrix=cost_matrix,
        p=p,
        distance_method=distance_method,
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_pmedian(inputs: PMedianInputs) -> PMedianResult:
    """Solve the P-Median IP using scipy.optimize.milp (HiGHS backend).

    Variable layout (row-major):
      Indices 0 … m-1            : y_j  (depot selection)
      Indices m … m + n*m - 1    : z_{ij} flattened as z[i*m + j]

    where n = |I| (demand zones), m = |J| (candidates).

    Parameters
    ----------
    inputs : PMedianInputs

    Returns
    -------
    PMedianResult
    """
    n = inputs.n_zones  # demand zones
    m = inputs.n_candidates  # candidate depots
    p = inputs.p

    w = inputs.demand_zones["weight"].to_numpy(dtype=float)
    C = inputs.cost_matrix.to_numpy(dtype=float)  # shape (n, m)

    # ----- Objective vector --------------------------------------------------
    # y_j variables: 0 cost (depot selection has no cost in objective)
    c_y = np.zeros(m)
    # z_{ij} variables: w_i * d_{ij}
    c_z = (w[:, np.newaxis] * C).flatten()  # shape (n*m,)
    c = np.concatenate([c_y, c_z])

    # ----- Constraints -------------------------------------------------------
    # (C1) Σ_j z_{ij} = 1  for each i  (n equality constraints)
    #   Each z_{ij} for fixed i spans columns  m + i*m .. m + i*m + m - 1
    A_c1_rows = []
    for i in range(n):
        row = np.zeros(m + n * m)
        row[m + i * m : m + i * m + m] = 1.0
        A_c1_rows.append(row)
    A_c1 = np.vstack(A_c1_rows)
    b_c1_lo = np.ones(n)
    b_c1_hi = np.ones(n)

    # (C2) z_{ij} ≤ y_j  for all i, j  →  z_{ij} - y_j ≤ 0  (n*m inequality constraints)
    A_c2_rows = []
    for i in range(n):
        for j in range(m):
            row = np.zeros(m + n * m)
            row[j] = -1.0  # -y_j
            row[m + i * m + j] = 1.0  # +z_{ij}
            A_c2_rows.append(row)
    A_c2 = np.vstack(A_c2_rows)
    b_c2_lo = np.full(n * m, -np.inf)
    b_c2_hi = np.zeros(n * m)

    # (C3) Σ_j y_j = p
    A_c3 = np.zeros((1, m + n * m))
    A_c3[0, :m] = 1.0
    b_c3_lo = np.array([float(p)])
    b_c3_hi = np.array([float(p)])

    A_all = np.vstack([A_c1, A_c2, A_c3])
    b_lo = np.concatenate([b_c1_lo, b_c2_lo, b_c3_lo])
    b_hi = np.concatenate([b_c1_hi, b_c2_hi, b_c3_hi])

    constraints = LinearConstraint(A_all, lb=b_lo, ub=b_hi)

    # ----- Variable bounds and integrality -----------------------------------
    bounds = Bounds(lb=np.zeros(m + n * m), ub=np.ones(m + n * m))
    # All variables are binary
    integrality = np.ones(m + n * m)  # 1 = integer-valued

    # ----- Solve -------------------------------------------------------------
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": 300.0},
    )

    status_map = {
        0: "optimal",
        1: "iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_difficulty",
    }
    status = status_map.get(result.status, "error")

    if status != "optimal":
        return PMedianResult(
            status=status,
            solver_message=result.message,
            p=p,
            n_zones=n,
            n_candidates=m,
            objective_value=float("nan"),
            objective_value_km=float("nan"),
            total_unweighted_distance_meters=float("nan"),
            total_unweighted_distance_km=float("nan"),
            average_assignment_distance_meters=float("nan"),
            selected_depots=[],
            assignments=[],
            distance_method=inputs.distance_method,
        )

    # ----- Extract solution --------------------------------------------------
    x = result.x
    y_vals = x[:m]
    z_vals = x[m:].reshape(n, m)

    # Round to binary
    y_bin = (y_vals > 0.5).astype(int)
    z_bin = (z_vals > 0.5).astype(int)

    # ----- Build selected depots list ----------------------------------------
    depot_df = inputs.candidate_depots
    zone_df = inputs.demand_zones

    selected_j = [j for j in range(m) if y_bin[j] == 1]

    # Pre-compute per-depot stats
    selected_depots: list[SelectedDepot] = []
    for j in selected_j:
        served_i = [i for i in range(n) if z_bin[i, j] == 1]
        total_w = float(sum(w[i] for i in served_i))
        total_d = float(sum(C[i, j] for i in served_i))
        wtd_d = float(sum(w[i] * C[i, j] for i in served_i))
        selected_depots.append(
            SelectedDepot(
                depot_ca=int(depot_df.iloc[j]["ca_number"]),
                depot_name=str(depot_df.iloc[j]["ca_name"]),
                latitude=float(depot_df.iloc[j]["latitude"]),
                longitude=float(depot_df.iloc[j]["longitude"]),
                zones_served=len(served_i),
                total_weight_served=total_w,
                total_distance_served_meters=round(total_d, 1),
                weighted_distance_served=round(wtd_d, 1),
            )
        )
    selected_depots.sort(key=lambda sd: -sd.zones_served)

    # For each zone, find its assigned depot
    assignments: list[ZoneAssignment] = []
    total_unweighted = 0.0
    for i in range(n):
        assigned_j_list = [j for j in range(m) if z_bin[i, j] == 1]
        if not assigned_j_list:
            # Should not happen for a feasible solution; assign to closest as safety
            assigned_j = int(np.argmin(C[i, :]))
        else:
            assigned_j = assigned_j_list[0]

        d_m = float(C[i, assigned_j])
        wi = float(w[i])
        total_unweighted += d_m
        assignments.append(
            ZoneAssignment(
                zone_ca=int(zone_df.iloc[i]["ca_number"]),
                zone_name=str(zone_df.iloc[i]["ca_name"]),
                depot_ca=int(depot_df.iloc[assigned_j]["ca_number"]),
                depot_name=str(depot_df.iloc[assigned_j]["ca_name"]),
                weight=wi,
                distance_meters=round(d_m, 1),
                weighted_distance=round(wi * d_m, 1),
            )
        )

    obj_val = float(sum(a.weighted_distance for a in assignments))
    avg_d = total_unweighted / n if n > 0 else 0.0

    return PMedianResult(
        status=status,
        solver_message=result.message,
        p=p,
        n_zones=n,
        n_candidates=m,
        objective_value=round(obj_val, 2),
        objective_value_km=round(obj_val / 1000.0, 3),
        total_unweighted_distance_meters=round(total_unweighted, 2),
        total_unweighted_distance_km=round(total_unweighted / 1000.0, 3),
        average_assignment_distance_meters=round(avg_d, 2),
        selected_depots=selected_depots,
        assignments=assignments,
        distance_method=inputs.distance_method,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_eda_outputs(
    eda_dir: Path,
    distance_method: str = DEFAULT_DISTANCE_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load zone_flows, centroids, and distance matrix from Phase 2 EDA outputs.

    Returns
    -------
    (zone_flows, centroids, dist_matrix)
    """
    flows_path = eda_dir / "zone_flows_and_imbalance.csv"
    centroids_path = eda_dir / "zone_centroids.csv"
    dist_filename = (
        "zone_distances_manhattan_meters.csv"
        if distance_method == "manhattan"
        else "zone_distances_haversine_meters.csv"
    )
    dist_path = eda_dir / dist_filename

    for path in (flows_path, centroids_path, dist_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required EDA file not found: {path.resolve()}\n"
                "Run Phase 2 EDA first: uv run escooter-eda"
            )

    zone_flows = pd.read_csv(flows_path)
    centroids = pd.read_csv(centroids_path)
    dist_matrix = pd.read_csv(dist_path, index_col=0)
    dist_matrix.index = dist_matrix.index.astype(int)
    dist_matrix.columns = dist_matrix.columns.astype(int)

    return zone_flows, centroids, dist_matrix


def save_results(result: PMedianResult, out_dir: Path) -> None:
    """Save P-Median results to CSV and JSON files.

    Files produced:
      - pmedian_solution.json  — full result dict
      - pmedian_summary.csv    — one-row summary metrics
      - selected_depots.csv    — one row per selected depot
      - zone_assignments.csv   — one row per demand zone
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "pmedian_solution.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    summary = {
        "status": result.status,
        "p": result.p,
        "n_zones": result.n_zones,
        "n_candidates": result.n_candidates,
        "objective_value_km": result.objective_value_km,
        "total_unweighted_distance_km": result.total_unweighted_distance_km,
        "average_assignment_distance_meters": result.average_assignment_distance_meters,
        "distance_method": result.distance_method,
        "solved_at_utc": result.solved_at_utc,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "pmedian_summary.csv", index=False)

    if result.selected_depots:
        depot_rows = [asdict(sd) for sd in result.selected_depots]
        pd.DataFrame(depot_rows).to_csv(out_dir / "selected_depots.csv", index=False)

    if result.assignments:
        assign_rows = []
        for a in result.assignments:
            row = asdict(a)
            row["distance_km"] = a.distance_km
            row["weighted_distance_km"] = a.weighted_distance_km
            assign_rows.append(row)
        pd.DataFrame(assign_rows).to_csv(out_dir / "zone_assignments.csv", index=False)


def format_solution_report(result: PMedianResult) -> str:
    """Format a human-readable solution report."""
    lines = [
        "=" * 70,
        "Chicago E-Scooter Phase 4: P-Median Depot Location Solution",
        "=" * 70,
        "",
        "DISCLAIMER: Selected depot locations are modelled optimal candidates",
        "derived from Community Area centroids.  They are NOT approved or",
        "physically available depot sites.",
        "",
        f"Status:          {result.status}",
        f"Solver message:  {result.solver_message}",
        f"Distance method: {result.distance_method}",
        f"Solved at (UTC): {result.solved_at_utc}",
        "",
        "--- Problem Scale ---",
        f"Demand zones:    {result.n_zones}",
        f"Candidates:      {result.n_candidates}",
        f"Depots (p):      {result.p}",
        "",
        "--- Objective & Metrics ---",
        f"Weighted objective:           {result.objective_value_km:>12,.1f} km-weight",
        f"Total unweighted distance:    {result.total_unweighted_distance_km:>12,.1f} km",
        f"Mean assignment distance:     {result.average_assignment_distance_meters:>12,.1f} m",
        "",
        "--- Selected Depot Locations ---",
        f"{'CA':>4}  {'Name':<26}  {'Zones':>5}  {'Wt-Dist(km)':>12}",
        "-" * 60,
    ]
    for sd in result.selected_depots:
        lines.append(
            f"{sd.depot_ca:>4}  {sd.depot_name:<26}  {sd.zones_served:>5}  "
            f"{sd.weighted_distance_served / 1000:>12,.1f}"
        )

    lines += [
        "",
        "--- Zone Assignments (first 20) ---",
        f"{'Zone':<26}  {'Depot':<26}  {'Dist(km)':>8}",
        "-" * 66,
    ]
    for a in sorted(result.assignments, key=lambda a: a.zone_name)[:20]:
        lines.append(f"{a.zone_name:<26}  {a.depot_name:<26}  {a.distance_km:>8.2f}")
    if len(result.assignments) > 20:
        lines.append(
            f"  ... and {len(result.assignments) - 20} more (see zone_assignments.csv)"
        )

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Phase 4 P-Median depot location."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: P-Median Depot Location for Chicago E-Scooter Rebalancing.\n\n"
            "DISCLAIMER: Selected depot locations are modelled optimal candidates, "
            "NOT approved or physically available depot sites."
        )
    )
    parser.add_argument(
        "--eda-dir",
        type=Path,
        default=DEFAULT_EDA_DIR,
        help=f"Directory containing Phase 2 EDA CSV outputs (default: {DEFAULT_EDA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to write Phase 4 output files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=DEFAULT_P,
        help=f"Number of depots to site (default: {DEFAULT_P}).",
    )
    parser.add_argument(
        "--distance-method",
        choices=["manhattan", "haversine"],
        default=DEFAULT_DISTANCE_METHOD,
        help=f"Distance metric for cost matrix (default: {DEFAULT_DISTANCE_METHOD}).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip generating visualisation figures.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Chicago E-Scooter: Phase 4 P-Median Depot Location")
    print("=" * 70)
    print(
        "\nDISCLAIMER: Selected depot locations are modelled optimal candidates\n"
        "derived from zone centroids.  They are NOT approved depot sites.\n"
    )

    print(f"Loading EDA outputs from: {args.eda_dir.resolve()}")
    zone_flows, centroids, dist_matrix = load_eda_outputs(
        args.eda_dir, args.distance_method
    )
    print(f"  Zone flows:      {len(zone_flows)} zones")
    print(f"  Centroids:       {len(centroids)} zones")
    print(
        f"  Distance matrix: {dist_matrix.shape[0]}x{dist_matrix.shape[1]} ({args.distance_method})"
    )

    print(f"\nBuilding P-Median inputs (p={args.p}) ...")
    inputs = build_pmedian_inputs(
        zone_flows=zone_flows,
        centroids=centroids,
        dist_matrix=dist_matrix,
        p=args.p,
        distance_method=args.distance_method,
    )
    print(f"  Demand zones:   {inputs.n_zones}")
    print(f"  Candidates:     {inputs.n_candidates}")
    print(
        f"  Variables:      {inputs.n_vars:,} (y: {inputs.n_candidates}, z: {inputs.n_zones * inputs.n_candidates})"
    )

    print("\nSolving P-Median IP ...")
    result = solve_pmedian(inputs)

    print()
    print(format_solution_report(result))

    save_results(result, args.results_dir)
    print(f"\nResults saved to: {args.results_dir.resolve()}")

    if not args.no_figures:
        try:
            from escooter_opt.optimization.pmedian_viz import generate_all_figures

            fig_dir = args.results_dir / "figures"
            generate_all_figures(result, inputs, fig_dir)
            print(f"Figures saved to:  {fig_dir.resolve()}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: figure generation failed — {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
