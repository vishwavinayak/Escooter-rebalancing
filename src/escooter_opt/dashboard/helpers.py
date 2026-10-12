"""Reusable helper and business-logic functions for the Streamlit dashboard.

Phase 6 – Streamlit Dashboard Helpers
======================================
This module houses UI-agnostic data loading, input validation, and execution
wrappers for the cleaning pipeline, EDA summaries, Transportation Problem,
and P-Median Facility Location models. Keeping this logic separate ensures
the Streamlit pages remain thin presentation layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from escooter_opt.data.clean import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARQUET_FILENAME,
    DEFAULT_REPORT_FILENAME,
    CleaningConfig,
    clean_dataframe,
)
from escooter_opt.optimization.pmedian import (
    PMedianInputs,
    PMedianResult,
    build_pmedian_inputs,
    solve_pmedian,
)
from escooter_opt.optimization.transport import (
    TransportProblemInputs,
    TransportProblemResult,
    build_transport_inputs,
    solve_transport_problem,
)
from escooter_opt.scenarios.runner import (
    ScenarioConfig,
    ScenarioResult,
    build_pmedian_suite,
    build_transport_suite,
    run_scenario_suite,
)

DEFAULT_EDA_DIR = Path("results/eda")
DEFAULT_CLEANED_PARQUET = DEFAULT_OUTPUT_DIR / DEFAULT_PARQUET_FILENAME
DEFAULT_CLEANING_REPORT = DEFAULT_OUTPUT_DIR / DEFAULT_REPORT_FILENAME


# ---------------------------------------------------------------------------
# Data & EDA Loading Helpers
# ---------------------------------------------------------------------------


def load_cleaned_data(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned trip parquet dataset.

    Parameters
    ----------
    path : Path | None
        Custom parquet path. Defaults to data/processed/escooter_trips_2020_cleaned.parquet.

    Returns
    -------
    pd.DataFrame
    """
    target = path or DEFAULT_CLEANED_PARQUET
    if not target.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {target.resolve()}. "
            "Please run the cleaning pipeline or upload a dataset."
        )
    return pd.read_parquet(target)


def load_cleaning_report(path: Path | None = None) -> dict[str, Any]:
    """Load the machine-readable cleaning audit report JSON.

    Parameters
    ----------
    path : Path | None
        Custom report JSON path. Defaults to data/processed/cleaning_report.json.

    Returns
    -------
    dict[str, Any]
    """
    target = path or DEFAULT_CLEANING_REPORT
    if not target.is_file():
        return {}
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def load_eda_tables(eda_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load precomputed EDA tables from results/eda.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'zone_flows', 'centroids', 'od_matrix', 'hourly', 'dow', 'daily',
        'dist_manhattan', 'dist_haversine'. Missing files omitted.
    """
    base = eda_dir or DEFAULT_EDA_DIR
    tables: dict[str, pd.DataFrame] = {}
    mapping = {
        "zone_flows": base / "zone_flows_and_imbalance.csv",
        "centroids": base / "zone_centroids.csv",
        "od_matrix": base / "od_matrix.csv",
        "hourly": base / "temporal_hourly.csv",
        "dow": base / "temporal_day_of_week.csv",
        "daily": base / "temporal_daily.csv",
        "dist_manhattan": base / "zone_distances_manhattan_meters.csv",
        "dist_haversine": base / "zone_distances_haversine_meters.csv",
    }
    for key, path in mapping.items():
        if path.is_file():
            if "matrix" in key or "dist" in key:
                # Read with index column
                tables[key] = pd.read_csv(path, index_col=0)
            else:
                tables[key] = pd.read_csv(path)
    return tables


def load_eda_summary_metrics(eda_dir: Path | None = None) -> dict[str, Any]:
    """Load high-level EDA summary metrics from JSON or CSV."""
    base = eda_dir or DEFAULT_EDA_DIR
    json_path = base / "eda_summary_metrics.json"
    if json_path.is_file():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_cleaning_from_dataframe(
    raw_df: pd.DataFrame,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a supplied raw DataFrame using Policy C rules.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Unprocessed trips DataFrame.
    config : CleaningConfig | None
        Optional custom threshold config.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Cleaned DataFrame and report dictionary.
    """
    cleaned_df, report = clean_dataframe(raw_df, config=config)
    return cleaned_df, report.to_dict()


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_transport_params(
    distance_method: str,
    demand_scale: float,
) -> list[str]:
    """Validate user inputs for the Transportation Problem.

    Returns
    -------
    list[str]
        List of actionable error messages, empty if valid.
    """
    errors: list[str] = []
    if distance_method not in ("manhattan", "haversine"):
        errors.append(
            f"Invalid distance metric '{distance_method}'. Must be 'manhattan' or 'haversine'."
        )
    if demand_scale <= 0.0:
        errors.append(
            f"Demand scale factor must be strictly positive (> 0), got {demand_scale}."
        )
    if demand_scale > 10.0:
        errors.append(
            f"Demand scale factor is unusually large (> 10.0), got {demand_scale}."
        )
    return errors


def validate_pmedian_params(
    p: int,
    max_p: int,
    distance_method: str,
    candidate_cas: list[int] | None = None,
) -> list[str]:
    """Validate user inputs for the P-Median Facility Location problem.

    Returns
    -------
    list[str]
        List of actionable error messages, empty if valid.
    """
    errors: list[str] = []
    if p < 1:
        errors.append(f"Number of depots (p) must be at least 1, got {p}.")
    if max_p > 0 and p > max_p:
        errors.append(
            f"Number of depots (p={p}) cannot exceed available candidates ({max_p})."
        )
    if distance_method not in ("manhattan", "haversine"):
        errors.append(
            f"Invalid distance metric '{distance_method}'. Must be 'manhattan' or 'haversine'."
        )
    if candidate_cas is not None:
        if len(candidate_cas) == 0:
            errors.append("Selected candidate depots list cannot be empty.")
        elif p > len(candidate_cas):
            errors.append(
                f"Requested p={p} exceeds chosen candidates ({len(candidate_cas)})."
            )
    return errors


def validate_scenario_params(
    p_values: list[int],
    max_p: int,
    distance_methods: list[str],
    demand_scales: list[float] | None = None,
) -> list[str]:
    """Validate user inputs for the Scenario Analysis suite.

    Returns
    -------
    list[str]
        List of actionable error messages, empty if valid.
    """
    errors: list[str] = []
    if not p_values:
        errors.append("At least one depot count (p) must be selected.")
    else:
        for p in p_values:
            if p < 1:
                errors.append(f"Depot count must be positive integer, got {p}.")
            elif max_p > 0 and p > max_p:
                errors.append(
                    f"Depot count p={p} exceeds max candidate sites ({max_p})."
                )

    if not distance_methods:
        errors.append(
            "At least one distance method ('manhattan', 'haversine') must be selected."
        )
    for dm in distance_methods:
        if dm not in ("manhattan", "haversine"):
            errors.append(f"Invalid distance method '{dm}'.")

    if demand_scales is not None:
        if not demand_scales:
            errors.append("At least one demand scale factor must be provided.")
        for s in demand_scales:
            if s <= 0.0:
                errors.append(f"Demand scale must be strictly positive, got {s}.")
    return errors


# ---------------------------------------------------------------------------
# Solver Execution Wrappers
# ---------------------------------------------------------------------------


def execute_transport_solver(
    zone_flows: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    distance_method: str = "manhattan",
    demand_scale: float = 1.0,
) -> tuple[TransportProblemResult, pd.DataFrame, TransportProblemInputs]:
    """Execute the Transportation Problem with optional demand scaling.

    Parameters
    ----------
    zone_flows : pd.DataFrame
        Zone flow data containing net_flow, arrivals, departures.
    dist_matrix : pd.DataFrame
        Pairwise distance matrix between zone centroids in meters.
    distance_method : str
        'manhattan' or 'haversine'.
    demand_scale : float
        Uniform scale factor for flow imbalances (default 1.0).

    Returns
    -------
    tuple[TransportProblemResult, pd.DataFrame, TransportProblemInputs]
        Result object, detailed allocations DataFrame, and inputs object.
    """
    flows = zone_flows.copy()
    if demand_scale != 1.0:
        flows["net_flow"] = (flows["net_flow"] * demand_scale).round().astype(int)
        flows["arrivals"] = (flows["arrivals"] * demand_scale).round().astype(int)
        flows["departures"] = (flows["departures"] * demand_scale).round().astype(int)

    inputs = build_transport_inputs(
        zone_flows=flows,
        dist_matrix=dist_matrix,
        distance_method=distance_method,
    )
    result = solve_transport_problem(inputs)

    # Convert allocations to DataFrame
    alloc_rows = [
        {
            "Surplus CA": a.surplus_ca,
            "Surplus Name": a.surplus_name,
            "Deficit CA": a.deficit_ca,
            "Deficit Name": a.deficit_name,
            "Units Transferred": a.units,
            "Distance (m)": round(a.distance_meters, 1),
            "Distance (km)": a.distance_km,
            "Total Cost (km-units)": a.total_cost_km,
        }
        for a in result.allocations
    ]
    alloc_df = pd.DataFrame(alloc_rows)
    return result, alloc_df, inputs


def execute_pmedian_solver(
    zone_flows: pd.DataFrame,
    centroids: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    p: int = 5,
    distance_method: str = "manhattan",
    candidate_cas: list[int] | None = None,
) -> tuple[PMedianResult, pd.DataFrame, pd.DataFrame, PMedianInputs]:
    """Execute the P-Median Integer Program solver.

    Parameters
    ----------
    zone_flows : pd.DataFrame
        Zone flow summaries with arrivals and departures.
    centroids : pd.DataFrame
        Zone geographic centroids.
    dist_matrix : pd.DataFrame
        Pairwise distance matrix in meters.
    p : int
        Number of depots to site.
    distance_method : str
        'manhattan' or 'haversine'.
    candidate_cas : list[int] | None
        Optional subset of CA numbers eligible as candidate depots.

    Returns
    -------
    tuple[PMedianResult, pd.DataFrame, pd.DataFrame, PMedianInputs]
        Result, Selected Depots DataFrame, Zone Assignments DataFrame, Inputs.
    """
    flows = zone_flows.copy()
    if "latitude" not in flows.columns or "longitude" not in flows.columns:
        coords = centroids[["ca_number", "latitude", "longitude"]].drop_duplicates(
            subset=["ca_number"]
        )
        flows = flows.merge(coords, on="ca_number", how="left")

    inputs = build_pmedian_inputs(
        zone_flows=flows,
        centroids=centroids,
        dist_matrix=dist_matrix,
        p=p,
        distance_method=distance_method,
        candidate_ca_numbers=candidate_cas,
    )
    result = solve_pmedian(inputs)

    # Depots DataFrame
    depot_rows = [
        {
            "Depot CA": d.depot_ca,
            "Depot Name": d.depot_name,
            "Latitude": d.latitude,
            "Longitude": d.longitude,
            "Assigned Demand Zones": d.zones_served,
            "Assigned Total Weight": d.total_weight_served,
            "Total Weighted Distance (km)": round(
                d.weighted_distance_served / 1000.0, 3
            ),
            "Mean Zone Distance (m)": round(
                d.total_distance_served_meters / max(d.zones_served, 1), 1
            ),
        }
        for d in result.selected_depots
    ]
    depots_df = pd.DataFrame(depot_rows)

    # Assignments DataFrame
    assign_rows = [
        {
            "Zone CA": a.zone_ca,
            "Zone Name": a.zone_name,
            "Assigned Depot CA": a.depot_ca,
            "Assigned Depot Name": a.depot_name,
            "Demand Weight": a.weight,
            "Distance (m)": round(a.distance_meters, 1),
            "Distance (km)": a.distance_km,
            "Weighted Distance (km)": a.weighted_distance_km,
        }
        for a in result.assignments
    ]
    assign_df = pd.DataFrame(assign_rows)
    return result, depots_df, assign_df, inputs


def execute_scenario_suite(
    p_values: list[int],
    distance_methods: list[str],
    include_transport: bool = True,
    demand_scales: list[float] | None = None,
    eda_dir: Path | None = None,
) -> list[ScenarioResult]:
    """Execute a configurable suite of P-Median and Transport scenarios."""
    base_eda = eda_dir or DEFAULT_EDA_DIR
    configs: list[ScenarioConfig] = []

    # Build P-Median configs
    for dm in distance_methods:
        configs.extend(build_pmedian_suite(sorted(p_values), distance_method=dm))

    # Build Transport configs
    if include_transport:
        scales = demand_scales or [1.0, 0.5]
        custom_transport = []
        for dm in distance_methods:
            for s in scales:
                lbl = f"{'baseline' if s == 1.0 else 'offpeak'}_{dm}"
                custom_transport.append(
                    {
                        "label": lbl,
                        "distance_method": dm,
                        "demand_scale": s,
                    }
                )
        configs.extend(build_transport_suite(custom_transport))

    return run_scenario_suite(configs, eda_dir=base_eda)
