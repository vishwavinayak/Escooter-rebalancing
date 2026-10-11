"""Exploratory Data Analysis orchestrator and CLI for cleaned Chicago E-Scooter Trips."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from escooter_opt.analysis.spatial import (
    calculate_od_matrix,
    calculate_zone_flows,
    compute_zone_distance_matrix,
    derive_zone_centroids,
    summarize_spatial_coverage,
)
from escooter_opt.analysis.temporal import (
    calculate_daily_volume,
    calculate_day_of_week_volume,
    calculate_hourly_volume,
    summarize_temporal_patterns,
)
from escooter_opt.analysis.visualize import (
    plot_daily_time_series,
    plot_day_of_week_volume,
    plot_hourly_volume,
    plot_top_activity_zones,
    plot_zone_imbalances,
)
from escooter_opt.data.clean import DEFAULT_OUTPUT_DIR, DEFAULT_PARQUET_FILENAME

DEFAULT_CLEANED_PARQUET_PATH = DEFAULT_OUTPUT_DIR / DEFAULT_PARQUET_FILENAME
DEFAULT_RESULTS_DIR = Path("results/eda")


@dataclass
class EDASummary:
    """High-level summary of exploratory data analysis findings."""

    total_cleaned_trips: int
    date_range_start: str
    date_range_end: str
    total_calendar_days: int
    mean_daily_trips: float

    # Usable geographic records
    usable_geographic_records: int
    geographic_coverage_pct: float
    total_chicago_zones: int
    usable_zones_count: int

    # Flow reconciliation
    total_departures_sum: int
    total_arrivals_sum: int
    net_flow_sum: int
    flows_reconcile: bool

    # Zone imbalance breakdown
    surplus_zones_count: int
    deficit_zones_count: int
    balanced_zones_count: int

    # Temporal peaks
    peak_hour: int
    peak_hour_volume: int
    peak_hour_pct: float
    peak_day_of_week: str
    weekend_pct: float
    weekday_pct: float

    # Distance summary
    intra_zone_pct: float
    inter_zone_pct: float
    distance_method: str = "manhattan_street_grid"


def run_exploratory_analysis(
    cleaned_parquet_path: Path | None = None,
    results_dir: Path | None = None,
    generate_figures: bool = True,
) -> tuple[dict[str, pd.DataFrame], EDASummary]:
    """Execute complete Phase 2 exploratory data analysis.

    Reuses cleaned dataset from Phase 1 without repeating cleaning.
    Produces tabular summary CSVs and visualizations.
    """
    parquet_file = cleaned_parquet_path or DEFAULT_CLEANED_PARQUET_PATH
    out_dir = results_dir or DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    if not parquet_file.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {parquet_file.resolve()}.\n"
            f"Please run the cleaning pipeline first via `uv run escooter-clean`."
        )

    df = pd.read_parquet(parquet_file)

    # 1. Spatial Analysis
    zone_flows = calculate_zone_flows(df)
    centroids = derive_zone_centroids(df)
    od_matrix = calculate_od_matrix(df)
    dist_manhattan = compute_zone_distance_matrix(centroids, method="manhattan")
    dist_haversine = compute_zone_distance_matrix(centroids, method="haversine")
    spatial_summary = summarize_spatial_coverage(df)

    # 2. Temporal Analysis
    hourly = calculate_hourly_volume(df)
    dow = calculate_day_of_week_volume(df)
    daily = calculate_daily_volume(df)
    temporal_summary = summarize_temporal_patterns(df)

    # 3. Flow Reconciliation Checks
    total_dep = int(zone_flows["departures"].sum())
    total_arr = int(zone_flows["arrivals"].sum())
    net_sum = int(zone_flows["net_flow"].sum())
    flows_reconcile = (
        (total_dep == len(df)) and (total_arr == len(df)) and (net_sum == 0)
    )

    surplus_cnt = int((zone_flows["net_flow"] > 0).sum())
    deficit_cnt = int((zone_flows["net_flow"] < 0).sum())
    balanced_cnt = int((zone_flows["net_flow"] == 0).sum())

    eda_summary = EDASummary(
        total_cleaned_trips=len(df),
        date_range_start=temporal_summary["start_date"],
        date_range_end=temporal_summary["end_date"],
        total_calendar_days=temporal_summary["total_calendar_days"],
        mean_daily_trips=temporal_summary["mean_daily_trips"],
        usable_geographic_records=spatial_summary["usable_spatial_records"],
        geographic_coverage_pct=spatial_summary["spatial_coverage_cleaned_pct"],
        total_chicago_zones=spatial_summary["total_chicago_zones"],
        usable_zones_count=spatial_summary["usable_geographic_zones"],
        total_departures_sum=total_dep,
        total_arrivals_sum=total_arr,
        net_flow_sum=net_sum,
        flows_reconcile=flows_reconcile,
        surplus_zones_count=surplus_cnt,
        deficit_zones_count=deficit_cnt,
        balanced_zones_count=balanced_cnt,
        peak_hour=temporal_summary["peak_hour"],
        peak_hour_volume=temporal_summary["peak_hour_volume"],
        peak_hour_pct=temporal_summary["peak_hour_pct"],
        peak_day_of_week=temporal_summary["peak_day_of_week"],
        weekend_pct=temporal_summary["weekend_pct"],
        weekday_pct=temporal_summary["weekday_pct"],
        intra_zone_pct=spatial_summary["intra_zone_pct"],
        inter_zone_pct=spatial_summary["inter_zone_pct"],
    )

    # 4. Save CSV Tables
    zone_flows.to_csv(out_dir / "zone_flows_and_imbalance.csv", index=False)
    centroids.to_csv(out_dir / "zone_centroids.csv", index=False)
    od_matrix.to_csv(out_dir / "od_matrix.csv")
    dist_manhattan.to_csv(out_dir / "zone_distances_manhattan_meters.csv")
    dist_haversine.to_csv(out_dir / "zone_distances_haversine_meters.csv")
    hourly.to_csv(out_dir / "temporal_hourly.csv", index=False)
    dow.to_csv(out_dir / "temporal_day_of_week.csv", index=False)
    daily.to_csv(out_dir / "temporal_daily.csv", index=False)

    summary_dict: dict[str, Any] = asdict(eda_summary)
    with open(out_dir / "eda_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    pd.DataFrame([summary_dict]).to_csv(
        out_dir / "eda_summary_metrics.csv", index=False
    )

    # 5. Generate Visualizations
    if generate_figures:
        plot_hourly_volume(hourly, fig_dir)
        plot_day_of_week_volume(dow, fig_dir)
        plot_daily_time_series(daily, fig_dir)
        plot_zone_imbalances(zone_flows, fig_dir)
        plot_top_activity_zones(zone_flows, fig_dir)

    tables = {
        "zone_flows": zone_flows,
        "centroids": centroids,
        "od_matrix": od_matrix,
        "dist_manhattan": dist_manhattan,
        "dist_haversine": dist_haversine,
        "hourly": hourly,
        "dow": dow,
        "daily": daily,
    }

    return tables, eda_summary


def main() -> None:
    """CLI entry point for exploratory data analysis."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 exploratory data analysis on cleaned Chicago E-Scooter dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CLEANED_PARQUET_PATH,
        help="Path to cleaned parquet file (default: data/processed/escooter_trips_2020_cleaned.parquet).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save CSV summary tables and figures (default: results/eda).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure rendering (faster execution).",
    )

    args = parser.parse_args()

    print("================================================================")
    print("Chicago E-Scooter Trips: Phase 2 Exploratory Data Analysis")
    print("================================================================")
    _, summary = run_exploratory_analysis(
        cleaned_parquet_path=args.input,
        results_dir=args.results_dir,
        generate_figures=not args.no_figures,
    )

    print(f"Cleaned Dataset: {args.input}")
    print(f"Total Trips: {summary.total_cleaned_trips:,}")
    print(
        f"Date Range: {summary.date_range_start} to {summary.date_range_end} ({summary.total_calendar_days} days)"
    )
    print(f"Mean Daily Trips: {summary.mean_daily_trips:,.1f}")
    print("\n--- Geographic Usability & Coverage ---")
    print(
        f"Usable Geographic Records: {summary.usable_geographic_records:,} ({summary.geographic_coverage_pct:.1f}%)"
    )
    print(
        f"Usable Community Area Zones: {summary.usable_zones_count} / {summary.total_chicago_zones}"
    )
    print(
        f"Intra-Zone Trips (Same CA): {summary.intra_zone_pct:.1f}% | Inter-Zone Trips: {summary.inter_zone_pct:.1f}%"
    )
    print("\n--- Flow Balance Reconciliation ---")
    print(f"Sum of Departures: {summary.total_departures_sum:,}")
    print(f"Sum of Arrivals:   {summary.total_arrivals_sum:,}")
    print(
        f"Net Flow Sum (A - D): {summary.net_flow_sum:,} (Reconciled: {summary.flows_reconcile})"
    )
    print(f"Potential Surplus Zones (Net > 0): {summary.surplus_zones_count}")
    print(f"Potential Deficit Zones (Net < 0): {summary.deficit_zones_count}")
    print("\n--- Temporal Dynamics ---")
    print(
        f"Peak Hour: {summary.peak_hour}:00 ({summary.peak_hour_volume:,} trips, {summary.peak_hour_pct:.1f}%)"
    )
    print(f"Peak Day of Week: {summary.peak_day_of_week}")
    print(
        f"Weekday vs Weekend: {summary.weekday_pct:.1f}% vs {summary.weekend_pct:.1f}%"
    )
    print(f"\nResults and figures saved to: {args.results_dir.resolve()}")
    print("================================================================")


if __name__ == "__main__":
    main()
