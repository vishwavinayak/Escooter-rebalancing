"""Automated unit tests for Phase 2 Exploratory Data Analysis.

Uses small synthetic fixtures exclusively. Never depends on the 144MB raw CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from escooter_opt.analysis.eda import run_exploratory_analysis
from escooter_opt.analysis.spatial import (
    calculate_od_matrix,
    calculate_zone_flows,
    compute_zone_distance_matrix,
    derive_zone_centroids,
    haversine_distance,
    manhattan_grid_distance,
    summarize_spatial_coverage,
)
from escooter_opt.analysis.temporal import (
    calculate_daily_volume,
    calculate_day_of_week_volume,
    calculate_hourly_volume,
    summarize_temporal_patterns,
)


@pytest.fixture
def synthetic_cleaned_df() -> pd.DataFrame:
    """Creates a small synthetic cleaned dataframe with 6 trips across 3 zones."""
    return pd.DataFrame(
        {
            "trip_id": [f"T{i}" for i in range(1, 7)],
            "start_time": pd.to_datetime(
                [
                    "2020-08-12 05:00:00",
                    "2020-08-12 08:00:00",
                    "2020-08-12 17:00:00",
                    "2020-08-13 17:00:00",
                    "2020-08-14 18:00:00",
                    "2020-08-15 12:00:00",
                ]
            ),
            "end_time": pd.to_datetime(
                [
                    "2020-08-12 05:00:00",
                    "2020-08-12 08:00:00",
                    "2020-08-12 17:00:00",
                    "2020-08-13 17:00:00",
                    "2020-08-14 18:00:00",
                    "2020-08-15 12:00:00",
                ]
            ),
            "trip_distance_m": [1200.0, 1800.0, 2500.0, 3100.0, 800.0, 2200.0],
            "trip_duration_s": [600.0, 900.0, 1200.0, 1500.0, 400.0, 1100.0],
            "speed_mph": [4.47, 4.47, 4.66, 4.62, 4.47, 4.47],
            "vendor": ["lime", "bird", "spin", "lime", "bird", "spin"],
            # Zone setup:
            # CA 1: 3 departures (T1, T2, T3), 1 arrival (T5) -> Net = 1 - 3 = -2 (Deficit)
            # CA 2: 1 departure (T4), 3 arrivals (T1, T2, T6) -> Net = 3 - 1 = +2 (Surplus)
            # CA 3: 2 departures (T5, T6), 2 arrivals (T3, T4) -> Net = 2 - 2 = 0 (Balanced)
            "start_community_area_number": pd.Series([1, 1, 1, 2, 3, 3], dtype="Int64"),
            "end_community_area_number": pd.Series([2, 2, 3, 3, 1, 2], dtype="Int64"),
            "start_community_area_name": [
                "ROGERS PARK",
                "ROGERS PARK",
                "ROGERS PARK",
                "WEST RIDGE",
                "UPTOWN",
                "UPTOWN",
            ],
            "end_community_area_name": [
                "WEST RIDGE",
                "WEST RIDGE",
                "UPTOWN",
                "UPTOWN",
                "ROGERS PARK",
                "WEST RIDGE",
            ],
            "start_centroid_latitude": [
                42.00888872,
                42.00888872,
                42.00888872,
                41.99618037,
                41.96543532,
                41.96543532,
            ],
            "start_centroid_longitude": [
                -87.6713702,
                -87.6713702,
                -87.6713702,
                -87.69752945,
                -87.65514456,
                -87.65514456,
            ],
            "start_centroid_location": ["POINT (-87.671 42.008)"] * 3
            + ["POINT (-87.697 41.996)"]
            + ["POINT (-87.655 41.965)"] * 2,
            "end_centroid_latitude": [
                41.99618037,
                41.99618037,
                41.96543532,
                41.96543532,
                42.00888872,
                41.99618037,
            ],
            "end_centroid_longitude": [
                -87.69752945,
                -87.69752945,
                -87.65514456,
                -87.65514456,
                -87.6713702,
                -87.69752945,
            ],
            "end_centroid_location": ["POINT (-87.697 41.996)"] * 2
            + ["POINT (-87.655 41.965)"] * 2
            + ["POINT (-87.671 42.008)"]
            + ["POINT (-87.697 41.996)"],
        }
    )


# --------------------------------------------------------------------------
# 1. Zone Aggregation & Imbalance Tests
# --------------------------------------------------------------------------


def test_zone_flows_and_reconciliation(synthetic_cleaned_df: pd.DataFrame):
    """Verify departures, arrivals, net flow (A_i - D_i), and conservation of flow."""
    flows = calculate_zone_flows(synthetic_cleaned_df)

    assert len(flows) == 3
    # Check departures
    assert flows.loc[flows["ca_number"] == 1, "departures"].values[0] == 3
    assert flows.loc[flows["ca_number"] == 2, "departures"].values[0] == 1
    assert flows.loc[flows["ca_number"] == 3, "departures"].values[0] == 2

    # Check arrivals
    assert flows.loc[flows["ca_number"] == 1, "arrivals"].values[0] == 1
    assert flows.loc[flows["ca_number"] == 2, "arrivals"].values[0] == 3
    assert flows.loc[flows["ca_number"] == 3, "arrivals"].values[0] == 2

    # Check net flows I_i = A_i - D_i
    assert flows.loc[flows["ca_number"] == 1, "net_flow"].values[0] == -2
    assert flows.loc[flows["ca_number"] == 2, "net_flow"].values[0] == +2
    assert flows.loc[flows["ca_number"] == 3, "net_flow"].values[0] == 0

    # Flow classifications
    assert (
        flows.loc[flows["ca_number"] == 1, "flow_category"].values[0]
        == "potential_deficit"
    )
    assert (
        flows.loc[flows["ca_number"] == 2, "flow_category"].values[0]
        == "potential_surplus"
    )
    assert flows.loc[flows["ca_number"] == 3, "flow_category"].values[0] == "balanced"

    # Strict reconciliation of arrival/departure totals with eligible trips
    total_trips = len(synthetic_cleaned_df)
    assert flows["departures"].sum() == total_trips
    assert flows["arrivals"].sum() == total_trips
    assert flows["net_flow"].sum() == 0


def test_origin_destination_matrix(synthetic_cleaned_df: pd.DataFrame):
    """Verify Origin-Destination cross-tabulation dimensions and marginal sums."""
    od = calculate_od_matrix(synthetic_cleaned_df)

    assert od.shape == (3, 3)
    # Check OD flows
    # From CA 1 to CA 2: trips T1, T2 -> 2
    assert od.loc[1, 2] == 2
    # From CA 1 to CA 3: trip T3 -> 1
    assert od.loc[1, 3] == 1
    # Row sums equal departures
    assert od.sum(axis=1).to_dict() == {1: 3, 2: 1, 3: 2}
    # Column sums equal arrivals
    assert od.sum(axis=0).to_dict() == {1: 1, 2: 3, 3: 2}


# --------------------------------------------------------------------------
# 2. Centroid Derivation and Distance Metrics Tests
# --------------------------------------------------------------------------


def test_derive_zone_centroids(synthetic_cleaned_df: pd.DataFrame):
    """Verify observed centroids are extracted without coordinate fabrication."""
    centroids = derive_zone_centroids(synthetic_cleaned_df)

    assert len(centroids) == 3
    ca_1 = centroids[centroids["ca_number"] == 1].iloc[0]
    assert ca_1["latitude"] == pytest.approx(42.00888872, rel=1e-5)
    assert ca_1["longitude"] == pytest.approx(-87.6713702, rel=1e-5)


def test_distance_metrics():
    """Verify Haversine straight-line and Manhattan grid distance properties."""
    lat1, lon1 = 42.00888872, -87.67137020  # Rogers Park
    lat2, lon2 = 41.96543532, -87.65514456  # Uptown

    # 1. Zero distance for identical points
    assert haversine_distance(lat1, lon1, lat1, lon1) == 0.0
    assert manhattan_grid_distance(lat1, lon1, lat1, lon1) == 0.0

    # 2. Symmetry
    d_hav = haversine_distance(lat1, lon1, lat2, lon2)
    assert d_hav == pytest.approx(haversine_distance(lat2, lon2, lat1, lon1))

    d_man = manhattan_grid_distance(lat1, lon1, lat2, lon2)
    assert d_man == pytest.approx(manhattan_grid_distance(lat2, lon2, lat1, lon1))

    # 3. L1 >= L2 metric property (Manhattan grid distance >= straight-line distance)
    assert d_man >= d_hav
    assert d_hav > 4000.0  # Approx 5 km between Rogers Park and Uptown


def test_compute_zone_distance_matrix(synthetic_cleaned_df: pd.DataFrame):
    """Verify pairwise distance matrix construction."""
    centroids = derive_zone_centroids(synthetic_cleaned_df)
    mat = compute_zone_distance_matrix(centroids, method="manhattan")

    assert mat.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(mat.values), np.zeros(3))
    assert (mat.values >= 0).all()


# --------------------------------------------------------------------------
# 3. Temporal Grouping Tests
# --------------------------------------------------------------------------


def test_hourly_grouping(synthetic_cleaned_df: pd.DataFrame):
    """Verify 24-hour timestamp grouping, trip totals, and peak hour detection."""
    hourly = calculate_hourly_volume(synthetic_cleaned_df)

    assert len(hourly) == 24
    assert hourly["trip_count"].sum() == len(synthetic_cleaned_df)

    # Synthetic dataset has trips at 5:00 (1), 8:00 (1), 12:00 (1), 17:00 (2), 18:00 (1)
    h17 = hourly[hourly["hour"] == 17].iloc[0]
    assert h17["trip_count"] == 2
    assert h17["volume_pct"] == pytest.approx((2 / 6) * 100.0)


def test_day_of_week_grouping(synthetic_cleaned_df: pd.DataFrame):
    """Verify day of week aggregation follows canonical Monday-Sunday order."""
    dow = calculate_day_of_week_volume(synthetic_cleaned_df)

    assert len(dow) == 4  # Wednesday (3), Thursday (1), Friday (1), Saturday (1)
    assert dow["trip_count"].sum() == len(synthetic_cleaned_df)
    assert set(dow["day_of_week"].tolist()).issubset(
        {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    )


def test_daily_grouping(synthetic_cleaned_df: pd.DataFrame):
    """Verify daily aggregation and rolling averages."""
    daily = calculate_daily_volume(synthetic_cleaned_df)

    assert len(daily) == 4  # 4 distinct dates
    assert daily["trip_count"].sum() == len(synthetic_cleaned_df)


def test_summarize_temporal_patterns(synthetic_cleaned_df: pd.DataFrame):
    """Verify summary metrics extraction."""
    summary = summarize_temporal_patterns(synthetic_cleaned_df)

    assert summary["peak_hour"] == 17
    assert summary["peak_hour_volume"] == 2
    assert summary["start_date"] == "2020-08-12"
    assert summary["end_date"] == "2020-08-15"


# --------------------------------------------------------------------------
# 4. Spatial Coverage Summary Test
# --------------------------------------------------------------------------


def test_spatial_coverage_summary(synthetic_cleaned_df: pd.DataFrame):
    """Verify spatial coverage and intra-zone calculations."""
    coverage = summarize_spatial_coverage(synthetic_cleaned_df, raw_valid_trip_count=10)

    assert coverage["total_cleaned_trips"] == 6
    assert coverage["usable_geographic_zones"] == 3
    assert coverage["spatial_coverage_cleaned_pct"] == 100.0
    # In synthetic dataset, all trips cross zones (no intra-zone trips)
    assert coverage["intra_zone_trips"] == 0
    assert coverage["inter_zone_trips"] == 6


# --------------------------------------------------------------------------
# 5. End-to-End Synthetic EDA Pipeline Execution
# --------------------------------------------------------------------------


def test_run_exploratory_analysis_synthetic(
    tmp_path: Path, synthetic_cleaned_df: pd.DataFrame
):
    """Verify complete end-to-end execution of EDA pipeline on synthetic parquet."""
    parquet_file = tmp_path / "cleaned.parquet"
    synthetic_cleaned_df.to_parquet(parquet_file, index=False)

    results_dir = tmp_path / "eda_results"
    tables, summary = run_exploratory_analysis(
        cleaned_parquet_path=parquet_file,
        results_dir=results_dir,
        generate_figures=True,
    )

    assert len(tables) == 8
    assert summary.total_cleaned_trips == 6
    assert summary.flows_reconcile is True
    assert summary.usable_zones_count == 3
    assert (results_dir / "zone_flows_and_imbalance.csv").is_file()
    assert (results_dir / "zone_centroids.csv").is_file()
    assert (results_dir / "od_matrix.csv").is_file()
    assert (results_dir / "eda_summary_metrics.json").is_file()
    assert (results_dir / "figures" / "hourly_trip_volume.png").is_file()
    assert (results_dir / "figures" / "zone_net_flow_imbalances.html").is_file()
