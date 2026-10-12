"""Unit tests for Phase 6: Streamlit Dashboard helpers, validation, and visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from escooter_opt.dashboard.helpers import (
    execute_pmedian_solver,
    execute_scenario_suite,
    execute_transport_solver,
    load_cleaning_report,
    load_eda_summary_metrics,
    load_eda_tables,
    validate_pmedian_params,
    validate_scenario_params,
    validate_transport_params,
)
from escooter_opt.dashboard.plotting import (
    create_daily_trend_fig,
    create_day_of_week_fig,
    create_hourly_profile_fig,
    create_net_flow_imbalance_fig,
    create_pmedian_dist_hist_fig,
    create_pmedian_load_fig,
    create_pmedian_map_fig,
    create_scenario_avg_dist_fig,
    create_scenario_elbow_fig,
    create_scenario_transport_obj_fig,
    create_top_zones_fig,
    create_transport_allocation_fig,
    create_transport_dist_hist_fig,
)
from escooter_opt.scenarios.runner import ScenarioConfig, ScenarioResult

# ---------------------------------------------------------------------------
# Fixtures for Testing
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_zone_flows() -> pd.DataFrame:
    """Small 3-zone flow DataFrame for integration testing."""
    return pd.DataFrame(
        [
            {
                "ca_number": 1,
                "ca_name": "Rogers Park",
                "departures": 100,
                "arrivals": 150,
                "net_flow": 50,
                "turnover": 250,
                "flow_category": "potential_surplus",
                "latitude": 42.01,
                "longitude": -87.67,
            },
            {
                "ca_number": 2,
                "ca_name": "West Ridge",
                "departures": 120,
                "arrivals": 70,
                "net_flow": -50,
                "turnover": 190,
                "flow_category": "potential_deficit",
                "latitude": 42.00,
                "longitude": -87.69,
            },
            {
                "ca_number": 3,
                "ca_name": "Uptown",
                "departures": 80,
                "arrivals": 80,
                "net_flow": 0,
                "turnover": 160,
                "flow_category": "balanced",
                "latitude": 41.97,
                "longitude": -87.66,
            },
        ]
    )


@pytest.fixture
def mock_centroids() -> pd.DataFrame:
    """Matching 3-zone centroids."""
    return pd.DataFrame(
        [
            {
                "ca_number": 1,
                "ca_name": "Rogers Park",
                "latitude": 42.01,
                "longitude": -87.67,
            },
            {
                "ca_number": 2,
                "ca_name": "West Ridge",
                "latitude": 42.00,
                "longitude": -87.69,
            },
            {
                "ca_number": 3,
                "ca_name": "Uptown",
                "latitude": 41.97,
                "longitude": -87.66,
            },
        ]
    )


@pytest.fixture
def mock_dist_matrix() -> pd.DataFrame:
    """Pairwise distance matrix for 3 zones in meters."""
    data = [
        [0.0, 3000.0, 5000.0],
        [3000.0, 0.0, 4000.0],
        [5000.0, 4000.0, 0.0],
    ]
    return pd.DataFrame(data, index=[1, 2, 3], columns=[1, 2, 3])


# ---------------------------------------------------------------------------
# Tests: Input Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validation test suite for UI parameters."""

    def test_transport_valid_params(self) -> None:
        errors = validate_transport_params("manhattan", 1.0)
        assert len(errors) == 0

    def test_transport_invalid_distance_metric(self) -> None:
        errors = validate_transport_params("euclidean", 1.0)
        assert len(errors) == 1
        assert "Invalid distance metric" in errors[0]

    def test_transport_invalid_demand_scale_zero(self) -> None:
        errors = validate_transport_params("manhattan", 0.0)
        assert len(errors) == 1
        assert "strictly positive" in errors[0]

    def test_transport_invalid_demand_scale_negative(self) -> None:
        errors = validate_transport_params("haversine", -0.5)
        assert len(errors) == 1
        assert "strictly positive" in errors[0]

    def test_transport_excessive_demand_scale(self) -> None:
        errors = validate_transport_params("manhattan", 15.0)
        assert len(errors) == 1
        assert "unusually large" in errors[0]

    def test_pmedian_valid_params(self) -> None:
        errors = validate_pmedian_params(p=3, max_p=77, distance_method="manhattan")
        assert len(errors) == 0

    def test_pmedian_p_less_than_one(self) -> None:
        errors = validate_pmedian_params(p=0, max_p=77, distance_method="manhattan")
        assert len(errors) == 1
        assert "at least 1" in errors[0]

    def test_pmedian_p_exceeds_max(self) -> None:
        errors = validate_pmedian_params(p=80, max_p=77, distance_method="manhattan")
        assert len(errors) == 1
        assert "cannot exceed available candidates" in errors[0]

    def test_pmedian_invalid_distance_metric(self) -> None:
        errors = validate_pmedian_params(p=3, max_p=77, distance_method="chebyshev")
        assert len(errors) == 1
        assert "Invalid distance metric" in errors[0]

    def test_pmedian_empty_candidates_list(self) -> None:
        errors = validate_pmedian_params(
            p=2, max_p=77, distance_method="manhattan", candidate_cas=[]
        )
        assert len(errors) == 1
        assert "cannot be empty" in errors[0]

    def test_pmedian_p_exceeds_candidates_list(self) -> None:
        errors = validate_pmedian_params(
            p=4, max_p=77, distance_method="manhattan", candidate_cas=[1, 2, 3]
        )
        assert len(errors) == 1
        assert "exceeds chosen candidates" in errors[0]

    def test_scenario_valid_params(self) -> None:
        errors = validate_scenario_params(
            p_values=[1, 2, 3], max_p=77, distance_methods=["manhattan", "haversine"]
        )
        assert len(errors) == 0

    def test_scenario_empty_p_values(self) -> None:
        errors = validate_scenario_params(
            p_values=[], max_p=77, distance_methods=["manhattan"]
        )
        assert len(errors) == 1
        assert "At least one depot count" in errors[0]

    def test_scenario_p_exceeds_max(self) -> None:
        errors = validate_scenario_params(
            p_values=[1, 90], max_p=77, distance_methods=["manhattan"]
        )
        assert len(errors) == 1
        assert "exceeds max candidate sites" in errors[0]

    def test_scenario_empty_distance_methods(self) -> None:
        errors = validate_scenario_params(
            p_values=[1, 2], max_p=77, distance_methods=[]
        )
        assert len(errors) == 1
        assert "At least one distance method" in errors[0]

    def test_scenario_invalid_demand_scale(self) -> None:
        errors = validate_scenario_params(
            p_values=[1, 2],
            max_p=77,
            distance_methods=["manhattan"],
            demand_scales=[1.0, -0.5],
        )
        assert len(errors) == 1
        assert "strictly positive" in errors[0]


# ---------------------------------------------------------------------------
# Tests: Data Loading Helpers
# ---------------------------------------------------------------------------


class TestDataLoadingHelpers:
    """Test filesystem loader helpers."""

    def test_load_cleaning_report_missing_file(self, tmp_path: Path) -> None:
        rep = load_cleaning_report(tmp_path / "nonexistent.json")
        assert rep == {}

    def test_load_cleaning_report_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "test_report.json"
        data = {"final_retained_rows": 525784, "status": "ok"}
        target.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_cleaning_report(target)
        assert loaded["final_retained_rows"] == 525784

    def test_load_eda_tables_missing_dir(self, tmp_path: Path) -> None:
        tables = load_eda_tables(tmp_path / "empty_dir")
        assert tables == {}

    def test_load_eda_summary_metrics(self, tmp_path: Path) -> None:
        json_path = tmp_path / "eda_summary_metrics.json"
        json_path.write_text(
            json.dumps({"total_cleaned_trips": 525784}), encoding="utf-8"
        )
        summary = load_eda_summary_metrics(tmp_path)
        assert summary["total_cleaned_trips"] == 525784


# ---------------------------------------------------------------------------
# Tests: Model Integration Wrappers
# ---------------------------------------------------------------------------


class TestSolverExecutionIntegration:
    """Integration tests executing actual optimization solvers through UI wrappers."""

    def test_execute_transport_solver_balanced(
        self,
        mock_zone_flows: pd.DataFrame,
        mock_dist_matrix: pd.DataFrame,
    ) -> None:
        result, alloc_df, _inputs = execute_transport_solver(
            zone_flows=mock_zone_flows,
            dist_matrix=mock_dist_matrix,
            distance_method="manhattan",
            demand_scale=1.0,
        )
        assert result.status == "optimal"
        assert len(alloc_df) == 1
        assert alloc_df.iloc[0]["Units Transferred"] == 50
        assert alloc_df.iloc[0]["Surplus CA"] == 1
        assert alloc_df.iloc[0]["Deficit CA"] == 2
        assert result.total_units_transferred == 50

    def test_execute_transport_solver_scaled_demand(
        self,
        mock_zone_flows: pd.DataFrame,
        mock_dist_matrix: pd.DataFrame,
    ) -> None:
        result, alloc_df, _inputs = execute_transport_solver(
            zone_flows=mock_zone_flows,
            dist_matrix=mock_dist_matrix,
            distance_method="manhattan",
            demand_scale=0.5,
        )
        assert result.status == "optimal"
        assert alloc_df.iloc[0]["Units Transferred"] == 25
        assert result.total_units_transferred == 25

    def test_execute_pmedian_solver_optimal(
        self,
        mock_zone_flows: pd.DataFrame,
        mock_centroids: pd.DataFrame,
        mock_dist_matrix: pd.DataFrame,
    ) -> None:
        result, depots_df, assign_df, _inputs = execute_pmedian_solver(
            zone_flows=mock_zone_flows,
            centroids=mock_centroids,
            dist_matrix=mock_dist_matrix,
            p=1,
            distance_method="manhattan",
        )
        assert result.status == "optimal"
        assert len(depots_df) == 1
        assert len(assign_df) == 3
        assert result.average_assignment_distance_meters >= 0.0

    def test_execute_pmedian_candidate_restriction(
        self,
        mock_zone_flows: pd.DataFrame,
        mock_centroids: pd.DataFrame,
        mock_dist_matrix: pd.DataFrame,
    ) -> None:
        result, depots_df, _assign_df, _inputs = execute_pmedian_solver(
            zone_flows=mock_zone_flows,
            centroids=mock_centroids,
            dist_matrix=mock_dist_matrix,
            p=1,
            distance_method="manhattan",
            candidate_cas=[2],
        )
        assert result.status == "optimal"
        assert depots_df.iloc[0]["Depot CA"] == 2

    def test_execute_scenario_suite_integration(self) -> None:
        results = execute_scenario_suite(
            p_values=[1],
            distance_methods=["manhattan"],
            include_transport=True,
            demand_scales=[1.0],
            eda_dir=Path("results/eda"),
        )
        assert len(results) == 2
        assert all(r.status == "optimal" for r in results)


# ---------------------------------------------------------------------------
# Tests: Plotting Helpers
# ---------------------------------------------------------------------------


class TestPlottingHelpers:
    """Test that all dashboard plotting helpers construct valid Plotly figures."""

    def test_create_hourly_profile_fig(self) -> None:
        df = pd.DataFrame(
            {
                "hour": list(range(24)),
                "trip_count": [100] * 24,
                "mean_speed_mph": [10.5] * 24,
            }
        )
        fig = create_hourly_profile_fig(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_create_day_of_week_fig(self) -> None:
        df = pd.DataFrame(
            {"day_name": ["Monday", "Tuesday", "Sunday"], "trip_count": [50, 60, 90]}
        )
        fig = create_day_of_week_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_daily_trend_fig(self) -> None:
        df = pd.DataFrame(
            {"date": ["2020-08-12", "2020-08-13"], "trip_count": [500, 600]}
        )
        fig = create_daily_trend_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_top_zones_fig(self, mock_zone_flows: pd.DataFrame) -> None:
        fig = create_top_zones_fig(mock_zone_flows, top_n=2)
        assert isinstance(fig, go.Figure)

    def test_create_net_flow_imbalance_fig(self, mock_zone_flows: pd.DataFrame) -> None:
        fig = create_net_flow_imbalance_fig(mock_zone_flows, top_n=2)
        assert isinstance(fig, go.Figure)

    def test_create_transport_allocation_fig(self) -> None:
        df = pd.DataFrame(
            [{"Surplus Name": "A", "Deficit Name": "B", "Units Transferred": 100}]
        )
        fig = create_transport_allocation_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_transport_allocation_fig_empty(self) -> None:
        fig = create_transport_allocation_fig(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_create_transport_dist_hist_fig(self) -> None:
        df = pd.DataFrame({"Distance (km)": [1.2, 3.4, 5.6]})
        fig = create_transport_dist_hist_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_pmedian_map_fig(self, mock_centroids: pd.DataFrame) -> None:
        assign_df = pd.DataFrame(
            [
                {
                    "Zone CA": 1,
                    "Zone Name": "Rogers Park",
                    "Assigned Depot CA": 2,
                    "Assigned Depot Name": "West Ridge",
                    "Demand Weight": 250,
                    "Distance (km)": 3.0,
                }
            ]
        )
        depots_df = pd.DataFrame(
            [
                {
                    "Depot CA": 2,
                    "Depot Name": "West Ridge",
                    "Latitude": 42.0,
                    "Longitude": -87.69,
                    "Assigned Demand Zones": 1,
                    "Assigned Total Weight": 250,
                }
            ]
        )
        fig = create_pmedian_map_fig(assign_df, depots_df, mock_centroids)
        assert isinstance(fig, go.Figure)

    def test_create_pmedian_load_fig(self) -> None:
        df = pd.DataFrame([{"Depot Name": "West Ridge", "Assigned Total Weight": 500}])
        fig = create_pmedian_load_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_pmedian_dist_hist_fig(self) -> None:
        df = pd.DataFrame({"Distance (km)": [2.5, 4.0, 6.2]})
        fig = create_pmedian_dist_hist_fig(df)
        assert isinstance(fig, go.Figure)

    def test_create_scenario_elbow_fig(self) -> None:
        res = [
            ScenarioResult(
                config=ScenarioConfig(
                    scenario_id="p1", model="pmedian", p=1, distance_method="manhattan"
                ),
                status="optimal",
                runtime_seconds=0.1,
                pmedian_objective_km=1000.0,
            ),
            ScenarioResult(
                config=ScenarioConfig(
                    scenario_id="p2", model="pmedian", p=2, distance_method="manhattan"
                ),
                status="optimal",
                runtime_seconds=0.1,
                pmedian_objective_km=600.0,
            ),
        ]
        fig = create_scenario_elbow_fig(res)
        assert isinstance(fig, go.Figure)

    def test_create_scenario_avg_dist_fig(self) -> None:
        res = [
            ScenarioResult(
                config=ScenarioConfig(
                    scenario_id="p1", model="pmedian", p=1, distance_method="manhattan"
                ),
                status="optimal",
                runtime_seconds=0.1,
                pmedian_avg_dist_m=8500.0,
            )
        ]
        fig = create_scenario_avg_dist_fig(res)
        assert isinstance(fig, go.Figure)

    def test_create_scenario_transport_obj_fig(self) -> None:
        res = [
            ScenarioResult(
                config=ScenarioConfig(
                    scenario_id="t1", model="transport", label="baseline_manhattan"
                ),
                status="optimal",
                runtime_seconds=0.1,
                transport_objective_km=31000.0,
            )
        ]
        fig = create_scenario_transport_obj_fig(res)
        assert isinstance(fig, go.Figure)
