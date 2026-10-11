"""Unit tests for Phase 5: Scenario & Sensitivity Analysis.

Coverage:
  - ScenarioConfig validation (label defaulting, demand_scale).
  - ScenarioResult.to_row() completeness and typing.
  - build_pmedian_suite(): correct number/labelling of configs.
  - build_transport_suite(): baseline vs sensitivity distinction.
  - run_pmedian_scenario(): correct solve dispatch, metric extraction.
  - run_transport_scenario(): demand_scale applied, metric extraction.
  - run_scenario_suite(): EDA data reuse, result ordering.
  - Repeatability: two identical configs → identical objective.
  - Missing/failed scenarios: no fake values, status='error' or 'infeasible'.
  - Invalid parameters rejected before solve.
  - results_to_dataframe(): one row per scenario, expected columns.
  - save_suite_results(): files created, JSON structure.
  - format_suite_report(): disclaimer present, summary lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from escooter_opt.scenarios.runner import (
    ScenarioConfig,
    ScenarioResult,
    build_pmedian_suite,
    build_transport_suite,
    format_suite_report,
    results_to_dataframe,
    run_pmedian_scenario,
    run_scenario_suite,
    run_transport_scenario,
    save_suite_results,
)

# ---------------------------------------------------------------------------
# Tiny synthetic dataset helpers (reused from Phase 3/4 pattern)
# ---------------------------------------------------------------------------


def _zone_flows_4() -> pd.DataFrame:
    """4-zone dataset with 2 surplus and 2 deficit zones."""
    return pd.DataFrame(
        [
            {
                "ca_number": 1,
                "ca_name": "Alpha",
                "departures": 120,
                "arrivals": 220,
                "net_flow": 100,
                "turnover": 340,
                "imbalance_ratio": 0.14,
                "flow_category": "potential_surplus",
                "latitude": 41.80,
                "longitude": -87.70,
            },
            {
                "ca_number": 2,
                "ca_name": "Beta",
                "departures": 200,
                "arrivals": 400,
                "net_flow": 200,
                "turnover": 600,
                "imbalance_ratio": 0.20,
                "flow_category": "potential_surplus",
                "latitude": 41.82,
                "longitude": -87.72,
            },
            {
                "ca_number": 3,
                "ca_name": "Gamma",
                "departures": 350,
                "arrivals": 200,
                "net_flow": -150,
                "turnover": 550,
                "imbalance_ratio": -0.14,
                "flow_category": "potential_deficit",
                "latitude": 41.84,
                "longitude": -87.74,
            },
            {
                "ca_number": 4,
                "ca_name": "Delta",
                "departures": 250,
                "arrivals": 100,
                "net_flow": -150,
                "turnover": 350,
                "imbalance_ratio": -0.21,
                "flow_category": "potential_deficit",
                "latitude": 41.86,
                "longitude": -87.76,
            },
        ]
    )


def _centroids_4() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ca_number": ca,
                "ca_name": name,
                "latitude": lat,
                "longitude": lon,
                "location_wkt": "",
            }
            for ca, name, lat, lon in [
                (1, "Alpha", 41.80, -87.70),
                (2, "Beta", 41.82, -87.72),
                (3, "Gamma", 41.84, -87.74),
                (4, "Delta", 41.86, -87.76),
            ]
        ]
    )


def _dist_matrix_4() -> pd.DataFrame:
    cas = [1, 2, 3, 4]
    dists = {
        (1, 2): 2000.0,
        (1, 3): 4000.0,
        (1, 4): 6000.0,
        (2, 3): 3000.0,
        (2, 4): 4500.0,
        (3, 4): 2500.0,
    }
    mat = np.zeros((4, 4))
    for i, ci in enumerate(cas):
        for j, cj in enumerate(cas):
            if i != j:
                key = (min(ci, cj), max(ci, cj))
                mat[i, j] = dists.get(key, abs(ci - cj) * 1000.0)
    return pd.DataFrame(mat, index=cas, columns=cas)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zone_flows() -> pd.DataFrame:
    return _zone_flows_4()


@pytest.fixture
def centroids() -> pd.DataFrame:
    return _centroids_4()


@pytest.fixture
def dist_matrix() -> pd.DataFrame:
    return _dist_matrix_4()


@pytest.fixture
def pmedian_cfg_p2() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="pmedian_p2_manhattan",
        model="pmedian",
        p=2,
        distance_method="manhattan",
    )


@pytest.fixture
def transport_baseline_cfg() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="transport_baseline",
        model="transport",
        distance_method="manhattan",
        demand_scale=1.0,
    )


@pytest.fixture
def transport_offpeak_cfg() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="transport_offpeak",
        model="transport",
        distance_method="manhattan",
        demand_scale=0.5,
    )


# ---------------------------------------------------------------------------
# Tests: ScenarioConfig
# ---------------------------------------------------------------------------


class TestScenarioConfig:
    def test_label_defaults_to_scenario_id(self) -> None:
        cfg = ScenarioConfig(scenario_id="test_123", model="pmedian", p=3)
        assert cfg.label == "test_123"

    def test_explicit_label_preserved(self) -> None:
        cfg = ScenarioConfig(
            scenario_id="test_123", model="pmedian", p=3, label="p=3 scenario"
        )
        assert cfg.label == "p=3 scenario"

    def test_demand_scale_default_is_one(self) -> None:
        cfg = ScenarioConfig(scenario_id="x", model="transport")
        assert cfg.demand_scale == 1.0

    def test_distance_method_default(self) -> None:
        cfg = ScenarioConfig(scenario_id="x", model="pmedian", p=1)
        assert cfg.distance_method == "manhattan"


# ---------------------------------------------------------------------------
# Tests: ScenarioResult.to_row
# ---------------------------------------------------------------------------


class TestScenarioResultToRow:
    def test_to_row_has_required_keys(self, pmedian_cfg_p2: ScenarioConfig) -> None:
        sr = ScenarioResult(
            config=pmedian_cfg_p2, status="optimal", runtime_seconds=0.1
        )
        row = sr.to_row()
        required = {"scenario_id", "model", "status", "runtime_s", "distance_method"}
        assert required.issubset(row.keys())

    def test_to_row_pmedian_fields_none_when_not_set(
        self, transport_baseline_cfg: ScenarioConfig
    ) -> None:
        sr = ScenarioResult(
            config=transport_baseline_cfg, status="optimal", runtime_seconds=0.05
        )
        row = sr.to_row()
        assert row["pm_obj_km_wt"] is None
        assert row["pm_avg_dist_m"] is None

    def test_to_row_transport_fields_none_when_not_set(
        self, pmedian_cfg_p2: ScenarioConfig
    ) -> None:
        sr = ScenarioResult(
            config=pmedian_cfg_p2, status="optimal", runtime_seconds=0.05
        )
        row = sr.to_row()
        assert row["tr_obj_km"] is None
        assert row["tr_units"] is None


# ---------------------------------------------------------------------------
# Tests: build_pmedian_suite
# ---------------------------------------------------------------------------


class TestBuildPmedianSuite:
    def test_correct_count(self) -> None:
        configs = build_pmedian_suite([1, 2, 3])
        assert len(configs) == 3

    def test_p_values_set(self) -> None:
        configs = build_pmedian_suite([2, 4])
        assert [c.p for c in configs] == [2, 4]

    def test_model_is_pmedian(self) -> None:
        configs = build_pmedian_suite([1, 2])
        assert all(c.model == "pmedian" for c in configs)

    def test_distance_method_propagated(self) -> None:
        configs = build_pmedian_suite([1, 2], distance_method="haversine")
        assert all(c.distance_method == "haversine" for c in configs)

    def test_unique_scenario_ids(self) -> None:
        configs = build_pmedian_suite([1, 2, 3, 4, 5])
        ids = [c.scenario_id for c in configs]
        assert len(ids) == len(set(ids))

    def test_labels_contain_p(self) -> None:
        configs = build_pmedian_suite([3])
        assert "3" in configs[0].label


# ---------------------------------------------------------------------------
# Tests: build_transport_suite
# ---------------------------------------------------------------------------


class TestBuildTransportSuite:
    def test_default_suite_has_four_scenarios(self) -> None:
        configs = build_transport_suite()
        assert len(configs) == 4

    def test_baseline_has_scale_1(self) -> None:
        configs = build_transport_suite()
        baselines = [c for c in configs if c.demand_scale == 1.0]
        assert len(baselines) >= 1

    def test_sensitivity_has_scale_lt_1(self) -> None:
        configs = build_transport_suite()
        sensitivity = [c for c in configs if c.demand_scale < 1.0]
        assert len(sensitivity) >= 1

    def test_model_is_transport(self) -> None:
        configs = build_transport_suite()
        assert all(c.model == "transport" for c in configs)

    def test_custom_suite(self) -> None:
        custom = [
            {"label": "custom_a", "distance_method": "manhattan", "demand_scale": 0.75},
        ]
        configs = build_transport_suite(custom)
        assert len(configs) == 1
        assert configs[0].demand_scale == 0.75

    def test_unique_scenario_ids(self) -> None:
        configs = build_transport_suite()
        ids = [c.scenario_id for c in configs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Tests: run_pmedian_scenario
# ---------------------------------------------------------------------------


class TestRunPmedianScenario:
    def test_status_optimal(
        self,
        pmedian_cfg_p2: ScenarioConfig,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_pmedian_scenario(pmedian_cfg_p2, zone_flows, centroids, dist_matrix)
        assert sr.status == "optimal"

    def test_objective_set(
        self,
        pmedian_cfg_p2: ScenarioConfig,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_pmedian_scenario(pmedian_cfg_p2, zone_flows, centroids, dist_matrix)
        assert sr.pmedian_objective_km is not None
        assert sr.pmedian_objective_km >= 0.0

    def test_avg_dist_set(
        self,
        pmedian_cfg_p2: ScenarioConfig,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_pmedian_scenario(pmedian_cfg_p2, zone_flows, centroids, dist_matrix)
        assert sr.pmedian_avg_dist_m is not None
        assert sr.pmedian_avg_dist_m >= 0.0

    def test_depots_count_equals_p(
        self,
        pmedian_cfg_p2: ScenarioConfig,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_pmedian_scenario(pmedian_cfg_p2, zone_flows, centroids, dist_matrix)
        assert len(sr.pmedian_depots) == 2

    def test_runtime_positive(
        self,
        pmedian_cfg_p2: ScenarioConfig,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_pmedian_scenario(pmedian_cfg_p2, zone_flows, centroids, dist_matrix)
        assert sr.runtime_seconds >= 0.0

    def test_invalid_p_returns_error(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        bad_cfg = ScenarioConfig(scenario_id="bad", model="pmedian", p=999)
        sr = run_pmedian_scenario(bad_cfg, zone_flows, centroids, dist_matrix)
        assert sr.status == "error"
        assert sr.pmedian_objective_km is None
        assert sr.error_message  # non-empty

    def test_p1_has_one_depot(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="p1", model="pmedian", p=1)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        assert sr.status == "optimal"
        assert len(sr.pmedian_depots) == 1

    def test_p_equals_n_objective_near_zero(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="p4", model="pmedian", p=4)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        assert sr.status == "optimal"
        assert sr.pmedian_objective_km == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Tests: run_transport_scenario
# ---------------------------------------------------------------------------


class TestRunTransportScenario:
    def test_baseline_status_optimal(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.status == "optimal"

    def test_baseline_objective_set(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.transport_objective_km is not None
        assert sr.transport_objective_km >= 0.0

    def test_offpeak_objective_less_than_baseline(
        self,
        transport_baseline_cfg: ScenarioConfig,
        transport_offpeak_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        """Off-peak (demand_scale=0.5) should have strictly lower objective."""
        sr_base = run_transport_scenario(
            transport_baseline_cfg, zone_flows, dist_matrix
        )
        sr_off = run_transport_scenario(transport_offpeak_cfg, zone_flows, dist_matrix)
        assert sr_base.status == "optimal"
        assert sr_off.status == "optimal"
        assert sr_off.transport_objective_km < sr_base.transport_objective_km

    def test_offpeak_units_less_than_baseline(
        self,
        transport_baseline_cfg: ScenarioConfig,
        transport_offpeak_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr_base = run_transport_scenario(
            transport_baseline_cfg, zone_flows, dist_matrix
        )
        sr_off = run_transport_scenario(transport_offpeak_cfg, zone_flows, dist_matrix)
        assert sr_off.transport_units < sr_base.transport_units  # type: ignore[operator]

    def test_supply_fully_utilised_baseline(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.transport_supply_util_pct == pytest.approx(100.0, abs=0.1)

    def test_demand_fully_satisfied_baseline(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.transport_demand_sat_pct == pytest.approx(100.0, abs=0.1)

    def test_runtime_positive(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.runtime_seconds >= 0.0

    def test_feasibility_flags_set(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr.transport_supply_feasible is True
        assert sr.transport_demand_feasible is True


# ---------------------------------------------------------------------------
# Tests: run_scenario_suite
# ---------------------------------------------------------------------------


class TestRunScenarioSuite:
    def test_result_order_preserved(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Results must be returned in the same order as the input configs."""
        # Monkeypatch EDA loaders: write minimal CSVs to tmp_path
        eda_dir = _write_minimal_eda(tmp_path, zone_flows, centroids, dist_matrix)
        configs = build_pmedian_suite([1, 2], distance_method="manhattan")
        results = run_scenario_suite(configs, eda_dir=eda_dir)
        assert len(results) == 2
        for cfg, sr in zip(configs, results):
            assert sr.config.scenario_id == cfg.scenario_id

    def test_unknown_model_returns_error(self, tmp_path: Path) -> None:
        eda_dir = tmp_path  # doesn't need real data for this path
        bad_cfg = ScenarioConfig(scenario_id="bad", model="unknown_model")
        results = run_scenario_suite([bad_cfg], eda_dir=eda_dir)
        assert results[0].status == "error"
        assert "Unknown model" in results[0].error_message

    def test_eda_data_loaded_once_per_dm(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Multiple configs with same distance_method should not fail."""
        eda_dir = _write_minimal_eda(tmp_path, zone_flows, centroids, dist_matrix)
        configs = build_pmedian_suite([1, 2, 3], distance_method="manhattan")
        results = run_scenario_suite(configs, eda_dir=eda_dir)
        assert all(r.status == "optimal" for r in results)


# ---------------------------------------------------------------------------
# Tests: repeatability
# ---------------------------------------------------------------------------


class TestRepeatability:
    def test_identical_configs_give_identical_objective(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        """Two runs of the same config should yield the same objective."""
        cfg = ScenarioConfig(scenario_id="rep", model="pmedian", p=2)
        sr1 = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        sr2 = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        assert sr1.pmedian_objective_km == pytest.approx(
            sr2.pmedian_objective_km, rel=1e-6
        )

    def test_transport_repeatability(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr1 = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        sr2 = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        assert sr1.transport_objective_km == pytest.approx(
            sr2.transport_objective_km, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Tests: results_to_dataframe
# ---------------------------------------------------------------------------


class TestResultsToDataframe:
    def test_one_row_per_result(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        configs = build_pmedian_suite([1, 2])
        results = [
            run_pmedian_scenario(c, zone_flows, centroids, dist_matrix) for c in configs
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2

    def test_expected_columns_present(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="x", model="pmedian", p=1)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        df = results_to_dataframe([sr])
        required = {"scenario_id", "model", "status", "runtime_s", "pm_obj_km_wt"}
        assert required.issubset(set(df.columns))

    def test_failed_scenario_row_has_no_objective(self) -> None:
        cfg = ScenarioConfig(scenario_id="fail", model="pmedian", p=99)
        sr = ScenarioResult(
            config=cfg,
            status="error",
            runtime_seconds=0.0,
            error_message="out of range",
        )
        df = results_to_dataframe([sr])
        assert pd.isna(df.iloc[0]["pm_obj_km_wt"])


# ---------------------------------------------------------------------------
# Tests: save_suite_results
# ---------------------------------------------------------------------------


class TestSaveSuiteResults:
    def test_files_created(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="save_test", model="pmedian", p=1)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        save_suite_results([sr], tmp_path, suite_name="test_suite")
        assert (tmp_path / "test_suite_results.csv").is_file()
        assert (tmp_path / "test_suite_results.json").is_file()

    def test_json_scenario_count(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        configs = build_pmedian_suite([1, 2])
        results = [
            run_pmedian_scenario(c, zone_flows, centroids, dist_matrix) for c in configs
        ]
        save_suite_results(results, tmp_path, "suite2")
        with open(tmp_path / "suite2_results.json") as f:
            data = json.load(f)
        assert data["n_scenarios"] == 2

    def test_csv_scenario_ids(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        configs = build_pmedian_suite([1, 2])
        results = [
            run_pmedian_scenario(c, zone_flows, centroids, dist_matrix) for c in configs
        ]
        save_suite_results(results, tmp_path, "suite3")
        df = pd.read_csv(tmp_path / "suite3_results.csv")
        assert set(df["scenario_id"].tolist()) == {c.scenario_id for c in configs}


# ---------------------------------------------------------------------------
# Tests: format_suite_report
# ---------------------------------------------------------------------------


class TestFormatSuiteReport:
    def test_disclaimer_present(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="rpt", model="pmedian", p=1)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        report = format_suite_report([sr])
        assert "DISCLAIMER" in report or "historical" in report.lower()

    def test_pmedian_header_present(
        self,
        zone_flows: pd.DataFrame,
        centroids: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        cfg = ScenarioConfig(scenario_id="rpt2", model="pmedian", p=2)
        sr = run_pmedian_scenario(cfg, zone_flows, centroids, dist_matrix)
        report = format_suite_report([sr])
        assert "P-Median" in report

    def test_transport_header_present(
        self,
        transport_baseline_cfg: ScenarioConfig,
        zone_flows: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> None:
        sr = run_transport_scenario(transport_baseline_cfg, zone_flows, dist_matrix)
        report = format_suite_report([sr])
        assert "Transport" in report


# ---------------------------------------------------------------------------
# Helper: write minimal EDA CSVs to tmp_path
# ---------------------------------------------------------------------------


def _write_minimal_eda(
    tmp_path: Path,
    zone_flows: pd.DataFrame,
    centroids: pd.DataFrame,
    dist_matrix: pd.DataFrame,
) -> Path:
    """Write the three EDA files needed by load_eda_outputs to a temp directory."""
    zone_flows.to_csv(tmp_path / "zone_flows_and_imbalance.csv", index=False)
    centroids.to_csv(tmp_path / "zone_centroids.csv", index=False)
    dist_matrix.to_csv(tmp_path / "zone_distances_manhattan_meters.csv")
    dist_matrix.to_csv(tmp_path / "zone_distances_haversine_meters.csv")
    return tmp_path
