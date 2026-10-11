"""Unit tests for Phase 4: P-Median Facility Location IP.

Coverage:
  - build_pmedian_inputs(): parsing, validation, alignment.
  - _validate_inputs(): invalid p, negative/NaN weights, wrong matrix shape.
  - solve_pmedian(): small known-optimal examples, constraint satisfaction,
    objective recomputation, infeasible handling.
  - save_results(): output file creation.
  - format_solution_report(): disclaimer presence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from escooter_opt.optimization.pmedian import (
    PMedianInputs,
    _validate_inputs,
    build_pmedian_inputs,
    format_solution_report,
    save_results,
    solve_pmedian,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zone_flows(
    zones: list[tuple[int, str, int]],  # (ca_number, ca_name, turnover)
    lat_base: float = 41.8,
    lon_base: float = -87.7,
) -> pd.DataFrame:
    rows = []
    for idx, (ca, name, turnover) in enumerate(zones):
        rows.append(
            {
                "ca_number": ca,
                "ca_name": name,
                "departures": turnover // 2,
                "arrivals": turnover - turnover // 2,
                "net_flow": 0,
                "turnover": turnover,
                "imbalance_ratio": 0.0,
                "flow_category": "balanced",
                "latitude": lat_base + idx * 0.02,
                "longitude": lon_base + idx * 0.02,
            }
        )
    return pd.DataFrame(rows)


def _make_centroids(
    zones: list[tuple[int, str, int]], lat_base: float = 41.8, lon_base: float = -87.7
) -> pd.DataFrame:
    rows = []
    for idx, (ca, name, _) in enumerate(zones):
        rows.append(
            {
                "ca_number": ca,
                "ca_name": name,
                "latitude": lat_base + idx * 0.02,
                "longitude": lon_base + idx * 0.02,
                "location_wkt": "",
            }
        )
    return pd.DataFrame(rows)


def _make_dist_matrix(cas: list[int], dist_fn=None) -> pd.DataFrame:
    n = len(cas)
    mat = np.zeros((n, n))
    for i, ca_i in enumerate(cas):
        for j, ca_j in enumerate(cas):
            if i != j:
                mat[i, j] = dist_fn(i, j) if dist_fn else abs(ca_i - ca_j) * 1000.0
    return pd.DataFrame(mat, index=cas, columns=cas)


# Small 4-zone test setup:
# Zones: A(ca=1), B(ca=2), C(ca=3), D(ca=4)
# Distances (symmetric): A-B=1000, A-C=5000, A-D=6000, B-C=4500, B-D=5500, C-D=1500
# Weights: A=100, B=200, C=150, D=300
# With p=2, optimal depots should be B and C (or D) to minimize weighted sum.

ZONES_4 = [(1, "ZoneA", 100), (2, "ZoneB", 200), (3, "ZoneC", 150), (4, "ZoneD", 300)]
CAS_4 = [z[0] for z in ZONES_4]
DISTS_4 = {
    (0, 1): 1000.0,
    (0, 2): 5000.0,
    (0, 3): 6000.0,
    (1, 2): 4500.0,
    (1, 3): 5500.0,
    (2, 3): 1500.0,
}


def _dist4(i: int, j: int) -> float:
    key = (min(i, j), max(i, j))
    return DISTS_4.get(key, 0.0)


@pytest.fixture
def inputs_p2() -> PMedianInputs:
    """4-zone, p=2 test case."""
    zones = _make_zone_flows(ZONES_4)
    centroids = _make_centroids(ZONES_4)
    dist = _make_dist_matrix(CAS_4, _dist4)
    return build_pmedian_inputs(zones, centroids, dist, p=2)


@pytest.fixture
def inputs_p1() -> PMedianInputs:
    """4-zone, p=1 test case."""
    zones = _make_zone_flows(ZONES_4)
    centroids = _make_centroids(ZONES_4)
    dist = _make_dist_matrix(CAS_4, _dist4)
    return build_pmedian_inputs(zones, centroids, dist, p=1)


@pytest.fixture
def inputs_all_depots() -> PMedianInputs:
    """4-zone, p=4 — every zone is its own depot (trivially zero cost)."""
    zones = _make_zone_flows(ZONES_4)
    centroids = _make_centroids(ZONES_4)
    dist = _make_dist_matrix(CAS_4, _dist4)
    return build_pmedian_inputs(zones, centroids, dist, p=4)


# ---------------------------------------------------------------------------
# Tests: _validate_inputs
# ---------------------------------------------------------------------------


class TestValidateInputs:
    def _make_valid(self, p: int = 2):
        dz = pd.DataFrame(
            {
                "ca_number": [1, 2, 3],
                "ca_name": ["A", "B", "C"],
                "weight": [100.0, 200.0, 150.0],
                "latitude": [41.8, 41.82, 41.84],
                "longitude": [-87.7, -87.72, -87.74],
            }
        )
        cd = pd.DataFrame(
            {
                "ca_number": [1, 2, 3],
                "ca_name": ["A", "B", "C"],
                "latitude": [41.8, 41.82, 41.84],
                "longitude": [-87.7, -87.72, -87.74],
            }
        )
        cm = pd.DataFrame(
            [[0, 1000, 2000], [1000, 0, 1500], [2000, 1500, 0]],
            index=[1, 2, 3],
            columns=[1, 2, 3],
        ).astype(float)
        return dz, cd, cm, p

    def test_valid_passes(self):
        dz, cd, cm, p = self._make_valid()
        _validate_inputs(dz, cd, cm, p)  # should not raise

    def test_p_zero_raises(self):
        dz, cd, cm, _ = self._make_valid()
        with pytest.raises(ValueError, match="p=0"):
            _validate_inputs(dz, cd, cm, 0)

    def test_p_too_large_raises(self):
        dz, cd, cm, _ = self._make_valid()
        with pytest.raises(ValueError, match="p=4"):
            _validate_inputs(dz, cd, cm, 4)

    def test_p_float_raises(self):
        dz, cd, cm, _ = self._make_valid()
        with pytest.raises(TypeError, match="plain integer"):
            _validate_inputs(dz, cd, cm, 2.0)  # type: ignore

    def test_p_bool_raises(self):
        dz, cd, cm, _ = self._make_valid()
        with pytest.raises(TypeError, match="plain integer"):
            _validate_inputs(dz, cd, cm, True)  # type: ignore

    def test_negative_weight_raises(self):
        dz, cd, cm, p = self._make_valid()
        dz = dz.copy()
        dz.loc[0, "weight"] = -1.0
        with pytest.raises(ValueError, match="negative weight"):
            _validate_inputs(dz, cd, cm, p)

    def test_nan_weight_raises(self):
        dz, cd, cm, p = self._make_valid()
        dz = dz.copy()
        dz.loc[1, "weight"] = float("nan")
        with pytest.raises(ValueError, match="NaN weight"):
            _validate_inputs(dz, cd, cm, p)

    def test_wrong_matrix_shape_raises(self):
        dz, cd, cm, p = self._make_valid()
        bad_cm = cm.iloc[:2, :]  # wrong number of rows
        with pytest.raises(ValueError, match="shape"):
            _validate_inputs(dz, cd, bad_cm, p)

    def test_negative_distance_raises(self):
        dz, cd, cm, p = self._make_valid()
        bad_cm = cm.copy()
        bad_cm.iloc[0, 1] = -5.0
        with pytest.raises(ValueError, match="negative distance"):
            _validate_inputs(dz, cd, bad_cm, p)

    def test_nan_distance_raises(self):
        dz, cd, cm, p = self._make_valid()
        bad_cm = cm.copy()
        bad_cm.iloc[1, 2] = float("nan")
        with pytest.raises(ValueError, match="NaN values"):
            _validate_inputs(dz, cd, bad_cm, p)

    def test_empty_demand_zones_raises(self):
        _, cd, cm, p = self._make_valid()
        empty_dz = pd.DataFrame(columns=["ca_number", "ca_name", "weight"])
        with pytest.raises(ValueError, match="empty"):
            _validate_inputs(empty_dz, cd, cm, p)


# ---------------------------------------------------------------------------
# Tests: build_pmedian_inputs
# ---------------------------------------------------------------------------


class TestBuildPMedianInputs:
    def test_zone_count(self, inputs_p2: PMedianInputs):
        assert inputs_p2.n_zones == 4

    def test_candidate_count(self, inputs_p2: PMedianInputs):
        assert inputs_p2.n_candidates == 4

    def test_p_set(self, inputs_p2: PMedianInputs):
        assert inputs_p2.p == 2

    def test_weights_positive(self, inputs_p2: PMedianInputs):
        assert (inputs_p2.demand_zones["weight"] > 0).all()

    def test_cost_matrix_shape(self, inputs_p2: PMedianInputs):
        assert inputs_p2.cost_matrix.shape == (4, 4)

    def test_cost_matrix_diagonal_zero(self, inputs_p2: PMedianInputs):
        for idx in range(4):
            assert inputs_p2.cost_matrix.iloc[idx, idx] == pytest.approx(0.0)

    def test_distance_method_label(self, inputs_p2: PMedianInputs):
        assert inputs_p2.distance_method == "manhattan"

    def test_missing_zone_in_dist_raises(self):
        zones = _make_zone_flows(ZONES_4)
        centroids = _make_centroids(ZONES_4)
        dist = _make_dist_matrix([1, 2, 3])  # missing ca=4
        with pytest.raises(KeyError, match="missing"):
            build_pmedian_inputs(zones, centroids, dist, p=2)

    def test_candidate_subset(self):
        zones = _make_zone_flows(ZONES_4)
        centroids = _make_centroids(ZONES_4)
        dist = _make_dist_matrix(CAS_4, _dist4)
        inp = build_pmedian_inputs(
            zones, centroids, dist, p=2, candidate_ca_numbers=[1, 2]
        )
        assert inp.n_candidates == 2
        assert set(inp.candidate_depots["ca_number"].tolist()) == {1, 2}

    def test_zero_turnover_zones_excluded(self):
        zones_with_zero = ZONES_4 + [(5, "ZeroZone", 0)]
        zones = _make_zone_flows(zones_with_zero)
        centroids = _make_centroids(zones_with_zero)
        dist = _make_dist_matrix([1, 2, 3, 4, 5], lambda i, j: abs(i - j) * 1000.0)
        inp = build_pmedian_inputs(zones, centroids, dist, p=2)
        assert 5 not in inp.demand_zones["ca_number"].tolist()


# ---------------------------------------------------------------------------
# Tests: solve_pmedian
# ---------------------------------------------------------------------------


class TestSolvePMedian:
    def test_status_optimal_p2(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert r.status == "optimal"

    def test_exactly_p_depots_selected(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert len(r.selected_depots) == 2

    def test_exactly_p1_depots(self, inputs_p1: PMedianInputs):
        r = solve_pmedian(inputs_p1)
        assert len(r.selected_depots) == 1

    def test_every_zone_assigned_exactly_once(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert len(r.assignments) == inputs_p2.n_zones
        zone_cas = {a.zone_ca for a in r.assignments}
        expected_cas = set(inputs_p2.demand_zones["ca_number"].tolist())
        assert zone_cas == expected_cas

    def test_no_zone_assigned_to_unselected_depot(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        selected_depot_cas = {sd.depot_ca for sd in r.selected_depots}
        for a in r.assignments:
            assert a.depot_ca in selected_depot_cas

    def test_p_equals_n_objective_zero(self, inputs_all_depots: PMedianInputs):
        """When p = |I|, every zone is its own depot — assignment distance = 0."""
        r = solve_pmedian(inputs_all_depots)
        assert r.status == "optimal"
        assert r.objective_value == pytest.approx(0.0, abs=1.0)

    def test_objective_independently_verified(self, inputs_p2: PMedianInputs):
        """Recompute objective from returned assignments and check consistency."""
        r = solve_pmedian(inputs_p2)
        recomputed = sum(a.weighted_distance for a in r.assignments)
        assert abs(recomputed - r.objective_value) < 5.0  # floating-point tolerance

    def test_objective_km_consistent(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert abs(r.objective_value_km - r.objective_value / 1000.0) < 0.01

    def test_total_unweighted_distance_consistent(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        recomputed = sum(a.distance_meters for a in r.assignments)
        assert abs(recomputed - r.total_unweighted_distance_meters) < 5.0

    def test_average_assignment_distance_consistent(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        expected_avg = r.total_unweighted_distance_meters / r.n_zones
        assert abs(r.average_assignment_distance_meters - expected_avg) < 1.0

    def test_zone_counts_match_input(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert r.n_zones == 4
        assert r.n_candidates == 4
        assert r.p == 2

    def test_all_assignments_have_nonneg_distance(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        for a in r.assignments:
            assert a.distance_meters >= 0.0

    def test_all_depots_have_nonneg_stats(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        for sd in r.selected_depots:
            assert sd.zones_served >= 0
            assert sd.total_weight_served >= 0.0

    def test_known_optimal_single_pair(self):
        """2 zones, p=1: only candidate is zone 1 → all zones assigned to it."""
        zones = _make_zone_flows([(1, "Alpha", 100), (2, "Beta", 100)])
        centroids = _make_centroids([(1, "Alpha", 100), (2, "Beta", 100)])
        dist = _make_dist_matrix([1, 2], lambda i, j: 3000.0 if i != j else 0.0)
        inp = build_pmedian_inputs(zones, centroids, dist, p=1)
        r = solve_pmedian(inp)
        assert r.status == "optimal"
        assert len(r.selected_depots) == 1
        assert len(r.assignments) == 2

    def test_infeasible_p_too_large_caught_by_validation(self):
        """p > n_candidates is rejected at build step."""
        zones = _make_zone_flows(ZONES_4)
        centroids = _make_centroids(ZONES_4)
        dist = _make_dist_matrix(CAS_4, _dist4)
        with pytest.raises(ValueError, match="out of range"):
            build_pmedian_inputs(zones, centroids, dist, p=99)

    def test_distance_method_label_preserved(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert r.distance_method == "manhattan"

    def test_solved_at_utc_set(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        assert r.solved_at_utc  # non-empty

    def test_to_dict_has_assignments(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        d = r.to_dict()
        assert "assignments" in d
        assert "selected_depots" in d

    def test_assignment_distance_km_property(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        for a in r.assignments:
            assert abs(a.distance_km - a.distance_meters / 1000.0) < 0.001


# ---------------------------------------------------------------------------
# Tests: save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_files_created(self, inputs_p2: PMedianInputs, tmp_path: Path):
        r = solve_pmedian(inputs_p2)
        save_results(r, tmp_path)
        assert (tmp_path / "pmedian_solution.json").is_file()
        assert (tmp_path / "pmedian_summary.csv").is_file()
        assert (tmp_path / "selected_depots.csv").is_file()
        assert (tmp_path / "zone_assignments.csv").is_file()

    def test_json_status_optimal(self, inputs_p2: PMedianInputs, tmp_path: Path):
        r = solve_pmedian(inputs_p2)
        save_results(r, tmp_path)
        with open(tmp_path / "pmedian_solution.json") as f:
            data = json.load(f)
        assert data["status"] == "optimal"

    def test_depots_csv_has_p_rows(self, inputs_p2: PMedianInputs, tmp_path: Path):
        r = solve_pmedian(inputs_p2)
        save_results(r, tmp_path)
        df = pd.read_csv(tmp_path / "selected_depots.csv")
        assert len(df) == inputs_p2.p

    def test_assignments_csv_has_n_rows(self, inputs_p2: PMedianInputs, tmp_path: Path):
        r = solve_pmedian(inputs_p2)
        save_results(r, tmp_path)
        df = pd.read_csv(tmp_path / "zone_assignments.csv")
        assert len(df) == inputs_p2.n_zones

    def test_summary_csv_columns(self, inputs_p2: PMedianInputs, tmp_path: Path):
        r = solve_pmedian(inputs_p2)
        save_results(r, tmp_path)
        df = pd.read_csv(tmp_path / "pmedian_summary.csv")
        required = {"status", "p", "n_zones", "n_candidates", "objective_value_km"}
        assert required.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Tests: format_solution_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_disclaimer_present(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        report = format_solution_report(r)
        assert "DISCLAIMER" in report or "disclaimer" in report.lower()

    def test_status_in_report(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        report = format_solution_report(r)
        assert "optimal" in report

    def test_p_in_report(self, inputs_p2: PMedianInputs):
        r = solve_pmedian(inputs_p2)
        report = format_solution_report(r)
        assert "p=2" in report or "Depots (p):      2" in report
