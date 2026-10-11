"""Unit tests for Phase 3: Transportation Problem solver.

Tests cover:
  - build_transport_inputs(): correct parsing of zone flows and cost matrix.
  - solve_transport_problem(): optimality, feasibility, and integer rounding.
  - Slack balancing for unbalanced supply/demand.
  - Edge cases: empty inputs, single zone pairs, zero-distance diagonals.
  - Disclaimer labels on returned data structures.
  - save_results() writes expected files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from escooter_opt.optimization.transport import (
    TransportProblemInputs,
    _balance_problem,
    build_transport_inputs,
    format_solution_report,
    save_results,
    solve_transport_problem,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_zone_flows(
    surplus_pairs: list[tuple[int, str, int]], deficit_pairs: list[tuple[int, str, int]]
) -> pd.DataFrame:
    """Create a minimal zone_flows DataFrame."""
    rows = []
    for ca, name, flow in surplus_pairs:
        rows.append(
            {
                "ca_number": ca,
                "ca_name": name,
                "net_flow": flow,
                "flow_category": "potential_surplus",
                "departures": 100,
                "arrivals": 100 + flow,
            }
        )
    for ca, name, flow in deficit_pairs:
        rows.append(
            {
                "ca_number": ca,
                "ca_name": name,
                "net_flow": flow,  # negative
                "flow_category": "potential_deficit",
                "departures": 100 + abs(flow),
                "arrivals": 100,
            }
        )
    return pd.DataFrame(rows)


def _make_dist_matrix(
    all_cas: list[int], distances: dict[tuple[int, int], float] | None = None
) -> pd.DataFrame:
    """Create a symmetric distance matrix for given CA numbers."""
    n = len(all_cas)
    mat = np.zeros((n, n))
    for i, ca_i in enumerate(all_cas):
        for j, ca_j in enumerate(all_cas):
            if i == j:
                mat[i, j] = 0.0
            elif distances and (ca_i, ca_j) in distances:
                mat[i, j] = distances[(ca_i, ca_j)]
            elif distances and (ca_j, ca_i) in distances:
                mat[i, j] = distances[(ca_j, ca_i)]
            else:
                # Default: 1000m per zone-number difference
                mat[i, j] = abs(ca_i - ca_j) * 1000.0
    return pd.DataFrame(mat, index=all_cas, columns=all_cas)


@pytest.fixture
def simple_inputs() -> TransportProblemInputs:
    """2-surplus, 2-deficit balanced problem."""
    zone_flows = _make_zone_flows(
        surplus_pairs=[(1, "ZoneA", 100), (2, "ZoneB", 200)],
        deficit_pairs=[(3, "ZoneC", -150), (4, "ZoneD", -150)],
    )
    dist = _make_dist_matrix(
        [1, 2, 3, 4],
        {
            (1, 3): 2000.0,
            (1, 4): 5000.0,
            (2, 3): 6000.0,
            (2, 4): 1500.0,
        },
    )
    return build_transport_inputs(zone_flows, dist, distance_method="manhattan")


@pytest.fixture
def unbalanced_excess_supply_inputs() -> TransportProblemInputs:
    """Supply > demand — requires slack deficit zone."""
    zone_flows = _make_zone_flows(
        surplus_pairs=[(1, "ZoneA", 300)],
        deficit_pairs=[(2, "ZoneB", -100)],
    )
    dist = _make_dist_matrix([1, 2], {(1, 2): 3000.0})
    return build_transport_inputs(zone_flows, dist)


@pytest.fixture
def unbalanced_excess_demand_inputs() -> TransportProblemInputs:
    """Demand > supply — requires slack surplus zone."""
    zone_flows = _make_zone_flows(
        surplus_pairs=[(1, "ZoneA", 50)],
        deficit_pairs=[(2, "ZoneB", -200)],
    )
    dist = _make_dist_matrix([1, 2], {(1, 2): 4000.0})
    return build_transport_inputs(zone_flows, dist)


# ---------------------------------------------------------------------------
# Tests: build_transport_inputs
# ---------------------------------------------------------------------------


class TestBuildTransportInputs:
    def test_surplus_deficit_split(self, simple_inputs: TransportProblemInputs) -> None:
        assert len(simple_inputs.surplus_zones) == 2
        assert len(simple_inputs.deficit_zones) == 2

    def test_supply_values_positive(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        assert (simple_inputs.surplus_zones["supply"] > 0).all()

    def test_demand_values_positive(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        assert (simple_inputs.deficit_zones["demand"] > 0).all()

    def test_total_supply_matches(self, simple_inputs: TransportProblemInputs) -> None:
        assert simple_inputs.total_supply == 300

    def test_total_demand_matches(self, simple_inputs: TransportProblemInputs) -> None:
        assert simple_inputs.total_demand == 300

    def test_cost_matrix_shape(self, simple_inputs: TransportProblemInputs) -> None:
        assert simple_inputs.cost_matrix.shape == (2, 2)

    def test_cost_matrix_nonnegative(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        assert (simple_inputs.cost_matrix >= 0).all().all()

    def test_distance_method_label(self, simple_inputs: TransportProblemInputs) -> None:
        assert simple_inputs.distance_method == "manhattan"

    def test_missing_zone_raises(self) -> None:
        zone_flows = _make_zone_flows(
            surplus_pairs=[(1, "ZoneA", 10)],
            deficit_pairs=[(99, "ZoneMissing", -10)],  # 99 not in distance matrix
        )
        dist = _make_dist_matrix([1, 2])
        with pytest.raises(KeyError, match="Distance matrix missing zones"):
            build_transport_inputs(zone_flows, dist)


# ---------------------------------------------------------------------------
# Tests: _balance_problem
# ---------------------------------------------------------------------------


class TestBalanceProblem:
    def test_already_balanced(self) -> None:
        supply = np.array([100.0, 200.0])
        demand = np.array([150.0, 150.0])
        cost = np.ones((2, 2))
        s, d, c = _balance_problem(supply, demand, cost)
        np.testing.assert_array_equal(s, supply)
        np.testing.assert_array_equal(d, demand)
        assert c.shape == (2, 2)

    def test_excess_supply_adds_dummy_demand(self) -> None:
        supply = np.array([300.0])
        demand = np.array([100.0])
        cost = np.array([[5000.0]])
        _s, d, c = _balance_problem(supply, demand, cost)
        assert len(d) == 2
        assert d[-1] == 200.0  # slack
        assert c.shape == (1, 2)
        assert c[0, -1] == 0.0  # zero cost to slack

    def test_excess_demand_adds_dummy_supply(self) -> None:
        supply = np.array([50.0])
        demand = np.array([200.0])
        cost = np.array([[4000.0]])
        s, _d, c = _balance_problem(supply, demand, cost)
        assert len(s) == 2
        assert s[-1] == 150.0  # slack
        assert c.shape == (2, 1)
        assert c[-1, 0] == 0.0  # zero cost from slack


# ---------------------------------------------------------------------------
# Tests: solve_transport_problem
# ---------------------------------------------------------------------------


class TestSolveTransportProblem:
    def test_status_optimal(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.status == "optimal"

    def test_objective_nonnegative(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.objective_value_meters >= 0

    def test_allocations_are_nonnegative_integers(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        for a in result.allocations:
            assert isinstance(a.units, int)
            assert a.units > 0

    def test_supply_not_exceeded(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.rounded_supply_feasible

    def test_demand_satisfied(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.rounded_demand_feasible

    def test_balanced_problem_optimality(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        """For balanced 2×2 problem with distinct costs, verify minimum cost solution."""
        result = solve_transport_problem(simple_inputs)
        # ZoneA→ZoneC (cost 2000) and ZoneB→ZoneD (cost 1500) should be preferred
        # over ZoneA→ZoneD (cost 5000) and ZoneB→ZoneC (cost 6000)
        alloc_map = {(a.surplus_ca, a.deficit_ca): a.units for a in result.allocations}
        # Cheaper routes should carry more load
        total_cheap = alloc_map.get((1, 3), 0) + alloc_map.get((2, 4), 0)
        total_expensive = alloc_map.get((1, 4), 0) + alloc_map.get((2, 3), 0)
        assert total_cheap >= total_expensive

    def test_excess_supply_handled(
        self, unbalanced_excess_supply_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(unbalanced_excess_supply_inputs)
        assert result.status == "optimal"
        assert result.rounded_demand_feasible

    def test_excess_demand_handled(
        self, unbalanced_excess_demand_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(unbalanced_excess_demand_inputs)
        assert result.status == "optimal"
        # Supply utilisation capped at 100%
        assert result.supply_utilisation_pct <= 100.0

    def test_zone_counts_correct(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.surplus_zone_count == 2
        assert result.deficit_zone_count == 2

    def test_total_supply_demand_reported(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.total_supply == 300
        assert result.total_demand == 300

    def test_objective_km_consistent(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        assert (
            abs(result.objective_value_km - result.objective_value_meters / 1000.0)
            < 0.01
        )

    def test_distance_method_label_preserved(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.distance_method == "manhattan"

    def test_single_pair(self) -> None:
        """1-surplus, 1-deficit: entire supply/demand should be allocated."""
        zone_flows = _make_zone_flows(
            surplus_pairs=[(1, "Alpha", 50)],
            deficit_pairs=[(2, "Beta", -50)],
        )
        dist = _make_dist_matrix([1, 2], {(1, 2): 1234.0})
        inputs = build_transport_inputs(zone_flows, dist)
        result = solve_transport_problem(inputs)
        assert result.status == "optimal"
        assert len(result.allocations) == 1
        assert result.allocations[0].units == 50
        assert result.allocations[0].distance_meters == pytest.approx(1234.0, abs=1.0)
        assert result.total_units_transferred == 50

    def test_allocation_total_cost_consistent(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        computed_total = sum(a.total_cost_meters for a in result.allocations)
        assert (
            abs(computed_total - result.objective_value_meters) < 5.0
        )  # rounding tolerance


# ---------------------------------------------------------------------------
# Tests: format_solution_report
# ---------------------------------------------------------------------------


class TestFormatSolutionReport:
    def test_report_contains_disclaimer(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        report = format_solution_report(result)
        assert "estimated" in report.lower() or "heuristic" in report.lower()

    def test_report_contains_status(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        report = format_solution_report(result)
        assert "optimal" in report

    def test_report_contains_objective(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        report = format_solution_report(result)
        assert "km" in report


# ---------------------------------------------------------------------------
# Tests: save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_files_created(
        self, simple_inputs: TransportProblemInputs, tmp_path: Path
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        save_results(result, tmp_path)
        assert (tmp_path / "transport_solution.json").is_file()
        assert (tmp_path / "transport_summary.csv").is_file()
        assert (tmp_path / "allocations_detail.csv").is_file()

    def test_json_contains_status(
        self, simple_inputs: TransportProblemInputs, tmp_path: Path
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        save_results(result, tmp_path)
        with open(tmp_path / "transport_solution.json") as f:
            data = json.load(f)
        assert data["status"] == "optimal"

    def test_allocations_csv_nonnegative_units(
        self, simple_inputs: TransportProblemInputs, tmp_path: Path
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        save_results(result, tmp_path)
        df = pd.read_csv(tmp_path / "allocations_detail.csv")
        assert (df["units"] > 0).all()

    def test_summary_csv_columns(
        self, simple_inputs: TransportProblemInputs, tmp_path: Path
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        save_results(result, tmp_path)
        df = pd.read_csv(tmp_path / "transport_summary.csv")
        expected = {
            "status",
            "objective_value_km",
            "total_supply",
            "total_demand",
            "surplus_zone_count",
            "deficit_zone_count",
        }
        assert expected.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Tests: TransportProblemResult data structure
# ---------------------------------------------------------------------------


class TestTransportProblemResult:
    def test_to_dict_has_allocations_key(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        d = result.to_dict()
        assert "allocations" in d

    def test_allocations_in_dict_are_plain_dicts(
        self, simple_inputs: TransportProblemInputs
    ) -> None:
        result = solve_transport_problem(simple_inputs)
        d = result.to_dict()
        for item in d["allocations"]:
            assert isinstance(item, dict)

    def test_solved_at_utc_is_set(self, simple_inputs: TransportProblemInputs) -> None:
        result = solve_transport_problem(simple_inputs)
        assert result.solved_at_utc  # non-empty string

    def test_distance_km_property(self) -> None:
        from escooter_opt.optimization.transport import TransportAllocation

        a = TransportAllocation(
            surplus_ca=1,
            surplus_name="A",
            deficit_ca=2,
            deficit_name="B",
            units=10,
            distance_meters=5000.0,
            cost_unit_meters=5000.0,
            total_cost_meters=50000.0,
        )
        assert a.distance_km == pytest.approx(5.0)
        assert a.total_cost_km == pytest.approx(50.0)
