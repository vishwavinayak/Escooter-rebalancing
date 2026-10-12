÷# Phase 4: P-Median Facility Location for E-Scooter Depot Siting

## Overview

Phase 4 formulates the e-scooter depot siting task as a **P-Median Integer
Program** and solves it with the HiGHS MILP solver via `scipy.optimize.milp`.
The model identifies the `p` optimal Community Area centroid locations to
place depots such that the demand-weighted total zone-to-depot distance is
minimised.

> **DISCLAIMER**
> Selected depot locations are modelled optimal candidates derived from
> Community Area centroids in the observed dataset.  They are **NOT** approved
> or physically available depot sites.  Any operational use requires independent
> site selection, regulatory approval, and ground-truth validation.

---

## Mathematical Formulation

### Sets

| Symbol | Description |
|--------|-------------|
| `I` | Eligible demand zones — all 77 Chicago Community Areas with valid centroids and non-zero turnover |
| `J` | Candidate depot locations — by default equal to `I`; may be restricted via `--candidate-cas` |

### Parameters

| Symbol | Definition |
|--------|-----------|
| `w_i ≥ 0` | Demand weight for zone `i`: **turnover** = arrivals + departures (Phase 2 EDA) |
| `d_{ij} ≥ 0` | Distance from demand zone `i` to candidate depot `j` (meters; Manhattan or haversine) |
| `p` | Configurable number of depots to site; integer, `1 ≤ p ≤ |J|` |

**Demand weight justification**: Turnover (`arrivals + departures`) reflects total
historical activity at a zone.  It weights both trip origins and destinations,
giving high-activity zones proportionally more influence on depot placement.  Zero-
turnover zones are excluded from the model.

### Decision Variables

| Symbol | Domain | Meaning |
|--------|--------|---------|
| `y_j` | `{0, 1}` | 1 if candidate depot `j` is selected |
| `z_{ij}` | `{0, 1}` | 1 if demand zone `i` is assigned to depot `j` |

### Objective

```
minimize  Σ_{i∈I} Σ_{j∈J}  w_i · d_{ij} · z_{ij}
```

Minimise the demand-weighted total zone-to-depot assignment distance.

### Constraints

```
Σ_j z_{ij} = 1          ∀ i ∈ I          (C1) each zone assigned to exactly one depot
z_{ij} ≤ y_j            ∀ i∈I, j∈J       (C2) assign only to a selected depot
Σ_j y_j = p                               (C3) exactly p depots selected
y_j ∈ {0,1}             ∀ j ∈ J          (C4) binary depot selection
z_{ij} ∈ {0,1}          ∀ i∈I, j∈J       (C5) binary zone assignment
```

**Variable count**: `|J| + |I|·|J|` = 77 + 5,929 = **6,006 binary variables** (for full 77-zone problem).

**Constraint count**: `|I| + |I|·|J| + 1` = 77 + 5,929 + 1 = **6,007 rows**.

---

## Solver

`scipy.optimize.milp` with the HiGHS backend (scipy ≥ 1.9).
A 300-second time limit is applied; all tested instances solve to optimality
within 2–4 seconds for p ≤ 10 on the full 77-zone problem.

---

## Input Data (from Phase 2 EDA)

| File | Contents used |
|------|--------------|
| `results/eda/zone_flows_and_imbalance.csv` | `turnover` (demand weight), `latitude`, `longitude` |
| `results/eda/zone_centroids.csv` | Candidate depot coordinates (observed centroids, never invented) |
| `results/eda/zone_distances_manhattan_meters.csv` | Default distance matrix |
| `results/eda/zone_distances_haversine_meters.csv` | Sensitivity distance matrix |

---

## Results (p=5, Manhattan distance, default run)

| Metric | Value |
|--------|-------|
| Demand zones \|I\| | 77 |
| Candidate depots \|J\| | 77 |
| Depots selected (p) | 5 |
| Solver status | **Optimal** (HiGHS Status 7) |
| **Weighted objective** | **3,501,234.3 km-weight** |
| Total unweighted distance | 507.4 km |
| Mean assignment distance | **6,589 m** |

### Selected Depot Locations (p=5)

| CA | Name | Zones Served | Wt-Dist (km) |
|----|------|---:|---:|
| 42 | WOODLAWN | 25 | 796,037 |
| 20 | HERMOSA | 19 | 649,545 |
| 63 | GAGE PARK | 17 | 465,853 |
| 6 | LAKE VIEW | 8 | 995,340 |
| 24 | WEST TOWN | 8 | 594,461 |

These centroids span the North Side high-activity corridor (Lake View, West
Town, Hermosa), the central corridor (Gage Park), and the South Side
(Woodlawn), giving broad geographic coverage across all 77 zones.

### Elbow Curve: Objective vs p

| p | Status | Weighted Obj (km-wt) | Avg Distance (m) |
|---|--------|---------------------:|------------------:|
| 1 | optimal | 9,034,598 | 15,599 |
| 2 | optimal | 5,947,713 | 9,407 |
| 3 | optimal | 4,474,754 | 8,633 |
| 4 | optimal | 3,967,280 | 7,780 |
| **5** | **optimal** | **3,501,234** | **6,589** |
| 6 | optimal | 3,039,902 | 6,534 |
| 7 | optimal | 2,696,242 | 5,507 |
| 8 | optimal | 2,390,132 | 5,232 |
| 9 | optimal | 2,112,059 | 4,891 |
| 10 | optimal | 1,892,260 | 4,684 |

The largest marginal reduction occurs between p=1 and p=2 (−34%) and p=2 and
p=3 (−25%).  Diminishing returns become visible beyond p=5.

---

## Output Files

All outputs are written to `results/pmedian/`:

| File | Description |
|------|-------------|
| `pmedian_solution.json` | Full result including all depots, assignments, metadata |
| `pmedian_summary.csv` | One-row summary: status, objective, distances, p |
| `selected_depots.csv` | One row per selected depot with coordinates and service stats |
| `zone_assignments.csv` | One row per demand zone with depot assignment and distances |
| `figures/zone_depot_map.png` | Scatter map of zones (sized by weight) + selected depots (stars) |
| `figures/depot_summary_bars.png` | Per-depot zones served and weighted cost bar charts |
| `figures/assignment_distance_hist.png` | Histogram of zone-to-depot assignment distances |
| `figures/elbow_curve.png` | Objective vs p for p = 1..10 |

---

## Usage

```bash
# Default: p=5, manhattan distance
uv run escooter-pmedian

# Custom p
uv run escooter-pmedian --p 7

# Haversine sensitivity
uv run escooter-pmedian --p 5 --distance-method haversine

# Restrict candidates to specific Community Areas
# (done via code: candidate_ca_numbers=[...] in build_pmedian_inputs)

# No figures
uv run escooter-pmedian --p 5 --no-figures

# Help
uv run escooter-pmedian --help
```

---

## Validation

```
tests/test_pmedian.py — 48 unit tests, all pass
ruff check — 0 errors
```

Tested:
- Known-optimal small instances (including trivial p=n case with zero cost)
- Exactly p depots selected
- Every zone assigned exactly once
- No zone assigned to an unselected depot
- Invalid p (0, too large, float, bool) → `ValueError` / `TypeError`
- Negative/NaN weights → `ValueError`
- Negative/NaN distances → `ValueError`
- Objective independently recomputed from returned assignments
- Output CSV and JSON file correctness

---

## Known Limitations

1. **Centroid locations only**: All candidate sites and demand zone positions
   are Community Area centroids.  Actual within-zone intra-zone distances
   are not captured.

2. **Demand weights are historical**: Turnover is derived from 2020 trip data;
   temporal variability (time-of-day, seasonality) is not modelled.

3. **No operational constraints**: Fleet size, vehicle capacity, rebalancing
   frequency, and staffing costs are not included.

4. **No service radius**: By default there is no distance cutoff.  A service
   radius constraint can be implemented via `candidate_ca_numbers` filtering
   or by setting large distances to infinity before passing to the solver.

5. **Centroid-level distances**: Transfer costs use zone centroids as depot
   positions.  Actual depot-to-zone distances within a CA will differ.

6. **Not approved sites**: The model selects optimal candidates under its
   assumptions.  Physical availability, zoning, and land use are not considered.

---

## Code Structure

```
src/escooter_opt/optimization/
├── pmedian.py        — IP formulation, solver, validation, I/O, CLI
├── pmedian_viz.py    — Visualisations (map, bars, histogram, elbow curve)
├── transport.py      — Phase 3 transportation LP
└── __init__.py       — Package exports (Phases 3 & 4)

tests/
└── test_pmedian.py   — 48 unit tests (all pass)

results/pmedian/
├── pmedian_solution.json
├── pmedian_summary.csv
├── selected_depots.csv
├── zone_assignments.csv
└── figures/
    ├── zone_depot_map.png
    ├── depot_summary_bars.png
    ├── assignment_distance_hist.png
    └── elbow_curve.png
```
