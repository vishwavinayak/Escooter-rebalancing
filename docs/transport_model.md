# Phase 3: Transportation Problem for E-Scooter Rebalancing

## Overview

Phase 3 formulates the estimated e-scooter rebalancing task as a classical
**Transportation Problem** (a special case of linear programming) and solves it
with the HiGHS solver via `scipy.optimize.linprog`.

> **⚠ IMPORTANT LIMITATION**
> All supply and demand quantities in this model are *estimated* from historical
> trip-flow accumulation (Phase 2 EDA).  They represent proxy imbalances inferred
> from where trips started and ended — **not** measured scooter inventory.
> The resulting rebalancing recommendations are heuristic in nature and must be
> validated against ground-truth field counts before operational use.

---

## Mathematical Formulation

### Sets

| Symbol | Description |
|--------|-------------|
| `S` | Set of **potential surplus zones** — zones with positive net flow `I_i > 0` |
| `D` | Set of **potential deficit zones** — zones with negative net flow `I_i < 0` |

### Inputs

| Symbol | Definition |
|--------|-----------|
| `I_i` | Net flow for zone `i`: `A_i − D_i` (arrivals minus departures) |
| `s_i = max(I_i, 0)` | Estimated surplus supply for zone `i ∈ S` |
| `d_j = max(−I_j, 0)` | Estimated deficit demand for zone `j ∈ D` |
| `c_{ij}` | Estimated transfer cost per unit from zone `i` to zone `j` (meters) |

The default cost `c_{ij}` is the **Manhattan grid distance** between zone
centroids, matching Chicago's orthogonal street network.  A haversine
(great-circle straight-line) variant is available for sensitivity testing.

Centroids are derived from observed coordinates in the dataset (Phase 2);
no coordinates are invented.

### Decision Variable

`x_{ij} ∈ ℤ⁺` — integer units transferred from surplus zone `i` to deficit zone `j`.

### Objective

```
minimize  ∑_{i∈S} ∑_{j∈D}  c_{ij} · x_{ij}
```

Minimise total estimated transfer cost (meter-units).

### Constraints

```
∑_j x_{ij} ≤ s_i      ∀ i ∈ S    (supply cannot exceed estimated surplus)
∑_i x_{ij} ≥ d_j      ∀ j ∈ D    (demand must be satisfied from estimated deficit)
x_{ij} ≥ 0, integer   ∀ i, j     (non-negative integer flows)
```

### Feasibility Balancing

If `∑ s_i ≠ ∑ d_j` (supply ≠ demand in aggregate), a **slack zone** is
inserted at zero cost to balance the problem before solving.  The slack is
stripped from the final output; original imbalance magnitudes are preserved.

### Integrality

The transportation polytope's constraint matrix is **totally dual integral**
(TDI).  When supply and demand vectors contain only integers (as here), the
LP relaxation optimum is necessarily integral.  The implementation solves the
continuous LP via HiGHS, rounds the result to the nearest integer, and
verifies feasibility post-rounding.

---

## Input Data (from Phase 2 EDA)

| File | Contents |
|------|----------|
| `results/eda/zone_flows_and_imbalance.csv` | Zone-level `departures`, `arrivals`, `net_flow`, `flow_category` |
| `results/eda/zone_distances_manhattan_meters.csv` | 77×77 pairwise Manhattan distances (meters) |
| `results/eda/zone_distances_haversine_meters.csv` | 77×77 pairwise haversine distances (meters) |

---

## Results (Manhattan distance, default run)

| Metric | Value |
|--------|-------|
| Surplus zones \|S\| | 46 |
| Deficit zones \|D\| | 31 |
| Total estimated supply | 6,885 units |
| Total estimated demand | 6,885 units |
| Solver status | **Optimal** (HiGHS Status 7) |
| Objective value | **31,115.6 km** (31,115,644 m) |
| Units transferred | 6,885 |
| Supply utilisation | 100.0% |
| Demand satisfaction | 100.0% |
| Non-zero transfer routes | 76 |
| Supply feasible (post-rounding) | True |
| Demand feasible (post-rounding) | True |

### Haversine Sensitivity Run

| Metric | Value |
|--------|-------|
| Objective value | **24,793.4 km** |
| Non-zero routes | 76 |

The haversine objective is ~20% lower, reflecting that straight-line distances
are shorter than Manhattan grid distances.  The route topology is similar.

### Top Allocations (Manhattan)

| Surplus Zone | Deficit Zone | Units | Dist (km) | Total Cost (km) |
|---|---|---:|---:|---:|
| AVONDALE | LINCOLN PARK | 333 | 7.32 | 2,436.7 |
| WEST RIDGE | UPTOWN | 304 | 7.57 | 2,302.4 |
| LOGAN SQUARE | LINCOLN PARK | 425 | 4.96 | 2,109.7 |
| NEAR NORTH SIDE | LINCOLN PARK | 563 | 3.48 | 1,960.9 |
| LINCOLN SQUARE | LINCOLN PARK | 161 | 8.95 | 1,441.0 |
| LINCOLN SQUARE | LAKE VIEW | 235 | 5.57 | 1,308.0 |
| EDGEWATER | LINCOLN PARK | 136 | 8.79 | 1,196.2 |
| NORTH CENTER | LAKE VIEW | 410 | 2.76 | 1,133.4 |
| ALBANY PARK | UPTOWN | 168 | 5.82 | 978.1 |
| DUNNING | PORTAGE PARK | 201 | 3.44 | 692.4 |

Lincoln Park and Lake View consistently appear as high-demand deficit zones,
consistent with Phase 2 EDA findings showing them as the highest-turnover
zones in the dataset.

---

## Output Files

All outputs are written to `results/transport/`:

| File | Description |
|------|-------------|
| `transport_solution.json` | Full result including all allocations, metadata, feasibility flags |
| `transport_summary.csv` | One-row summary of key metrics |
| `allocations_detail.csv` | One row per non-zero transfer (surplus_ca, deficit_ca, units, distances, costs) |
| `figures/allocation_bar_chart.png` | Horizontal bar chart of top-25 allocations by total cost |
| `figures/supply_demand_comparison.png` | Side-by-side supply vs dispatched / demand vs received charts |
| `figures/cost_matrix_heatmap.png` | Cost matrix heatmap for top 20 surplus × top 20 deficit zones with flow annotations |
| `figures/route_distance_distribution.png` | Distance distribution weighted by estimated units |

---

## Usage

```bash
# Default run (Manhattan distance)
uv run escooter-transport

# Haversine sensitivity
uv run escooter-transport --distance-method haversine --results-dir results/transport_haversine

# Custom output, top 30 routes, no figures
uv run escooter-transport \
  --eda-dir results/eda \
  --results-dir results/transport \
  --top-n 30 \
  --no-figures

# Help
uv run escooter-transport --help
```

---

## Known Limitations

1. **Estimated quantities only**: Supply `s_i` and demand `d_j` are historical
   trip-flow proxies, not observed scooter counts.  Real inventory may differ
   substantially.

2. **Centroid-level distances**: Transfer costs are computed between Community
   Area centroids.  Actual routing distances between depot and rebalancing
   locations within a zone may differ.

3. **No time dimension**: The model uses aggregate historical flows and does not
   capture intra-day or seasonal demand variability.

4. **Single period**: One rebalancing event is modelled.  In practice, multiple
   rebalancing rounds per day are needed.

5. **No vehicle capacity or fleet constraints**: The model assumes unlimited
   rebalancing vehicles.

6. **Intra-zone trips excluded**: Trips starting and ending in the same
   Community Area (68.96% of trips) do not contribute to inter-zone imbalance
   and are not modelled.

---

## Code Structure

```
src/escooter_opt/optimization/
├── __init__.py          — Package exports
├── transport.py         — LP formulation, solver, I/O, and CLI
└── visualize.py         — Result visualisations (matplotlib)

tests/
└── test_transport.py    — 37 unit tests (all pass)
```
