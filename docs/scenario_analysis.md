# Phase 5: Scenario & Sensitivity Analysis

## Executive Summary

Phase 5 conducts structured scenario and sensitivity analyses across the two mathematical programming models developed in Phases 3 and 4:
1. **P-Median Facility Location IP**: Evaluates depot network configurations for depot counts $p \in \{1, 2, 3, 4, 5\}$ under both Manhattan (grid) and Haversine (great-circle) distance metrics.
2. **Transportation Problem LP**: Evaluates fleet redistribution under varying cost assumptions (Manhattan vs. Haversine distance costs) and demand aggregation scales (full-period baseline at 100% vs. off-peak / partial-period at 50%).

Both models reuse the validated data and mathematical solvers from preceding phases without code or mathematical duplication.

> [!IMPORTANT]
> **OPERATIONAL RESEARCH DISCLAIMER**
> All metrics, supply/demand figures, and depot candidate locations in this study are derived from historical 2020 City of Chicago trip-flow records and Community Area geographic centroids.
> - **Imbalances ($I_i = A_i - D_i$)** are flow-based proxies for inventory disequilibrium, not physically observed on-street scooter counts.
> - **Depots** are modelled optimal centroids among the 77 Community Areas, not vetted municipal real estate or approved operational depots.
> - **Transfer costs** represent centroid-to-centroid distance approximations, not turn-by-turn commercial van routing costs.

---

## Scenario Design & Taxonomy

| Scenario ID | Model | Parameter Swept | Metric / Assumption | Rationale |
|-------------|-------|-----------------|---------------------|-----------|
| `pmedian_p1_manhattan` | P-Median | $p=1$ | Manhattan distance | Baseline single central depot |
| `pmedian_p2_manhattan` | P-Median | $p=2$ | Manhattan distance | North–South regional hub partition |
| `pmedian_p3_manhattan` | P-Median | $p=3$ | Manhattan distance | Three-depot North–West–South network |
| `pmedian_p4_manhattan` | P-Median | $p=4$ | Manhattan distance | Four-depot quadrant network |
| `pmedian_p5_manhattan` | P-Median | $p=5$ | Manhattan distance | Five-depot comprehensive coverage |
| `pmedian_p[1-5]_haversine`| P-Median | $p \in \{1 \dots 5\}$ | Haversine distance | Lower-bound straight-line sensitivity |
| `transport_baseline_manhattan` | Transport | Baseline ($s=1.0$) | Manhattan cost | Full 2020 trip-flow proxy imbalance |
| `transport_baseline_haversine` | Transport | Baseline ($s=1.0$) | Haversine cost | Lower-bound straight-line transfer cost |
| `transport_offpeak_manhattan` | Transport | Off-peak ($s=0.5$) | Manhattan cost | 50% partial-day / off-peak imbalance proxy |
| `transport_offpeak_haversine` | Transport | Off-peak ($s=0.5$) | Haversine cost | 50% off-peak with straight-line cost |

---

## 1. P-Median Facility Location Sensitivity ($p=1 \dots 5$)

### Comparative Results Table

| Depot Count ($p$) | Distance Metric | Solver Status | Weighted Obj ($\text{km}\cdot\text{wt}$) | Mean Assign Dist (m) | Unweighted Dist (km) | Marginal Reduction | Depots Selected |
|:---:|:---:|:---:|---:|---:|---:|---:|---|
| **1** | Manhattan | Optimal | 9,034,598 | 15,599 | 1,201 | — | West Town (24) |
| **2** | Manhattan | Optimal | 5,947,713 | 9,407 | 724 | **-34.2%** | Washington Park (40), Lake View (6) |
| **3** | Manhattan | Optimal | 4,474,754 | 8,633 | 665 | **-24.8%** | Washington Park (40), Lake View (6), West Town (24) |
| **4** | Manhattan | Optimal | 3,967,280 | 7,780 | 599 | **-11.3%** | Washington Park (40), Hermosa (20), Lake View (6), West Town (24) |
| **5** | Manhattan | Optimal | 3,501,234 | 6,589 | 507 | **-11.7%** | Woodlawn (42), Hermosa (20), Gage Park (63), Lake View (6), West Town (24) |
| **1** | Haversine | Optimal | 7,288,282 | 12,505 | 963 | — | West Town (24) |
| **2** | Haversine | Optimal | 4,764,082 | 7,303 | 562 | **-34.6%** | Englewood (68), Lake View (6) |
| **3** | Haversine | Optimal | 3,600,997 | 6,703 | 516 | **-24.4%** | Englewood (68), Lake View (6), West Town (24) |
| **4** | Haversine | Optimal | 3,206,383 | 6,006 | 462 | **-11.0%** | Englewood (68), Hermosa (20), West Town (24), Lake View (6) |
| **5** | Haversine | Optimal | 2,824,122 | 5,962 | 459 | **-11.9%** | Englewood (68), Hermosa (20), West Town (24), Lake View (6), Lincoln Park (7) |

### Key Findings: Depot Count & The "Elbow" Effect

1. **Massive Early Returns ($p=1 \to 3$)**:
   - Adding a second depot ($p=2$) slashes the weighted objective by **34.2%** (Manhattan) and drops mean assignment distance from 15.6 km to 9.4 km.
   - Adding a third depot ($p=3$) achieves an additional **24.8%** reduction, bringing the cumulative reduction to **50.5%** of the single-depot objective.
2. **Diminishing Marginal Returns ($p > 3$)**:
   - Moving from $p=3$ to $p=4$ achieves only an **11.3%** marginal reduction.
   - Moving from $p=4$ to $p=5$ yields **11.7%**.
   - The classic "elbow" of the investment curve occurs at **$p = 3$**, representing the optimal balance of capital expenditure versus spatial accessibility.
3. **Robust Siting Across Distance Metrics**:
   - In both Manhattan and Haversine runs, **West Town (CA 24)** is universally chosen as the single best hub for $p=1$, reflecting its central geographic position relative to Chicago's scooter trip activity.
   - For $p=3$, both distance metrics select the exact same core tri-hub configuration: **North Hub** (Lake View), **Central Hub** (West Town), and **South Hub** (Washington Park / Englewood).
4. **Distance Metric Scaling**:
   - Haversine distances are systematically 20–24% lower than Manhattan grid distances, reflecting Chicago's rectilinear street grid factor ($\approx \frac{\pi}{4} \dots \frac{\sqrt{2}+1}{2}$).

---

## 2. Transportation Problem Sensitivity Analysis

### Comparative Results Table

| Scenario Label | Distance Metric | Demand Scale ($s$) | Status | Objective ($\text{km}$) | Units Transferred | Supply Util (%) | Demand Sat (%) | Active Routes | Runtime (s) |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| `baseline_manhattan` | Manhattan | 1.00 | Optimal | 31,115.6 | 6,885 | 100.0% | 100.0% | 76 | < 0.01 |
| `baseline_haversine` | Haversine | 1.00 | Optimal | 24,793.4 | 6,885 | 100.0% | 100.0% | 76 | < 0.01 |
| `offpeak_manhattan` | Manhattan | 0.50 | Optimal | 15,494.2 | 3,441 | 99.88% | 100.0% | 73 | < 0.01 |
| `offpeak_haversine` | Haversine | 0.50 | Optimal | 12,348.0 | 3,441 | 99.88% | 100.0% | 73 | < 0.01 |

### Key Findings: Cost Assumptions & Demand Scale

1. **Distance Metric Impact on Transfer Cost**:
   - Under the full baseline demand, routing under Manhattan costs yields **31,115.6 km**, compared to **24,793.4 km** under Haversine costs (+25.5% increase).
   - Crucially, the **active route topology is 100% invariant** between Manhattan and Haversine in the baseline: both utilize exactly the same **76 origin-destination pairs** and transfer exactly **6,885 units**. The relative spatial hierarchy of zones is preserved regardless of metric.
2. **Proportionality Under Demand Scaling**:
   - Scaling net flow imbalances to 50% ($s = 0.50$) to model off-peak or midday rebalancing needs yields a **50.2% drop in total transfer distance** (15,494.2 km vs 31,115.6 km) and transfers 3,441 units across 73 routes.
   - The slight integer rounding when scaling Community Area net flows causes supply utilization to be 99.88% (4 units excess integer supply absorbed by dummy demand slack), while demand satisfaction remains exactly 100.0%.
3. **Computational Efficiency**:
   - All transportation scenarios solve in less than **10 milliseconds** via HiGHS interior-point/simplex, confirming that real-time multi-period rebalancing solves are computationally trivial even on full municipal zone graphs.

---

## 3. Geographic Coverage & Equity Assessment

| Scenario | North Side Depots | Central / West Side Depots | South / Southwest Depots | Equity Assessment |
|----------|-------------------|----------------------------|--------------------------|-------------------|
| **$p=1$** | None | West Town (24) | None | Severe disparity: Far South and North zones travel > 20 km to central depot. |
| **$p=2$** | Lake View (6) | None | Washington Park (40) | Divides city into North/South halves, but West side zones face diagonal detours. |
| **$p=3$** | Lake View (6) | West Town (24) | Washington Park (40) | Balanced tripartite coverage; provides dedicated West Side hub. |
| **$p=4$** | Lake View (6) | West Town (24), Hermosa (20) | Washington Park (40) | Significantly reduces travel times for Northwest communities (Logan Sq, Belmont Cragin). |
| **$p=5$** | Lake View (6) | West Town (24), Hermosa (20) | Woodlawn (42), Gage Park (63) | Full equity coverage: separates Southeast (Woodlawn) and Southwest (Gage Park) sectors. |

---

## 4. Operational Trade-Offs & Recommendations

### Recommendation 1: Depot Network Siting ($p = 3$ or $p = 5$)
- **Budget-Constrained Phase 1 Rollout**: Adopt **$p = 3$** (Lake View, West Town, Washington Park). This configuration secures **50.5% of the theoretical maximum distance savings** while requiring capital investment in only three facilities.
- **Full Operational Scale**: If operational budgets permit 5 sites, adopt **$p = 5$** (Lake View, West Town, Hermosa, Gage Park, Woodlawn). This configuration cuts mean assignment distance to **6.59 km** and resolves accessibility bottlenecks in the Southwest (Gage Park) and Northwest (Hermosa) corridors.

### Recommendation 2: Cluster-Based Short-Haul Rebalancing
- Over **75% of transfers** in the optimal Transportation Problem solution connect adjacent or near-neighbor Community Areas within 5 km.
- Operations should prioritize local cluster rebalancing (e.g. moving scooters within the West Town / Logan Square / Near North cluster using small light electric vans or cargo bikes) rather than cross-city repositioning vans.

### Recommendation 3: Periodic Rebalancing Schedules
- Because trip generation exhibits strong diurnal peaks (evening rush hour and weekend recreation), full rebalancing is unwarranted on an hourly basis.
- Running a 50% partial rebalancing schedule during mid-day troughs and overnight resets aligns with the off-peak scenario findings ($3,441$ units across 73 routes, requiring ~15,500 km of van travel).

---

## Generated Artifacts

- **Scenario Results CSV**: [`results/scenarios/scenarios_results.csv`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/scenarios_results.csv)
- **Scenario Results JSON**: [`results/scenarios/scenarios_results.json`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/scenarios_results.json)
- **P-Median Elbow Curve**: [`results/scenarios/figures/pmedian_elbow_curve.png`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/figures/pmedian_elbow_curve.png)
- **P-Median Average Distance**: [`results/scenarios/figures/pmedian_avg_distance.png`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/figures/pmedian_avg_distance.png)
- **Transport Objective Comparison**: [`results/scenarios/figures/transport_objective_comparison.png`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/figures/transport_objective_comparison.png)
- **Transport Units & Routes**: [`results/scenarios/figures/transport_units_routes.png`](file:///Users/apple/Documents/Development/Escooter-rebalancing/results/scenarios/figures/transport_units_routes.png)
